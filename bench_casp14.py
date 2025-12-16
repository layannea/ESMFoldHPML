import os, re, gc, csv, json, time, tarfile, urllib.request, argparse, subprocess
import numpy as np
import torch
from jax.tree_util import tree_map
from esmfold import ESMFold

CASP_OUT_DIR = "casp14_targets"
CASP_URL = "https://predictioncenter.org/download_area/CASP14/targets/casp14.targets.T.public_11.29.2020.tar.gz"

def ensure_casp14_targets():
    os.makedirs(CASP_OUT_DIR, exist_ok=True)
    tgz_path = os.path.join(CASP_OUT_DIR, os.path.basename(CASP_URL))
    if not os.path.exists(tgz_path):
        print("Downloading CASP14 targets tarball...")
        urllib.request.urlretrieve(CASP_URL, tgz_path)

    with tarfile.open(tgz_path, "r:gz") as tf:
        members = tf.getmembers()

        for m in members:
            if ".." in m.name or m.name.startswith("/"):
                raise RuntimeError(f"Unsafe member in tar: {m.name}")
        tf.extractall(CASP_OUT_DIR)

def read_fasta_to_dict(path):
    seqs, pdbs = {}, {}
    name, seq_chunks = None, []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if name is not None:
                    pdb_path = os.path.join(CASP_OUT_DIR, name + ".pdb")
                    if os.path.isfile(pdb_path):
                        seqs[name] = "".join(seq_chunks)
                        pdbs[name] = pdb_path
                name = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if name is not None:
            pdb_path = os.path.join(CASP_OUT_DIR, name + ".pdb")
            if os.path.isfile(pdb_path):
                seqs[name] = "".join(seq_chunks)
                pdbs[name] = pdb_path
    return seqs, pdbs

def compute_tm_score(pred_pdb, target_pdb, exe="./TMalign"):
    out = subprocess.check_output([exe, pred_pdb, target_pdb], text=True)
    m = re.search(r"TM-score\s*=\s*([0-9.]+)", out)
    if not m:
        raise RuntimeError("TM-score not found in TM-align output")
    return float(m.group(1))

def load_model(chunk_size):
    print("Loading model...")
    pretrained = torch.load("esmfold.model", weights_only=False)
    model = ESMFold(esmfold_config=pretrained.cfg)
    model.load_state_dict(pretrained.state_dict())
    model.eval().cuda().requires_grad_(False)
    del pretrained
    gc.collect()
    torch.cuda.empty_cache()
    model.set_chunk_size(chunk_size)
    return model

def time_infer(model, seqs, num_recycles, repeats, warmup, residue_index_offset=512):
    for _ in range(warmup):
        _ = model.infer(seqs, num_recycles=num_recycles, residue_index_offset=residue_index_offset)
    torch.cuda.synchronize()

    times_ms = []
    peak_alloc = []
    peak_reserved = []
    last_out = None

    for _ in range(repeats):
        model.profiler.reset()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()

        last_out = model.infer(seqs, num_recycles=num_recycles, residue_index_offset=residue_index_offset)

        end.record()
        torch.cuda.synchronize()

        times_ms.append(start.elapsed_time(end))
        peak_alloc.append(torch.cuda.max_memory_allocated())
        peak_reserved.append(torch.cuda.max_memory_reserved())

    return float(np.median(times_ms)), int(max(peak_alloc)), int(max(peak_reserved)), last_out

def bucket_filter(L, bucket):
    if bucket == "all": return True
    if bucket == "short": return L < 300
    if bucket == "mid":   return 300 <= L <= 1000
    if bucket == "long":  return L > 1000
    raise ValueError("bucket must be one of: all/short/mid/long")

def pack_microbatches(items, max_batch_residues, max_batch_size):
    items = sorted(items, key=lambda x: len(x[1]))
    batches = []
    cur = []
    cur_sum = 0
    cur_max = 0
    for k, s in items:
        L = len(s)
        new_max = max(cur_max, L)
        new_size = len(cur) + 1
        new_token_cost = new_max * new_size
        if cur and (new_token_cost > max_batch_residues or new_size > max_batch_size):
            batches.append(cur)
            cur = []
            cur_sum = 0
            cur_max = 0
        cur.append((k, s))
        cur_sum += L
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    return batches

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="all", choices=["all","short","mid","long"])
    ap.add_argument("--max_len", type=int, default=2000)
    ap.add_argument("--max_proteins", type=int, default=50)
    ap.add_argument("--num_recycles", type=int, default=3)
    ap.add_argument("--chunk_size", type=int, default=128) 
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--microbatch", action="store_true")
    ap.add_argument("--max_batch_residues", type=int, default=2500)
    ap.add_argument("--max_batch_size", type=int, default=8)
    ap.add_argument("--compute_tm", action="store_true")
    ap.add_argument("--out_csv", default="results.csv")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    ensure_casp14_targets()
    seqs, pdbs = read_fasta_to_dict("casp14_seq.txt")

    items = [(k, s) for k, s in seqs.items()
             if len(s) < args.max_len and bucket_filter(len(s), args.bucket)]
    items = items[:args.max_proteins]
    print(f"Selected {len(items)} proteins (bucket={args.bucket})")

    chunk_size = None if args.chunk_size == -1 else args.chunk_size
    model = load_model(chunk_size)
    need_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow([
                "tag","bucket","k","L","B","num_recycles","chunk_size",
                "time_ms_median","residues_per_s",
                "peak_alloc_gb","peak_reserved_gb",
                "ptm","plddt_mean","tm_score",
                "profiler_percents_json",
                "padding_ratio"
            ])

        if not args.microbatch:
            for k, seq in items:
                B = 1
                t_ms, peak_a, peak_r, out_gpu = time_infer(
                    model, seq, args.num_recycles, args.repeats, args.warmup
                )
                out = tree_map(lambda x: x.detach().float().cpu().numpy(), out_gpu)
                del out_gpu
                torch.cuda.empty_cache()
                gc.collect()

                ptm = float(out["ptm"][0])
                plddt_mean = float(out["plddt"][0, ..., 1].mean())
                tm = ""
                if args.compute_tm:
                    pred_dir = f"pred_{k}"
                    os.makedirs(pred_dir, exist_ok=True)
                    pred_pdb = os.path.join(pred_dir, f"{k}.pdb")
                    pdb_str = model.output_to_pdb(tree_map(lambda x: torch.from_numpy(x).cuda() if isinstance(x, np.ndarray) else x, out))[0] if False else None
                    #regenerate pdb from cpu->gpu conversion avoided; just do one quick infer-to-pdb:
                    out2 = model.infer(seq, num_recycles=args.num_recycles, residue_index_offset=512)
                    pdb_str = model.output_to_pdb(out2)[0]
                    del out2
                    with open(pred_pdb, "w") as pf:
                        pf.write(pdb_str)
                    tm = compute_tm_score(pred_pdb, pdbs[k])

                perc = model.profiler.percents()
                perc_json = json.dumps(perc)

                residues_per_s = (len(seq) / (t_ms / 1000.0))
                w.writerow([
                    args.tag, args.bucket, k, len(seq), B,
                    args.num_recycles, str(chunk_size),
                    f"{t_ms:.3f}", f"{residues_per_s:.3f}",
                    f"{peak_a/1e9:.4f}", f"{peak_r/1e9:.4f}",
                    f"{ptm:.4f}", f"{plddt_mean:.4f}", tm,
                    perc_json,
                    "0.0"
                ])
                f.flush()
                print(f"{k} L={len(seq)}  t={t_ms:.1f}ms  peak={peak_a/1e9:.2f}GB  pTM={ptm:.3f}  pLDDT={plddt_mean:.2f}  TM={tm}")

        else:
            batches = pack_microbatches(items, args.max_batch_residues, args.max_batch_size)
            print(f"Packed into {len(batches)} microbatches (max_batch_residues={args.max_batch_residues}, max_batch_size={args.max_batch_size})")

            for b in batches:
                keys = [k for k,_ in b]
                seq_list = [s for _,s in b]
                B = len(seq_list)
                maxL = max(len(s) for s in seq_list)
                sumL = sum(len(s) for s in seq_list)
                padding_ratio = (maxL*B - sumL) / (maxL*B)

                t_ms, peak_a, peak_r, out_gpu = time_infer(
                    model, seq_list, args.num_recycles, args.repeats, args.warmup
                )

                out = tree_map(lambda x: x.detach().float().cpu().numpy(), out_gpu)
                del out_gpu
                torch.cuda.empty_cache()
                gc.collect()

                residues_per_s = (sumL / (t_ms / 1000.0))
                ptm_mean = float(np.mean(out["ptm"]))
                plddt_mean = float(np.mean(out["plddt"][:, :, 1]))

                perc_json = json.dumps(model.profiler.percents())
                w.writerow([
                    args.tag, args.bucket, "+".join(keys), maxL, B,
                    args.num_recycles, str(chunk_size),
                    f"{t_ms:.3f}", f"{residues_per_s:.3f}",
                    f"{peak_a/1e9:.4f}", f"{peak_r/1e9:.4f}",
                    f"{ptm_mean:.4f}", f"{pldddt_mean:.4f}", "",
                    perc_json,
                    f"{padding_ratio:.4f}"
                ])
                f.flush()
                print(f"B={B} maxL={maxL} sumL={sumL} pad={padding_ratio:.2f}  t={t_ms:.1f}ms  thr={residues_per_s:.0f} res/s  peak={peak_a/1e9:.2f}GB")

if __name__ == "__main__":
    main()

