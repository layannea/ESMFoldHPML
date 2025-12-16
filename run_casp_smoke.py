import os, re, gc, tarfile, urllib.request, subprocess
import numpy as np
import torch
from jax.tree_util import tree_map
from esmfold import ESMFold

OUT_DIR = "casp14_targets"
NUM_RECYCLES = 3
CHUNK_SIZE = 128
MAX_PROTEINS = 10

os.makedirs(OUT_DIR, exist_ok=True)

url = "https://predictioncenter.org/download_area/CASP14/targets/casp14.targets.T.public_11.29.2020.tar.gz"
tgz_path = os.path.join(OUT_DIR, os.path.basename(url))
if not os.path.exists(tgz_path):
    urllib.request.urlretrieve(url, tgz_path)

with tarfile.open(tgz_path, "r:gz") as tf:
    tf.extractall(OUT_DIR)

def read_fasta_to_dict(path):
    seqs, pdbs = {}, {}
    name, seq_chunks = None, []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None and os.path.isfile(os.path.join(OUT_DIR, name + ".pdb")):
                    seqs[name] = "".join(seq_chunks)
                    pdbs[name] = os.path.join(OUT_DIR, name + ".pdb")
                name = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if name is not None and os.path.isfile(os.path.join(OUT_DIR, name + ".pdb")):
            seqs[name] = "".join(seq_chunks)
            pdbs[name] = os.path.join(OUT_DIR, name + ".pdb")
    return seqs, pdbs

def compute_tm_score(pred_pdb, target_pdb, exe="./TMalign"):
    out = subprocess.check_output([exe, pred_pdb, target_pdb], text=True)
    m = re.search(r"TM-score\s*=\s*([0-9.]+)", out)
    if not m:
        raise RuntimeError("TM-score not found in TM-align output")
    return float(m.group(1))

print("Loading model...")
pretrained = torch.load("esmfold.model", weights_only=False)
model = ESMFold(esmfold_config=pretrained.cfg)
model.load_state_dict(pretrained.state_dict())
model.eval().cuda().requires_grad_(False)
del pretrained
gc.collect()
torch.cuda.empty_cache()

model.set_chunk_size(CHUNK_SIZE)

seqs, pdbs = read_fasta_to_dict("casp14_seq.txt")
items = list(seqs.items())[:MAX_PROTEINS]
print(f"Running smoke test on {len(items)} proteins...")

for k, seq in items:
    if len(seq) >= 2000:
        print("skip (too long):", k, len(seq))
        continue

    model.profiler.reset()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()

    out_gpu = model.infer(seq, num_recycles=NUM_RECYCLES, residue_index_offset=512)

    end.record(); torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    pdb_str = model.output_to_pdb(out_gpu)[0]
    out = tree_map(lambda x: x.cpu().numpy(), out_gpu)
    del out_gpu
    torch.cuda.empty_cache()

    ptm = out["ptm"][0]
    plddt_mean = out["plddt"][0,...,1].mean()

    pred_dir = f"pred_{k}"
    os.makedirs(pred_dir, exist_ok=True)
    pred_pdb = os.path.join(pred_dir, f"{k}.pdb")
    with open(pred_pdb, "w") as f:
        f.write(pdb_str)

    tm = compute_tm_score(pred_pdb, pdbs[k])

    print(f"{k:>6} L={len(seq):>4}  total_ms={total_ms:>8.1f}  peakGB={peak_gb:>5.2f}  pTM={ptm:>.3f}  pLDDT={plddt_mean:>.2f}  TM={tm:>.3f}")

print("Done.")
