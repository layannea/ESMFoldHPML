# High Performance Protein Prediction with ESMFold
COMS-E6998 Final Project

## Overview

Here we detail our experimental setup in profiling and optimizing Meta's 2022 protein prediction model ESMFold. 

## Setup

To setup the experiments first run the setup.ipynb file in a PyTorch environment. For reference, all our experiments were run on a vast.ai PyTorch instance with a H100 PCIE GPU.

Then to replicate any of our experiments, simply run one of the following notebooks:
- baseline.ipynb (for the baseline model)
- compile.ipynb (for the model with torch.compile)

## Profiling

The profiling is done on wandb, and the experiments should update their results automatically on the wandb website. Additionally, a summary of key metrics is saved in the model.profiler class, and can be used to analyze the data locally.
Link: https://wandb.ai/sun-stones-columbia-university/esmfold-experiments?nw=nwusersunstones

## Results
[Alt text](.assets/graph1.png)
Runtime breakdown for baseline forward pass.

[Alt text](.assets/cpucuda.png)
Comparing stages of the forward pass in CPU and GPU time.

[Alt text](.assets/comp.png)
Comparing the percent runtimes between a short and long protein sequence.
