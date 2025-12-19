# High Performance Protein Prediction with ESMFold
COMS-E6998 Final Project

## Overview

Here we detail our experimental setup in profiling and optimizing Meta's 2022 protein prediction model ESMFold. 

## Setup

To setup the experiments first run the setup.ipynb file in a PyTorch environment. For reference, all our experiments were run on a vast.ai PyTorch instance with a RTX 5090 GPU.

Then to replicate any of our experiments, simply run one of the following notebooks:
- baseline.ipynb (for the baseline model)
- compile.ipynb (for the model with torch.compile)
- batch.ipynb (for the microbatching experiments)

## Profiling

The profiling is done on wandb, and the experiments should update their results automatically on the wandb website. Additionally, a summary of key metrics is saved in the model.profiler class, and can be used to analyze the data locally. 

