# MCF Data Lab Tokamak Simulator – Gaussian Process Regression

## Repository Overview

This repository contains code for the MCF Data Lab Tokamak Simulator. The objective is to apply Gaussian Process Regression (GPR) to maximise the triple product by varying four input parameters. Simulations are performed using METIS, which is difficult to automate reliably. Because no stable wrapper for METIS is available, all simulation runs must be executed manually.

## Conceptual Overview

The repository provides routines for training and updating a GPR model. Conceptually, GPR operates as follows:

1. Provide initial data to the model.
2. Train the model and obtain predictions of regions of interest in parameter space, balancing uncertainty and expected maxima.
3. Acquire new simulation data in these regions.
4. Feed new data back into the model.
5. Repeat until a suitable convergence criterion is achieved.

Due to METIS being non-automatable, each new simulation must be run manually, meaning the training loop requires continuous human intervention.

## Repository Structure

This repository is under development. The planned structure is:

```
.
├── src/
│   ├── mcf_data_lab/        # Core modules (import only)
│   └── scripts/             # Executable scripts
├── data/
│   ├── data_ml/             # Data for GPR training
│   └── data_tokamak/        # Data for Data Lab Task 1
├── results/                 # Plots and predictions
└── README.md
```



## Usage Instructions

If you are only interested in final outputs, refer to the **results/** directory (when available).

If you want to replicate the work or experiment with GPR:

1. Download the **src/** and **data/** directories (or clone the entire repository).
2. Run **main.py** to train the model.
3. For detailed instructions, consult **instruction.txt** in `src/scripts` (in preparation).

Most utility functions called in `main.py` are defined in `utils.py`.

A document explaining the underlying principles of Gaussian Process Regression may be added later as **GPR Explained** at the repository root.


# Version Description

## V1.01
The initial github version. I was able to run the code locally, but the restructuring may mess the module importing in the main code. I have tested the model with an initial data batch (included in data/data_ml/data_batch_ini). It seemed to work (but with one data batch there is no way for the model to converge, so additional test is needed). To start the training, run main.py in src/scripts/. 
Todo: test whether the restructured code works; train the model with several data batches to see if it converges; add instructions and results. 

## V1.02
Stable version. Ready to run. I have fixed the file paths issue. Now, the main.py can safely import functions from utils.py. An initial data batch is tested and the model gives reasonable output (note that we do not expect the model to converge with only 20 randonly-drawn data points). 
TODO: Train the model with more data points

## V1.03
The code proved to be incredibly buggy. I tried to fix a few bugs and add an alternative kernel. I also took all the shared data and added it to the initial data batch. The prediction given by the model is very bad. This is the end of MCF data lab. I will continue working on this. For now, I think the issue is that the physics is very nonlinear. In the L-H transition region, the 'true' function is very unsmooth. GPR doesn't work well with unsmooth functions. There are some ways to improve the behaviour such as using a kernel that is more tolerable with unsmooth functions and placing bounds in hyperparameters. Getting rid of noisy data points may also help. As a starting point, I will reduce the code to 1D version to investigate how GPR fits a 1D unsmooth function.

---

