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

---

