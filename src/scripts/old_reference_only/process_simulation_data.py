# This code processes simulation data for machine learning applications.
# It reads .mat files, extracts relevant parameters, computes averages and errors,
# and prepares datasets for training.

# The idea is: we have a set of simulation output files from METIS. The code reads these files, extract the input 
# parameters, compute the triple product with error bars, and store them in arrays to be fed into the GPR model.

# The naming of the file doesn't matter actually, as long as we can read the .mat files.

# The naming convention of mat files is nbi_{}_ip_{}_b0_{}_n_{}.mat
# Where the {} are replaced with the actual parameter values. For decimal values, the dot is replaced with '_'.
# e.g. nbi_10_ip_3_5_b0_2_2_n_0_5.mat
# n is the normalised density (nbar). The unit is 10^20 m^-3.
# If I am right, n should be between 0 and 1.

import numpy as np

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import scipy 
import glob
import re


def list_subsections(dataset):
    # Prints the subsections in the dataset
    print(dataset["post"].dtype.names)

def list_indexes(dataset, subsection="zerod"):
    # Prints the indexes for a given subsection ("zerod" by default)
    print("Indexes in subsection " + subsection + ":")
    print(dataset["post"]['zerod'][0][0].dtype.names)

def get_output_data(dataset, index):
    # Gets a variable from the dataset
    var = dataset["post"]["zerod"][0][0][index][0][0].T[0]
    return var

def get_average(dataset, start, end, index):
    # Returns the mean and standard deviation for the dataset
    var = get_output_data(dataset, index)
    return (np.mean(var[start:end]), np.std(var[start:end]))

def get_input_parameter(dataset, index):
    # Returns the input parameters of the Metis simulation. Some acceptable indexes are:
    # ip, nbar, pnbi, b0

    if index in dataset["post"]["z0dinput"][0][0]["cons"][0][0][0].dtype.names:
        var = dataset["post"]["z0dinput"][0][0]["cons"][0][0][0][index][0].T[0]
    elif index in dataset["post"]["z0dinput"][0][0]["geo"][0][0].dtype.names:
        var = dataset["post"]["z0dinput"][0][0]["geo"][0][0][index][0][0].T[0]
    else:
        print("Whoops! Try again.")
    return var

# -----------------------------
# MAIN LOADER
# -----------------------------
def read_simulation_folder(folder, t_start=55, t_end=101):
    # Assume you have N mat files in the folder
    """
    Reads all METIS .mat files from a folder and returns:
      X_input: (N, 4) matrix of [NBI, Ip, B0, nbar]
      triple_prod:      (N,)
      triple_prod_err:  (N,)
      plus all intermediate quantities if needed
    """

    # Assume that you name the mat files as nbi_*.mat. It doesn't matter what the other parameters are in the filename, 
    # as long as you start with nbi_.
    files = sorted(glob.glob(f"{folder}/nbi_*.mat"))
    N = len(files)
    # Raise error if no files found
    if N == 0:
        raise RuntimeError("No METIS files found.")

    # Initialise arrays
    # Input parameters
    X_input = np.zeros((N, 4))
    # Triple product outputs
    triple = np.zeros(N)
    triple_err = np.zeros(N)
    # Intermediate variables
    ne0 = np.zeros(N)
    ne0_err = np.zeros(N)
    ni0 = np.zeros(N)
    ni0_err = np.zeros(N)
    te0 = np.zeros(N)
    te0_err = np.zeros(N)
    ti0 = np.zeros(N)
    ti0_err = np.zeros(N)
    taue = np.zeros(N)
    taue_err = np.zeros(N)

    # Loop over simulations
    for i, fname in enumerate(files):
        dataset = scipy.io.loadmat(fname)

        # -----------------------------
        # Read input parameters
        # -----------------------------
        # For NBI,ip,and b0, I tested that the index 60 (and many others) corresponds to the value we set as the input parameter
        # For nbar, I found that index 98 corresponds to the value we set as the input parameter
        nbi  = get_input_parameter(dataset, "pnbi")[55] / 10**6 # Convert from W to MW
        ip   = get_input_parameter(dataset, "ip")[60]/10**6    # Convert from A to MA
        b0   = get_input_parameter(dataset, "b0")[60] #Already in T
        nbar = get_input_parameter(dataset, "nbar")[98]/10**19 # Convert to normlised density
        X_input[i] = [nbi, ip, b0, nbar]
        #print(nbar)
        # -----------------------------
        # Extract averaged quantities
        # -----------------------------
        ne0[i], ne0_err[i] = get_average(dataset, t_start, t_end, "ne0")
        ni0[i], ni0_err[i] = get_average(dataset, t_start, t_end, "ni0")
        te0[i], te0_err[i] = get_average(dataset, t_start, t_end, "te0")

        # ti0 = tite * te0
        tite, tite_err = get_average(dataset, t_start, t_end, "tite")
        ti0[i] = tite * te0[i]
        ti0_err[i] = ti0[i] * np.sqrt((tite_err / tite)**2 + (te0_err[i] / te0[i])**2)

        taue[i], taue_err[i] = get_average(dataset, t_start, t_end, "taue")

        # Triple product and error propagation
        # As the triple product is ni0 * ti0 * taue, the error propagates in quadrature
        triple[i] = ni0[i] * ti0[i] * taue[i]
        triple_err[i] = triple[i] * np.sqrt(
            (ni0_err[i]/ni0[i])**2 +
            (ti0_err[i]/ti0[i])**2 +
            (taue_err[i]/taue[i])**2
        )
    # Uncomment if you want to return intermediate quantities too
    # The final line returns a list of the file paths corresponding to the simulations that were read. 
    # Useful to check what files are read. Comment the line if you don't want this feature.
    return {
        "X": X_input,
        "triple": triple,
        "triple_err": triple_err,
        #"ne0": (ne0, ne0_err),
        #"ni0": (ni0, ni0_err),
        #"te0": (te0, te0_err),
        #"ti0": (ti0, ti0_err),
        #"taue": (taue, taue_err),
        "files": files
    }








# Example usage

data = read_simulation_folder("./data/data_ml/data_batch_ini", t_start=55, t_end=100)

print(data["triple"])

print(data["X"].shape)         # (N, 4)
print(data["triple"].shape)    # (N,)
print(data["triple_err"].shape)  # (N,)
print(data["files"][:5])       # preview first 5 files' names
