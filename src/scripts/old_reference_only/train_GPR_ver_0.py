# This code trains the GPR model using the processed simulation data.
# It reads the processed data, sets up the GPR model, and fits it to the data.

# There are available python packages for GPR, such as scikit-learn and GPy. Here I write a simple GPR implementation from scratch.

import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from scipy.stats import qmc
from scipy.optimize import minimize
import glob
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
        nbar = get_input_parameter(dataset, "nbar")[98]/10**20 # Convert to normlised density
        X_input[i] = [nbi, ip, b0, nbar]

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

# ------------------------------
# Acquisition function
# ------------------------------
def acquisition_ucb(mu, sigma, kappa=2.0):
    """Upper Confidence Bound (UCB) acquisition function."""
    return mu + kappa * sigma

# Optional: implement Expected Improvement (EI)
def acquisition_ei(mu, sigma, y_best):
    # TODO: implement EI acquisition function 
    return np.zeros_like(mu)

# ------------------------
# Kernel function
# ------------------------
def kernel_rbf_ard(X1, X2, lengthscales, sigma_f):
    """
    Radial Basis Function (RBF) kernel with Automatic Relevance Determination (ARD)
    X1: (N1, 4) first set of inputs
    X2: (N2, 4) second set of inputs
    lengthscales: (4,) lengthscale for each dimension
    sigma_f: scalar
    Returns: (N1, N2) kernel matrix
    """
    dists = np.sum(((X1[:, None, :] - X2[None, :, :]) / lengthscales)**2, axis=2)
    K = sigma_f**2 * np.exp(-0.5 * dists)
    return K

# ------------------------
# Log Marginal Likelihood
# ------------------------
def log_marginal_likelihood(X_train, y_train, lengthscales, sigma_f, sigma_n):
    """
    Computes the log marginal likelihood of a Gaussian Process with ARD RBF kernel.
    X_train: (N_train, 4)
    y_train: (N_train,)
    lengthscales: (4,)
    sigma_f: scalar (training parameter)
    sigma_n: scalar (error bars of triple product)
    
    Returns: scalar log marginal likelihood
    """
    K = kernel_rbf_ard(X_train, X_train, lengthscales, sigma_f)
    K += np.diag(sigma_n**2)  # sigma_n is a vector

    # compute p = -1/2 y.T K^-1 y - 1/2 log|K| - n/2 log(2pi)
    lml = -0.5 * y_train.T @ np.linalg.inv(K) @ y_train - 0.5 * np.linalg.slogdet(K)[1] - (len(y_train)/2) * np.log(2 * np.pi)

    return lml

# ------------------------
# GP Prediction
# ------------------------
def gp_predict(X_train, y_train, X_test, lengthscales, sigma_f, sigma_n):
    """
    Predict mean and variance of the GP at test points X_test.
    
    Inputs:
    - X_train: (N_train, 4) training inputs
    - y_train: (N_train,) training targets
    - X_test:  (N_test, 4) points to predict
    - lengthscales: (4,) ARD kernel lengthscales
    - sigma_f: scalar kernel amplitude
    - sigma_n: (N_train,) vector of noise std deviations for each y_train

    Returns:
    - mu_star: (N_test,) posterior mean
    - sigma_star: (N_test,) posterior std (not variance)
    """

    # 1. Compute kernel matrices
    K = kernel_rbf_ard(X_train, X_train, lengthscales, sigma_f)
    K += np.diag(sigma_n**2)  # Include noise
    K_s = kernel_rbf_ard(X_train, X_test, lengthscales, sigma_f)
    K_ss = kernel_rbf_ard(X_test, X_test, lengthscales, sigma_f)
    
    # We use a fancy way to do the inversion for numerical stability

    # 2. Cholesky decomposition for numerical stability
    L = np.linalg.cholesky(K + 1e-10*np.eye(len(X_train)))

    # 3. Solve for alpha = K^-1 y using Cholesky
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # 4. Compute posterior mean
    mu_star = K_s.T @ alpha

    # 5. Compute posterior variance
    v = np.linalg.solve(L, K_s)
    sigma_star2 = np.diag(K_ss) - np.sum(v**2, axis=0)
    sigma_star = np.sqrt(np.maximum(sigma_star2, 0))  # avoid tiny negatives

    return mu_star, sigma_star

# ------------------------
# Hyperparameter Optimization
# ------------------------
def minimise_neg_lml(X_train, y_train, sigma_n, params0=None):
    """
    Optimize kernel hyperparameters by minimizing negative log marginal likelihood.

    Inputs:
        X_train: (N_train, 4) training inputs
        y_train: (N_train,) training targets
        sigma_n: (N_train,) noise (error bars)
        params0: initial guess for [lengthscales_1..4, sigma_f]

    Returns:
        lengthscales_opt: optimised ARD lengthscales (4,)
        sigma_f_opt: optimised kernel amplitude
    """

    if params0 is None:
        params0 = np.ones(5)  # Initial guess [length1..4, sigma_f]

    def objective(params):
        lengthscales = params[:4]
        sigma_f = params[4]
        return -log_marginal_likelihood(X_train, y_train, lengthscales, sigma_f, sigma_n)

    bounds = [(1e-5, None)]*4 + [(1e-5, None)]
    res = minimize(objective, params0, bounds=bounds)

    lengthscales_opt = res.x[:4]
    sigma_f_opt = res.x[4]
    return lengthscales_opt, sigma_f_opt

# -----------------------------------------------------
# LHS sampling
# -----------------------------------------------------
def generate_candidates_lhs(N):
    # Parameter ranges
    # We regulate the lower range of the parameters to 0.5, 0.5, 0.5, and 0.1, respectively.
    # This is because we don't expect anything exciting to happen when any of these gets to zero.
    nbi_min, nbi_max = 0.5, 40.0
    Ip_min,  Ip_max  = 0.5, 5.0
    B0_min,  B0_max  = 0.5, 4.0
    nbar_min, nbar_max = 0.1, 1.0

    sampler = qmc.LatinHypercube(d=4)
    U = sampler.random(N)   # shape (N, 4)

    # qmc.scale expects 2D arrays, so reshape each column
    nbi  = qmc.scale(U[:, 0:1], nbi_min,  nbi_max).flatten()
    Ip   = qmc.scale(U[:, 1:2], Ip_min,   Ip_max).flatten()
    B0   = qmc.scale(U[:, 2:2+1], B0_min, B0_max).flatten()
    nbar = qmc.scale(U[:, 3:3+1], nbar_min, nbar_max).flatten()

    # Round continuous parameters to 4 decimals
    nbi = np.round(nbi, 4)
    Ip   = np.round(Ip, 4)
    B0   = np.round(B0, 4)
    nbar = np.round(nbar, 4)

    # Stack into (N, 4)
    X = np.column_stack([nbi, Ip, B0, nbar])
    return X

# -----------------------------
# Tell me where to sample next
# -----------------------------
def select_next_points(X_candidates, mu, sigma, n_points=10, kappa=2.0):
    """
    Select the next locations to sample based on UCB acquisition function.
    Instead of examining all possible locations, we consider only a set of candidate points.
    The X_candidates can be generated using LHS sampling (say 200 points), and we pick the top n_points among them.
    
    Inputs:
        X_candidates: (N_cand, 4) candidate input points
        mu: (N_cand,) predicted mean at candidates
        sigma: (N_cand,) predicted std at candidates
        n_points: number of points to select
        kappa: UCB coefficient

    Returns:
        X_next: (n_points, 4) next locations to simulate
    """
    # Compute UCB acquisition
    a_values = mu + kappa * sigma

    # Pick top n_points
    idx_next = np.argsort(a_values)[-n_points:]
    X_next = X_candidates[idx_next]
    return X_next

def scale_X(X):
    '''
    Scale X to 0 to 1
    
    :param X: Description
    '''
    # Parameter ranges
    param_mins = np.array([0.5,0.5,0.5,0.1])
    param_maxs = np.array([40,5,4,1])

    # Scale X
    scaled_X = (X - param_mins)/(param_maxs - param_mins)
    # Round to 6 decimals
    scaled_X = np.round(scaled_X, 6)
    return scaled_X

def descale_X(X_scaled):
    '''
    Descale X from 0-1 to original ranges
    
    :param X_scaled: Description
    '''
    # Parameter ranges
    param_mins = np.array([0.5,0.5,0.5,0.1])
    param_maxs = np.array([40,5,4,1])

    # Descale X
    X = X_scaled * (param_maxs - param_mins) + param_mins
    # Round to 4 decimals
    X = np.round(X, 4)
    return X


# The main body
# Assume that an initial batch has been sampled.
# Directories: ./data_ml/data_batch_ini for initial batch and
# ./data_ml/data_batch_i for the ith round
if __name__ == "__main__": 
    # Load the initial batch
    # If you choose to delete the files entry in the function, delete , files below
    data = read_simulation_folder('./data_ml/data_batch_ini')
    X_train = data['X']
    Y_train = data['triple']
    sigma_n = data['triple_err']
    # Scale X_train
    X_train = scale_X(X_train)

    # Initial optimisation of hyperparameters
    # Note that the function allows for an initial guess for the params. If we pass params0=None it automatically does this
    #lengthscales, sigma_f = minimise_neg_lml(X_train, y_train, sigma_n, params0=None)

    # I want to do 8 rounds of sampling with a batch size of 10.
    n_rounds = 8 
    batch_size = 10
    
    
    # Optimise hyperparameters using all available data
    lengthscales, sigma_f = minimise_neg_lml(X_train, Y_train, sigma_n, params0=None)
    #print("Optimised lengthscales:", lengthscales)
    #print("Optimised sigma_f:", sigma_f)
    
    # Generate the possible candidates for the next round
    # And we scale X_cand to 0.something-1
    X_cand = generate_candidates_lhs(200)
    X_cand_scaled = scale_X(X_cand)

    # Compute mu and sigma using the current best prediction
    mu, sigma = gp_predict(X_train, Y_train, X_cand_scaled, lengthscales, sigma_f, sigma_n)
    #print("Predicted mu at candidates:", mu)
    #print("Predicted sigma at candidates:", sigma)

    # Compute acquisition function values
    a = acquisition_ucb(mu, sigma)
    #print("Acquisition values at candidates:", a)

    # Determine where to run the next 10 simulations
    x_next = select_next_points(X_cand_scaled, mu, sigma, n_points=10, kappa=2.0)

    x_next = descale_X(x_next)
    print("Next points to sample (descaled):")
    print(x_next)