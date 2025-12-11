
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from scipy.stats import qmc
from scipy.optimize import minimize
import glob
import time, os


# Parameter ranges
# Many functions will use these ranges for scaling and descaling
# I know this is a very bad practice, but I am lazy to pass these as function arguments everywhere.
# Modify these if you want different parameter ranges.

# Use these ranges if you limit nbar to 3e19 m^-3
# If you change ranges also change the normalisation in read_simulation_folder function.
#param_mins = np.array([0.5,0.1,0.5,0.1])
#param_maxs = np.array([40,2,4,3])

param_mins = np.array([0.5,0.5,0.5,0.1])
param_maxs = np.array([40,5,4,1])




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
def gp_predict(X_train, Y_train, X_test, lengthscales, sigma_f, sigma_n):
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
    # 0. scale inputs
    #X_train = scale_X(X_train)
    #X_test = scale_X(X_test)
    #Y_train, sigma_n, _, _ = scale_Y(Y_train, sigma_n)


    # 1. Compute kernel matrices
    K = kernel_rbf_ard(X_train, X_train, lengthscales, sigma_f)
    K += np.diag(sigma_n**2)  # Include noise
    K_s = kernel_rbf_ard(X_train, X_test, lengthscales, sigma_f)
    K_ss = kernel_rbf_ard(X_test, X_test, lengthscales, sigma_f)
    
    # We use a fancy way to do the inversion for numerical stability

    # 2. Cholesky decomposition for numerical stability
    L = np.linalg.cholesky(K + 1e-10*np.eye(len(X_train)))

    # 3. Solve for alpha = K^-1 y using Cholesky
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))

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
def minimise_neg_lml(X_train, y_train, sigma_n, params0):
    """
    Optimise kernel hyperparameters by minimizing negative log marginal likelihood.
    This is basically Routine A in my notes.
    Inputs:
        X_train: (N_train, 4) training inputs
        y_train: (N_train,) training targets
        sigma_n: (N_train,) noise (error bars)
        params0: [lengthscales_1..4, sigma_f]

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
    '''
    Generate N candidate points using Latin Hypercube Sampling (LHS)
    in the 4D input space defined by the parameter ranges at the beginning.
    '''

    # Read parameter ranges
    nbi_min, Ip_min, B0_min, nbar_min = param_mins
    nbi_max, Ip_max, B0_max, nbar_max = param_maxs

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
    a_values = acquisition_ucb(mu, sigma, kappa)

    # Pick top n_points
    idx_next = np.argsort(a_values)[-n_points:]
    X_next = X_candidates[idx_next]
    return X_next

def scale_X(X):
    '''
    Input: X: (N,4) array of input parameters in original ranges
    Output: scaled_X: (N,4) array of input parameters scaled to 0
    '''
    # Scale X
    scaled_X = (X - param_mins)/(param_maxs - param_mins)
    # Round to 6 decimals
    scaled_X = np.round(scaled_X, 6)
    return scaled_X

def descale_X(X_scaled):
    '''
    Input: X_scaled: (N,4) array of input parameters scaled to 0-1
    Output: X: (N,4) array of input parameters in original ranges
    '''
    # Descale X
    X = X_scaled * (param_maxs - param_mins) + param_mins
    # Round to 4 decimals
    X = np.round(X, 4)
    return X

def scale_Y(Y, Y_err):
    '''
    Scale Y. Subtlty here: we don't know the range of Y beforehand, so we have to compute mean and std from the training data.
    
    Input: Y: (N,) array of target values
    Output: Y_scaled: (N,) array of scaled target values
    '''
    Y_mean = np.mean(Y)
    Y_std  = np.std(Y)

    Y_scaled = (Y - Y_mean) / Y_std
    sigma_n_scaled = Y_err / Y_std
    return Y_scaled, sigma_n_scaled, Y_mean, Y_std

def descale_Y(Y_scaled, Y_mean, Y_std): 
    '''
    Descale Y from 0-1 to original ranges
    
    :param Y_scaled: Description
    '''
    Y = Y_scaled * Y_std + Y_mean
    return Y
# -----------------------------
# Extract maximum from GP
# -----------------------------
def extract_maximum(X_train, Y_train, lengthscales, sigma_f, sigma_n,
                    n_candidates=8000):
    """
    Finds the point in the input space that maximizes the GP posterior mean.
    Note that X_train and Y_train should be scaled already.

    Steps:
    1. Generate a large LHS candidate set
    2. Scale candidates (because GP was trained on scaled inputs)
    3. Predict mu(x) for all candidates
    4. Return the candidate with largest predicted mu
    """
    # 1. Generate candidates in original range
    X_cand = generate_candidates_lhs(n_candidates)

    # 2. Scale X_cand
    X_cand_scaled = scale_X(X_cand)

    # 3. Predict
    mu, sigma = gp_predict(X_train, Y_train,
                           X_cand_scaled, lengthscales, sigma_f, sigma_n)

    # 4. Find maximum mean prediction
    idx_best = np.argmax(mu)
    x_best_scaled = X_cand_scaled[idx_best]
    x_best = descale_X(x_best_scaled)
    return x_best, mu[idx_best], sigma[idx_best]



# -------------------------------
# Main GPR function (not used)
# -------------------------------
def gaussian_process_regression(_X_train, _Y_train, _lengthscales, _sigma_f, _sigma_n):
    '''
    Routine B as in my written notes.
    It takes in training data and hyperparameters, 
    and returns the trained GPR model and locations to sample next.

    The training data is assumed to be unscaled!!! 

    Inputs:
    - _X_train: (N_train, 4) training inputs
    - _Y_train: (N_train,) training targets
    - _lengthscales: (4,) ARD kernel lengthscales (can be initial guess or partly optimised)
    - _sigma_f: scalar kernel amplitude (can be initial guess or partly optimised)
    - _sigma_n: (N_train,) vector of noise std deviations for each y_train

    Outputs:
    - X_next: (n_points, 4) next locations to sample (descaled)
    - length_scales: optimised lengthscales after Routine A
    - sigma_f: optimised sigma_f after Routine A
    - Y_mean: mean of _Y_train (for descaling later)
    - Y_std: std of _Y_train (for descaling later)
    '''
    # 1. Scale training data
    _X_train = scale_X(_X_train)
    _Y_train, _sigma_n, Y_mean, Y_std = scale_Y(_Y_train, _sigma_n)

    # 2. Optimise hyperparameters (Routine A)
    length_scales, sigma_f = minimise_neg_lml(
        _X_train, _Y_train, _sigma_n,
        params0=np.concatenate([_lengthscales, [_sigma_f]])
    )

    # 3. Generate candidate points
    X_cand = generate_candidates_lhs(300)
    X_cand_scaled = scale_X(X_cand)

    # 4. Predict GP mean & variance at candidate points
    mu, sigma = gp_predict(
        _X_train, _Y_train, X_cand_scaled,
        length_scales, sigma_f, _sigma_n
    )

    # 5. Use acquisition function choose next points
    X_next_scaled = select_next_points(
        X_cand_scaled, mu, sigma,
        n_points=10,
        kappa=2.0
    )

    # 6. Descale X_next
    X_next = descale_X(X_next_scaled)

    return X_next, length_scales, sigma_f, Y_mean, Y_std


