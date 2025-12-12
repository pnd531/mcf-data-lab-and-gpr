import sys
import os
# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcf_data_lab.utils import *  # noqa: F403


data = read_simulation_folder('./data/data_ml/data_batch_ini', t_start=55, t_end=100)
X_train = data['X']
Y_train = data['triple']
sigma_n = data['triple_err']

X_train = scale_X(X_train)
Y_train, sigma_n, _, _ = scale_Y(Y_train, sigma_n)


print("The triple products of initial batch:",Y_train)

lengthscales = np.array([0.13818518, 0.47684491, 0.27157645, 0.42537776])
sigma_f = 0.6305

X_test = np.array([11.4099, 4.978, 3.9787, 0.9987])

mu, sigma_f = gp_predict(X_train, Y_train, X_test, lengthscales, sigma_f, sigma_n)
print(mu, sigma_f)