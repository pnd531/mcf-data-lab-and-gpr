from utils import *

if __name__ == "__main__":
    ##########################################
    # MAIN LOOP FOR BAYESIAN OPTIMISATION
    ##########################################  
    ##########################################
    # 1. Load initial batch
    ##########################################
    data = read_simulation_folder('./data_ml/data_batch_ini')
    X_train = data['X']
    Y_train = data['triple']
    sigma_n = data['triple_err']
    

    # ---------------------------------------
    # Bayesian optimisation settings
    # ---------------------------------------
    n_rounds = 8
    batch_size = 10
    candidate_pool_size = 300   # How many LHS points to sample each round
    # Initial hyperparameters
    lengthscales = np.array([0.2, 0.2, 0.2, 0.2])
    sigma_f = 0.1

    # Initialise data recorders
    y_mean_record = []
    y_std_record = []
    hyperparams_record = []
    # ---------------------------------------
    # 2. Main loop over rounds
    # ---------------------------------------
    for round_idx in range(1, n_rounds + 1):

        # Call GPR routine (Routine B)
        X_next, lengthscales, sigma_f, Y_mean, Y_std = gaussian_process_regression(
            X_train, Y_train, lengthscales, sigma_f, sigma_n
            )

        # 3. Save next batch, hyperparameters, and Y scaling info
        # Modify these if you don't want to save to these paths or save as files
        savefile = f"./data_ml/batch_to_run/batch_{round_idx}.txt"
        np.savetxt(savefile, X_next)
        print(f"→ Saved next batch to: {savefile}")


        # Record hyperparameters and Y scaling info
        y_mean_record.append(Y_mean)
        y_std_record.append(Y_std)
        hyperparams_record.append((lengthscales, sigma_f))

        #savedata = {
        #    "lengthscales": lengthscales,
        #    "sigma_f": sigma_f,
        #    "Y_mean_ini": Y_mean,
        #    "Y_std_ini": Y_std
        #    }
        #np.savetxt(f"./data_ml/gpr_model/gpr_hyperparams_{round_idx}.txt", savedata)
        #print(f"Saved GPR hyperparameters to: ./data_ml/gpr_model/gpr_hyperparams_{round_idx}.txt")

        # -------------------------------------------------------
        # If reaches the final round, skip the rest and break the loop
        if round_idx == n_rounds:
            break

        # -------------------------------------------------------
        # 4. Wait for simulations to finish externally
        new_folder = f"./data_ml/data_batch_{round_idx}"
        print(f"Please run the simulation for this batch and place results in {new_folder}.")
        j=True
        while j:
            user_input = input("Type 'continue' to continue once the simulation is finished and data is ready, or 'exit' to quit:")
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'continue':
                j=False
        if user_input.lower() == 'exit':
                break
        # check if folder exists, warn if not. I am not sure if os module works in Linux.
        #if not os.path.exists(new_folder):
        #    print(f"Warning: {new_folder} does not exist yet. The script will continue anyway.")

        print(" → Loading new simulation data...")
        

        # 5. Load new data
        new_data = read_simulation_folder(new_folder)
        X_train_new = new_data['X']
        Y_train_new = new_data['triple']
        sigma_n_new = new_data['triple_err']

        # 6. Append new training data
        # Crucial! The memory of GPR is in the training data, not in the hyperparameters!
        # So we must keep the old training data and append the new data to it.
        # And use both for a new round of training
        X_train = np.vstack([X_train, X_train_new])
        Y_train = np.concatenate([Y_train, Y_train_new])
        sigma_n = np.concatenate([sigma_n, sigma_n_new])

        # ---------------------------------------
        # Next round

    print('GPR Bayesian optimisation loop finished.')
    ##########################################  
    # 7. Extract the maximum predicted triple product
    # ----------------------------- 
    x_best, mu_best_scaled, sigma_best_scaled = extract_maximum(
        scale_X(X_train), scale_Y(Y_train, sigma_n)[0], lengthscales, sigma_f, scale_Y(Y_train,sigma_n)[1], n_candidates=10000
        )
    # Descale predicted mean and uncertainty
    mu_best = descale_Y(mu_best_scaled, Y_mean, Y_std)
    sigma_best = descale_Y(sigma_best_scaled, Y_mean, Y_std)
    print("\n==============================")
    print("Maximum predicted triple product (from GP):")
    print(f"At input parameters: {x_best}")
    print(f"Predicted mean triple product: {mu_best:.3e}")
    print(f"Predicted uncertainty: {sigma_best:.3e}")
    print("==============================")
    print(Y_mean, Y_std)
    print(mu_best_scaled, sigma_best_scaled)

    print(lengthscales)
    print(sigma_f)