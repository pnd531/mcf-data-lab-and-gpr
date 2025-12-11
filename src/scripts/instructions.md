# Treating GPR as a Black Box

This file contains instructions for training a GPR (Gaussian Process Regression) model. The principles behind GPR are explained in the main folder. Here, I provide only the instructions for running the code.  

---

## Getting Started

After cloning this Git repository, open the parent folder (`mcf-data-lab-and-gpr`). I have included my raw data in the repo.  

### Using the Provided Data

If you want to use my data, you do **not** need to run any simulations. In that case, simply run the Python file `main.py`.  

You will see a prompt like this:

---
→ Saved next batch to: ./data/data_ml/batch_to_run/batch_1.txt
Please run the simulation for this batch and place results in ./data/data_ml/data_batch_1.
Type 'continue' to continue once the simulation is finished and data is ready, or 'exit' to quit:
---


If you see this prompt, type `continue` (without quotes).  

- By default, the model trains for **8 rounds**, so you need to repeat this process 8 times.  
- You can interrupt at any time by typing `exit`. In this case, the model will stop training and return the predicted maximum triple product and the corresponding location.  

---

### Using Your Own Data / Running Your Own Simulations

If you want to use your own data or run custom simulations:  

1. Delete or move any pre-existing data in the `data/data_ml` directory.  
   - Delete the `.mat` files in each `data_batch_{i}` and `data_batch_ini` directory.  
   - **Do not delete the directories themselves**, as the model needs to save simulation results there.  

2. Workflow:  
   1. Use `LHS_sampling.py` to sample initial locations for simulation.  
   2. Run your simulations and save the `.mat` files in `data_batch_ini`.  
   3. Run `main.py`. The code will save a file called `batch_1.txt` in `data/data_ml/batch_to_run`.  
   4. This file lists interesting locations in parameter space. Run simulations for these locations and save the `.mat` files in `data_batch_1`.  
   5. Type `continue` in Python to proceed. Repeat this process for the desired number of rounds (default is 8).  
   6. After the final iteration, or if you manually type `exit`, the model will return the prediction.  

---

I am currently adding more routines to visualize and analyze the training results.
