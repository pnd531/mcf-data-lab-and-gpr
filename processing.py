import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import scipy 
import glob
import re

#Extract the numeric index from filenames ---
def get_index(fname):
    # Matches both nbi_5.mat and nbi_1_25.mat
    m = re.search(r"nbi_([0-9_]+)\.mat", fname)
    s = m.group(1).replace("_", ".")   # convert "1_25" → "1.25"
    return float(s)

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

#dataset = scipy.io.loadmat("./data_tokamak/simulation_output_default.mat")
#plt.figure()
#plt.plot(get_output_data(dataset,"te0"), label="Default")
left = 55
right = 101
n_runs = 9

files = glob.glob("./data_tokamak/nbi_*.mat")

# --- 3. Sort files numerically ---
files_sorted = sorted(files, key=get_index)

# Number of runs becomes number of files
n_runs = len(files_sorted)




ne0 = np.zeros(n_runs)
ne0_err = np.zeros(n_runs)
ni0 = np.zeros(n_runs)
ni0_err = np.zeros(n_runs)
te0 = np.zeros(n_runs)
te0_err = np.zeros(n_runs)
ti0 = np.zeros(n_runs)
ti0_err = np.zeros(n_runs)
taue = np.zeros(n_runs)
taue_err = np.zeros(n_runs)
betan = np.zeros(n_runs)
betan_err = np.zeros(n_runs)
triple_product = np.zeros(n_runs)
triple_product_err = np.zeros(n_runs)


P_NBI = np.zeros(n_runs)

for i, fname in enumerate(files_sorted):
    #dataset = scipy.io.loadmat(f"./data_tokamak/nbi_{i*5}.mat")
    dataset = scipy.io.loadmat(fname)
    # Extract numeric NBI value from filename
    P_NBI[i] = get_index(fname)



    ne0[i],ne0_err[i] = get_average(dataset, left, right, "ne0")
    ni0[i],ni0_err[i] = get_average(dataset, left, right, "ni0")
    te0[i],te0_err[i] = get_average(dataset, left, right, "te0")

    tite, tite_err = get_average(dataset, left, right, "tite")
    ti0[i] = tite* te0[i]
    ti0_err[i] = np.sqrt( (tite_err/tite)**2 + (te0_err[i]/te0[i])**2 ) * ti0[i]


    taue[i],taue_err[i] = get_average(dataset, left, right, "taue")
    betan[i],betan_err[i] = get_average(dataset, left, right, "betan")

    triple_product[i] = ni0[i]*ti0[i]*taue[i]
    triple_product_err[i] = triple_product[i] * np.sqrt( (ni0_err[i]/ni0[i])**2 + (ti0_err[i]/ti0[i])**2 + (taue_err[i]/taue[i])**2 )
    #te0 = get_output_data(dataset,"te0")
    #tite = get_output_data(dataset,"tite")
    #ni0 = get_output_data(dataset,"ni0")
    #time = get_output_data(dataset,"temps")
    #time = time[left:right]
    #te0 = te0[left:right]

    #plt.figure()
    #plt.plot(time,te0, label=f"NBI {i*5}MW")
    #plt.title(f'Te0 vs Time for NBI={i*5}MW')

#print(np.shape(get_output_data(dataset,"te0")))
print(te0)


    
plt.figure()
plt.errorbar(P_NBI, ne0, yerr=ne0_err, fmt='o', label='ne0')
plt.title('Central Electron Density vs NBI Power')
plt.xlabel('NBI Power (MW)')
plt.ylabel('Central Electron Density (m^-3)')
plt.grid()

plt.figure()
plt.errorbar(P_NBI, ni0, yerr=ni0_err, fmt='o', label='ni0')
plt.title('Central Ion Density vs NBI Power')
plt.xlabel('NBI Power (MW)')
plt.ylabel('Central Ion Density (m^-3)')
plt.grid()

plt.figure()
plt.errorbar(P_NBI, te0, yerr=te0_err, fmt='o', label='te0')
plt.title('Central Electron Temperature vs NBI Power')
plt.xlabel('NBI Power (MW)')
plt.ylabel('Central Electron Temperature (eV)')
plt.grid()

plt.figure()
plt.errorbar(P_NBI, ti0, yerr=ti0_err, fmt='o', label='ti0')
plt.title('Central Ion Temperature vs NBI Power')
plt.xlabel('NBI Power (MW)')
plt.ylabel('Central Ion Temperature (eV)')      
plt.grid()          

plt.figure()        
plt.errorbar(P_NBI, betan, yerr=betan_err, fmt='o', label='betan')
plt.title('Plasma Beta vs NBI Power')
plt.xlabel('NBI Power (MW)')
plt.ylabel('Plasma Beta')
plt.grid()


plt.figure()
plt.errorbar(P_NBI, taue, yerr=taue_err, fmt='o', label='taue')
plt.title('Confinement Time vs NBI Power')
plt.xlabel('NBI Power (MW)')
plt.ylabel('Confinement Time (s)')
plt.grid()

plt.figure()
plt.errorbar(P_NBI, triple_product, yerr=triple_product_err, fmt='o', label='triple_product')
plt.title('Triple Product vs NBI Power')
plt.xlabel('NBI Power (MW)')
plt.ylabel('Triple Product (ne0*te0*taue)')
plt.grid()


dataset = scipy.io.loadmat("./data_tokamak/nbi_1_5.mat")

#list_subsections(dataset)
#list_indexes(dataset)
#print(get_average(dataset, 50, 100, "ni0"))
plt.figure()
plt.plot(get_output_data(dataset, "modeh"))
plt.title('Confinement Mode vs Time for NBI=1.5MW')
plt.xlabel('Time (s)')
plt.ylabel('Confinement Mode')
plt.grid()
plt.show()