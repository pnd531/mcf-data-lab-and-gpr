from . import processing

dataset = scipy.io.loadmat("./data_tokamak/simulation_output_default.mat")

list_subsections(dataset)
list_indexes(dataset)
print(get_average(dataset, 50, 100, "te0"))
plt.plot(get_output_data(dataset, "te0"))
plt.show()