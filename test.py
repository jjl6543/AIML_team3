import numpy as np

# data = np.genfromtxt("final_rps.csv", delimiter = ',', dtype = np.str)
# print(data)

# np.save("final_rps", data)
convert_file = np.load("data.npy",allow_pickle=True)
data = np.load("final_rps.npy",allow_pickle=True)
# print(convert_file)
print(data)