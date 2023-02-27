import numpy as np
import csv
from itertools import islice

data = []

with open('final_rps.csv') as csv_file: 
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in islice(csv_reader, 1, None):
            data.append(row)
            
data = np.array(data)


np.save("final_rps", data)
convert_file = np.load("final_rps.npy",allow_pickle=True)

# Third column Player B's move
# data1 = np.concatenate((convert_file[:-1, :], convert_file[1:,1].reshape(-1,1)), axis=1) ## Re-arrange the array such that column 1 contains previous human moves, column 2 contains previous computer moves and column 3 contains the next computer moves


# Third column Player A's move
data1 = np.concatenate((convert_file[:-1, :], convert_file[1:,0].reshape(-1,1)), axis=1) ## Re-arrange the array such that column 1 contains previous human moves, column 2 contains previous computer moves and column 3 contains the next computer moves


print(data1)
