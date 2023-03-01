import numpy as np
from pomegranate import *

data = np.load("final_rps.npy", allow_pickle=True) ## Load user data
# Third column Player B's move
data = np.concatenate((data[:-1, :], data[1:,1].reshape(-1,1)), axis=1) ## Re-arrange the array such that column 1 contains previous human moves, column 2 contains previous computer moves and column 3 contains the next computer moves
# print(data)

model = BayesianNetwork.from_samples(data)
model.fit(data)
print ("Bayesian Network Summary: {}".format(model))
