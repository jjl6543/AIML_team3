import numpy as np
from rps_game import select_winner
from naive_bayes_net import naive_bayes_strategy
import pandas as pd

data = np.load("naive_bayes_vs_stay.npy", allow_pickle=True)
# print(data)

computer_move = data[:,1]
human_move = data[:,0]

result = []
for i in range(data.shape[0]):
    r = select_winner(data[i,1], data[i,0])
    result.append(r)

Data = np.column_stack((data,result))
# print(Data)

# Data.tofile('bayes_shift_result.csv',sep=',',format='%10.5f')
pd.DataFrame(Data).to_csv('naivebayes_stay_result.csv')