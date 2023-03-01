import numpy as np
from pomegranate import *



def naive_bayes_strategy(A,B):
    data = np.load("data.npy", allow_pickle=True) ## Load historical data
    #data = np.load("final_rps.npy", allow_pickle=True) ## Load user data
    # data = np.concatenate((data[:-1, :], data[1:,0].reshape(-1,0)), axis=1) ## Re-arrange the array such that column 1 contains previous human moves, column 2 contains previous computer moves and column 3 contains the next computer moves
    data = np.concatenate((data[:-1, :], data[1:,0].reshape(-1,1)), axis=1) ## Re-arrange the array such that column 1 contains previous human moves, column 2 contains previous computer moves and column 3 contains the next computer moves


    #print(data)
    poss = ['rock', 'paper', 'scissors']

    # Given Moves of Human and computer A, B

    prb = np.zeros((len(poss), 3))
    # P(A,Y)
    col_valuesA = np.array([data[:, 0], data[:, 2]]).T
    col_valuesB = np.array([data[:,1], data[:,2]]).T


    for elem in poss:
        
        #for P(A,Y)
        mtc = [A, elem]
        matching_rows = (col_valuesA[:,0] == mtc[0])& (col_valuesA[:,1] == mtc[1])

        freq = np.count_nonzero(matching_rows)
        aandy = freq/data.shape[0]
        
        #for P(B,Y)
        mtc = [B, elem]
        matching_rows = (col_valuesB[:,0] == mtc[0])& (col_valuesB[:,1] == mtc[1])

        freq = np.count_nonzero(matching_rows)
        bandy = freq/data.shape[0]

        
        

        # P(Y)
        matching_rows = (col_valuesA[:,1]== elem)

        freq = np.count_nonzero(matching_rows)
        y = freq/len(col_valuesA[:,1])
        
        
        #print(y)
        # P(A|Y)

        prb[poss.index(elem), 0] = aandy/y
        
        # P(B|Y)

        prb[poss.index(elem), 1] = bandy/y
        
        # P(Y)
        prb[poss.index(elem), 2] = y

    print(prb)

    P_YAB = prb[:,0]* prb[:,1]* prb[:,2]
    #print(P_YAB)


    return poss[np.argmax(P_YAB)]
