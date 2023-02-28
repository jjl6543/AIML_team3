''' 
This python program implements a Bayesian Strategy using the Pomegranate library
Official Documentation: https://pomegranate.readthedocs.io/en/latest/index.html
Github: https://github.com/jmschrei/pomegranate
'''

# Import required libraries
import numpy as np
from pomegranate import *

########### 
#data = np.load("data.npy", allow_pickle=True) ## Load historical data
data = np.load("final_rps.npy", allow_pickle=True) ## Load user data
# Third column Player B's move
#data = np.concatenate((data[:-1, :], data[1:,1].reshape(-1,1)), axis=1) ## Re-arrange the array such that column 1 contains previous human moves, column 2 contains previous computer moves and column 3 contains the next computer moves

# Third column Player A's move
data = np.concatenate((data[:-1, :], data[1:,0].reshape(-1,1)), axis=1) ## Re-arrange the array such that column 1 contains previous human moves, column 2 contains previous computer moves and column 3 contains the next computer moves

###########

# Assume human samples from a categorical distribution comprising of 3 outcomes, each of which are equally likely 
human = DiscreteDistribution({'rock': 1./3, 'paper': 1./3, 'scissors': 1./3})

# Assume computer samples from a categorical distribution comprising of 3 outcomes, each of which are equally likely 
computer = DiscreteDistribution({'rock': 1./3, 'paper': 1./3, 'scissors': 1./3})

# Prediction is dependent on both the human and computer moves. 
prediction = ConditionalProbabilityTable(
        [[ 'rock', 'rock', 'rock', 1./3 ],
         [ 'rock', 'rock', 'paper', 1./3 ],
         [ 'rock', 'rock', 'scissors', 1./3 ],
         [ 'rock', 'paper', 'rock', 1./3 ],
         [ 'rock', 'paper', 'paper', 1./3 ],
         [ 'rock', 'paper', 'scissors', 1./3 ],
         [ 'rock', 'scissors', 'rock', 1./3 ],
         [ 'rock', 'scissors', 'paper', 1./3 ],
         [ 'rock', 'scissors', 'scissors', 1./3 ],
         [ 'paper', 'rock', 'rock', 1./3 ],
         [ 'paper', 'rock', 'paper', 1./3 ],
         [ 'paper', 'rock', 'scissors', 1./3 ],
         [ 'paper', 'paper', 'rock', 1./3 ],
         [ 'paper', 'paper', 'paper', 1./3 ],
         [ 'paper', 'paper', 'scissors', 1./3 ],
         [ 'paper', 'scissors', 'rock', 1./3 ],
         [ 'paper', 'scissors', 'paper', 1./3 ],
         [ 'paper', 'scissors', 'scissors', 1./3 ],
         [ 'scissors', 'rock', 'rock', 1./3 ],
         [ 'scissors', 'rock', 'paper', 1./3 ],
         [ 'scissors', 'rock', 'scissors', 1./3 ],
         [ 'scissors', 'paper', 'rock', 1./3 ],
         [ 'scissors', 'paper', 'paper', 1./3 ],
         [ 'scissors', 'paper', 'scissors', 1./3 ],
         [ 'scissors', 'scissors', 'rock', 1./3 ],
         [ 'scissors', 'scissors', 'paper', 1./3 ],
         [ 'scissors', 'scissors', 'scissors', 1./3 ]], [human, computer]) 

# State objects hold both the distribution and the asscoiated node/state name. Both state and node mean the same in regard to the Bayesian Network
s1 = State(human, name="human")
s2 = State(computer, name="computer")
s3 = State(prediction, name="prediction")


# Create the Bayesian network object using a suitable name
model = BayesianNetwork("Rock Paper Scissors")

# Add the three states to the network 
model.add_states(s1, s2, s3)

# Add edges which represent conditional dependencies, where the prediction node is 
# conditionally dependent on its parent nodes (Prediction is dependent on both human and computer moves)
model.add_edge(s1, s3)
model.add_edge(s2, s3)

# Finalize the Bayesian Network
model.bake()
model.fit(data)


def bayes_strategy(A_move, B_move):
    # The following line returns the action that maximizes P(prediction of computer's next move| previous human_move, previous computer_move)
    # Here A = previous human move, B = previous computer move
    prediction = model.predict([[A_move, B_move, None]])
    print ("Argmax_Prediction:{}".format(prediction[-1][-1]))

    return prediction[-1][-1]
