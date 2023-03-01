import numpy as np
from rps_game import get_ai_move
from naive_bayes_net import naive_bayes_strategy
data = np.load("data.npy", allow_pickle=True)
print(data)

#print(get_ai_move(round_winner='computer', round_human_move = 'rock', variable = 'win_shift_lose_shift'))
print(naive_bayes_strategy("rock", "scissors"))