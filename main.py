import sys
from QAlgorithm import QAlgorithm
from Plotting import plotting

episodes = int(sys.argv[1])
learningRate = float(sys.argv[2])
discountRate = float(sys.argv[3])
epsilon = float(sys.argv[4])
minEpsilon = float(sys.argv[5])
greedyEpsilon = False
if sys.argv[6] == "greedy":
    greedyEpsilon = True
else:
    greedyEpsilon = False

rewards = QAlgorithm(0.2, 0.9, 0.8, 0.2, episodes, greedyEpsilon)
plotting(rewards, episodes, learningRate, discountRate, sys.argv[6])

