import matplotlib.pyplot as plt
import numpy as np

def plotting(rewards, episodes, learningRate, discountRate, greedy):
    plt.plot((np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (higher is better)')
    plt.title('learning rate: ' + str(learningRate) +
              ' ' + 'discountRate: ' + str(discountRate))
    plt.savefig('data' + str(episodes) + '_' +
                str(learningRate) + '_' + str(discountRate) + '_' + str(greedy) + '.jpg')
    plt.close()
