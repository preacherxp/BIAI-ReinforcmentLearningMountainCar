import numpy as np
import gym
env = gym.make('MountainCar-v0')
env.reset()


def QAlgorithm(learning, discount, epsilon, minEpsilon, episodes, greedyEpsilon):
    rewardList = []
    finalReward = []
    # Create space of states
    agentStates = np.round((env.observation_space.high -
                            env.observation_space.low) * np.array([10, 100]), 0).astype(int) + 1
    # Initialize Q table
    Q = np.zeros([agentStates[0], agentStates[1],
                  env.action_space.n])

    if greedyEpsilon:
        # Epsilon reduction
        reduction = (epsilon - minEpsilon)/episodes
        # Run learning algorithm
        for i in range(episodes):
            done = False
            currentReward, reward = 0, 0
            state = env.reset()
            # Create initial state
            initialState = np.round(
                (state - env.observation_space.low)*np.array([10, 100]), 0).astype(int)
            while done != True:
                # if i >= (episodes - 10):
                #     env.render()
                # Epsilon greedy approach
                if np.random.random_sample() < 1 - epsilon:
                    action = np.argmax(Q[initialState[0], initialState[1]])
                else:
                    action = np.random.randint(0, env.action_space.n)
                # Take next action
                observation, reward, done, info = env.step(action)
                # Create current state
                currentState = np.round((
                    observation - env.observation_space.low) * np.array([10, 100]), 0).astype(int)
                # Allow for terminal states
                if done and observation[0] >= 0.5:
                    Q[initialState[0], initialState[1], action] = reward
                # Adjust Q value for current state
                else:
                    calculatedDiscount = discount*np.max(Q[currentState[0],
                                                           currentState[1]]) - Q[initialState[0], initialState[1], action]
                    Q[initialState[0], initialState[1], action] += learning * \
                        (reward + calculatedDiscount)
                # Update variables
                currentReward += reward
                initialState = currentState
            # Decay epsilon
            if epsilon > minEpsilon:
                epsilon -= reduction
            # Track rewards
            rewardList.append(currentReward)
            averageReward = np.mean(rewardList)
            finalReward.append(averageReward)
            rewardList = []
            if averageReward != -200:
                print('Episode {} Average Reward: {}'.format(i+1, averageReward))
        env.close()
    else:
        for i in range(episodes):
            done = False
            currentReward, reward = 0, 0
            state = env.reset()
            # Create initial state
            initialState = np.round(
                (state - env.observation_space.low)*np.array([10, 100]), 0).astype(int)
            while done != True:
                # if i >= (episodes - 10):
                #     env.render()
                action = np.random.randint(0, env.action_space.n)
                # Take next action
                observation, reward, done, info = env.step(action)
                # Create current state
                currentState = np.round((
                    observation - env.observation_space.low) * np.array([10, 100]), 0).astype(int)
                # Allow for terminal states
                if done and observation[0] >= 0.5:
                    Q[initialState[0], initialState[1], action] = reward
                # Adjust Q value for current state
                else:
                    calculatedDiscount = discount*np.max(Q[currentState[0],
                                                           currentState[1]]) - Q[initialState[0], initialState[1], action]
                    Q[initialState[0], initialState[1], action] += learning * \
                        (reward + calculatedDiscount)
                # Update variables
                currentReward += reward

            # Track rewards
            rewardList.append(currentReward)
            averageReward = np.mean(rewardList)
            finalReward.append(averageReward)
            rewardList = []
            if averageReward != -200:
                print('Episode {} Average Reward: {}'.format(i+1, averageReward))
        env.close()
    return finalReward
