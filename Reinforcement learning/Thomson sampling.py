import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import random

# get Dataset
dataSet = pd.read_csv(r"Reinforcement learning\Ads_CTR_Optimisation.csv")

#implementing RL
N = 10000
d = 10
ads_selected = []
no_of_rewards_1 = [0] * d
no_of_rewards_0 = [0] * d
total_rewards = 0
for n in range(0, N):
    ads = 0
    max_random = 0
    for i in range(0, d):
        random_bete = random.betavariate(no_of_rewards_1[i] + 1, no_of_rewards_0[i] + 1)
        if(random_bete > max_random):
            max_random = random_bete
            ads = i
    ads_selected.append(ads)
    rewards = dataSet.values[n ,ads]
    if rewards == 1:
        no_of_rewards_1[ads] += 1
    else:
        no_of_rewards_0[ads] += 1
    total_rewards += rewards
        

#plot
plot.hist(ads_selected)
plot.title('Thomson Sampling')
plot.xlabel('ads')
plot.ylabel('users')
plot.show()
