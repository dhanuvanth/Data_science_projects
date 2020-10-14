import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import math

# get Dataset
dataSet = pd.read_csv(r"Reinforcement learning\Ads_CTR_Optimisation.csv")

#implementing RL
N = 10000
d = 10
ads_selected = []
number_of_selections = [0] * d
sum_of_selections = [0] * d
total_selection = 0
for n in range(0,N):
    ad = 0
    max_ads = 0
    for i in range(0,d):
        if(number_of_selections[i] > 0):
            avg_ads = sum_of_selections[i] / number_of_selections[i]
            confidence = math.sqrt((3/2) * (math.log(n + 1)/number_of_selections[i]))
            upper_bound = avg_ads + confidence
            # print(f"avg_ads = {avg_ads}")
            # print(f"confidence = {confidence}")
            # print(f"upper_bound = {upper_bound}")

        else:
            upper_bound = 1e400
        if(upper_bound > max_ads):
            max_ads = upper_bound
            ad = i
            # print(f"max_ads = {max_ads}")
            # print(f"ad = {ad}")

    ads_selected.append(ad)
    number_of_selections[ad] += 1
    reward = dataSet.values[n,ad]
    sum_of_selections[ad] += reward
    total_selection += reward
    # print(f"number_of_selections = {number_of_selections[ad]}")
    # print(f"sum_of_selections = {sum_of_selections[ad]}")
    # print(f"total_selection = {total_selection}")

#plot
plot.hist(ads_selected)
plot.title('UCP')
plot.xlabel('ads')
plot.ylabel('users')
plot.show()
