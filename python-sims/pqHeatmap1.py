from lpafunctions import *

N_values = [200, 400, 800]
community_values = [(i+1) for i in range(4)]
size_values = [33]
trial_values = [2]

for N in N_values:
    for comm in community_values:
        for size in size_values:
            for trials in trial_values:
                generatePQHeatmap(N, comm, size, trials)