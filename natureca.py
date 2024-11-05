# imports and stop sci notation
import numpy as np
import matplotlib.pyplot as plt
import time
np.set_printoptions(suppress=True)

# ACO Implementation using matrices to represent index of each item in a specific bin
# with accompanying pheromones for each step in the path from S to E

# k = num items
# b = num bins
# returns the pheromones graph
# Step 1: Randomly distribute small amounts of pheromone (between 0 and 1)
# for exploration in initial iteration and start of each trial (1-5)
def make_graph(k, b):
    pgraph = np.random.uniform(0, 1, (k, b))
    return pgraph

# p = number of ants
# g = construction graph with pheromones
# returns the generated set of paths
# Step 2: Generate a set of p ant paths from S to E (where p is a variable and specified below). 
def gen_paths(p, g):
    # get shape of construction graph
    k, b = g.shape
    # initialise paths as empty
    paths = np.zeros((p, k), dtype=int)

    # for each ant and item of ant
    for ant in range(p):
        for i in range(k):
            # get pheromone values for item across b bins
            pheromones = g[i, :]
            # calculate probability from pheromone levels
            # random but biased by pheromone levela
            probability = pheromones / pheromones.sum()
            # choose bin at random based on pheromone influence
            chosen_bin = np.random.choice(np.arange(b), p=probability)
            paths[ant, i] = chosen_bin
    return paths

# path = one ant path
# b = number of bins
# returns the fitness of a path
# Step 3a. Compute the fitness for the update of the ant path and add bin weight
def calc_fit(path, b):
    k = len(path)
    # set item weights based on b bins for BPP1 and BPP2
    if b == 50:
        # BPP2
        # item weight = [i^2/2]
        iw = np.array([(i**2) / 2 for i in range(1, k + 1)]) 
    else:
        # BPP1
        # item weight = i (1, 2, ... 500)
        iw = np.arange(1, k + 1)
    # initialize bin weights and add items based on path
    bw = np.zeros(b)
    for i in range(k):
        # set bin index to item in path
        bindex = path[i]
        # add item weight to bin weight at bin index
        bw[bindex] += iw[i]
    # calc difference between max and min bin weights for fitness
    fitness = bw.max() - bw.min()
    return fitness

# g = construction graph
# paths = ant paths
# b = number of bins
# returns the updated pheromone construction graph
# Step 3b. Update the pheromone in your pheromone table for each ant’s path according to its fitness, computed above
def update_pher(g, paths, b):
    fit_list = []
    Q = 100
    # compute fitness iteratively
    # get fitness of each ant, add to total list
    for path in paths:
        #calculate fitness and add item weights to bin
        f = calc_fit(path, b)
        fit_list.append(f)
    # calc pheromone update 
    #find path indices
    for path, f in zip(paths, fit_list):
        # prevent division by zero, update value = 100/fitness
        update = Q/f if f > 0 else 0.001
        # for each index of items and bins, update the path using the update value
        for ii, bi in enumerate(path):
            # add update to matrix
            g[ii, bi] += update

    # return updated graph
    return g

# g = construction graph with pheromoens
# paths = ant paths
# e = evaporation rate
# returns the evaporated pheromone construction graph
# Step 4. Evaporate the pheromone for all links in the graph.
def evap_pher(g, paths, e):
    # initialise evaporation matrix to 1s for multiplication
    e_matrix = np.ones_like(g)

    # find path indices
    for path in paths:
        for ii, bi in enumerate(path):
            # place evaporation factor along paths in empty matrix
            e_matrix[ii, bi] = e
        
    # apply evaporation at every ant path through matrix multiplication
    g *= e_matrix

    return g

# path = ant path
# b = number of bins
# Plot bin weights of specific path (bar chart)
def plot_bins(path, b):
    k = len(path)
    if b == 50:
        # BPP2
        # item weight = [i^2/2]
        iw = np.array([(i**2) / 2 for i in range(1, k + 1)]) 
    else:
        # BPP1
        # item weight = i (1, 2, ... 500)
        iw = np.arange(1, k + 1)
    bw = np.zeros(b)
     
     # add item weight to bin weights
    for i in range(k):
        bw[path[i]] += iw[i]

    print(f"Final Bin Weight Totals: {bw}")

    plt.bar(range(1, b + 1), bw)
    if(b == 10):
        plt.title(f'Best Bin Weight Distribution for BPP1')
    else:
        plt.title('Best Bin Weight Distribution for BPP2')
    plt.xlabel('Bins')
    plt.ylabel('Total Weight')
    plt.ylim(0, np.max(bw) + (2/3)*np.max(bw))
    plt.show()

# p = num ants (paths)
# e = evaporation rate
# k = num items
# b = num bins
# evals = number of fitness evaluations per trial
# num_trials = number of trials
# returns the best fitness from all trials, purely for testing
# main running function for use, will run the full implementation
# using all generation and computing functions, alongside individual computation
# to print clear results and plot graph at the end
def run_trial(p, e, k, b, evals=10000, num_trials=5):
    # intiialise accumulative trial variables
    best_overall_path = None
    best_fitness_across_trials = float('inf')
    all_trials_fitness_data = []  
    all_trial_avg_fits = []
    all_trial_best_fits = []

    # 5 trials
    for trial in range(num_trials):
        # BPP1 or 2
        if(b==10):
            print(f"\nBPP1:\n {p} ants, {e} evaporation rate \nStarting Trial {trial+1}")
        else:
            print(f"\nBPP2:\n {p} ants, {e} evaporation rate \nStarting Trial {trial+1}")
        # initialize random seed for each trial (10,000 evals per)
        np.random.seed(int(time.time()))
        # randomise pheromones (0-1) across graph
        g1 = make_graph(k, b)
        print("Current Pheromones:")
        print(g1)

        # initialise evaluation variables
        trial_fits = []
        total_evals = 0
        overall_bf = float('inf')
        overall_best_path = None
        
        # ensure all 10,000 evals complete
        while total_evals < evals:
            #generate paths
            ant_paths = gen_paths(p, g1)
            fitnesses = []
            bf = float('inf')
            best_path = None
            
            # iterate through paths to get fitnesses
            for path in ant_paths:
                fitness = calc_fit(path, b)
                fitnesses.append(fitness)
                # find lowest/best fitness
                if fitness < bf:
                    bf = fitness
                    best_path = path
            
            # flatten list of lists for operations
            trial_fits.extend(fitnesses)
            # get number of fitness evaluations for while loop
            total_evals += len(ant_paths) # num ants
            
            if bf < overall_bf:
                overall_bf = bf
                overall_best_path = best_path

            # update pheromone paths and evaporate 
            g2 = update_pher(g1, ant_paths, b)
            g2 = evap_pher(g2, ant_paths, e)
            g1 = g2 

        all_trial_best_fits.append(overall_bf)
        # get average fitnesses
        trial_avg_fit = sum(trial_fits) / len(trial_fits)
        all_trial_avg_fits.append(trial_avg_fit)
        all_trials_fitness_data.append(trial_fits)

        print(f"Trial {trial + 1} Pheromones:")
        print(g2)

        # trial results
        if b == 10:
            print(f"\n Trial: {trial + 1} \n BPP1 \n Ants: {p} \n Evaporation rate: {e} \n Average fitness after {evals} evaluations: {trial_avg_fit} \n Best fitness after {evals} evaluations: {overall_bf} \n")
        else:
            print(f"\n Trial: {trial + 1} \n BPP2 \n Ants: {p} \n Evaporation rate: {e} \n Average fitness after {evals} evaluations: {trial_avg_fit} \n Best fitness after {evals} evaluations: {overall_bf} \n")

        # Update best bin distribution across all trials
        if overall_bf < best_fitness_across_trials:
            best_fitness_across_trials = overall_bf
            best_overall_path = overall_best_path

    # final esults after all trials
    overall_avg_fitness = sum(all_trial_avg_fits)/len(all_trial_avg_fits)
    print(f"\nOverall Average Fitness Across All Trials: {overall_avg_fitness}")
    print(f"Best Fitness Across All Trials: {best_fitness_across_trials}")

    # best path bin weight
    plot_bins(best_overall_path, b)
    
    return all_trial_best_fits

# testing

# g1 = make_graph(4, 3)
# print (g1)

# paths = gen_paths(5, g1)
# print(paths)

# g2 = update_pher(g1, paths, 3)
# print(g2)

# g3 = evap_pher(g2, paths, 0.5)
# print(g3)

#Main: 

#BPP1
# k = 500
# b = 10
# item weight = item index i

# p10e6 = run_trial(p=10, e=0.6, k=500, b=10, num_trials=5)
# p10e9 = run_trial(p=10, e=0.9, k=500, b=10, num_trials=5)
p100e6 = run_trial(p=100, e=0.6, k=500, b=10, num_trials=5)
# p100e9 = run_trial(p=100, e=0.9, k=500, b=10, num_trials=5)

# fitness graph of all parameter combinations
# plt.plot(p10e6, marker='o', linestyle='-', label='p = 10, e = 0.6', color='blue')
# plt.plot(p10e9, marker='o', linestyle='-.', label='p = 10, e = 0.9', color='orange')
# plt.plot(p100e6, marker='o', linestyle='--', label='p = 100, e = 0.6', color='green')
# plt.plot(p100e9, marker='o', linestyle=':', label='p = 100, e = 0.9', color='red')

# plt.xlabel('Trial')
# plt.ylabel('Best Fitness')
# plt.title('BPP1 Best Fitness Aross 5 Trials by Parameter Combination')
# plt.xticks(range(5), [f'{i + 1}' for i in range(5)])  # Show only 5 trials
# plt.grid()
# plt.legend()
# plt.show()

#BPP2
# k = 500
# b = 50
# item weight = i^2/2

# run_trial(p=10, e=0.6, k=500, b=50, num_trials=5)
# run_trial(p=10, e=0.9, k=500, b=50, num_trials=5)
run_trial(p=100, e=0.6, k=500, b=50, num_trials=5)
# run_trial(p=100, e=0.9, k=500, b=50, num_trials=5)