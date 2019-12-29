import numpy as np
from Tree import Tree, TreeMixture
from em_algorithm import EM_Algorithm
from matplotlib import pyplot as plt

seed_val = 12124

def do_experiment(num_clusters, num_nodes):
    tm = TreeMixture(num_clusters, num_nodes)
    tm.simulate_pi(seed_val)
    tm.simulate_trees(seed_val)
    tm.sample_mixtures(2000, seed_val=seed_val)

    samples = tm.samples

    em = EM_Algorithm(samples, num_clusters, seed_val=seed_val)
    em.initialize(1, 2)
    loglikelihoods, topology_list, theta_list = em.optimize(max_num_iter=100)

    real_likelyhood = tm.likelihood_dataset(samples)
    learned_likelyhood = em.tree_mixture.likelihood_dataset(samples)

    print("Real likelyhood: "+str(real_likelyhood)+", learned "+str(learned_likelyhood))

    plt.title('Actual likelihood: '+str(real_likelyhood)+', inferred likelyhood: '+str(learned_likelyhood))
    plt.plot(range(len(loglikelihoods)), loglikelihoods, label=str(num_clusters)+' clusters, '+str(num_nodes)+' nodes')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    clusters = [2]
    nodes = 10

    for num_clusters in clusters:
        do_experiment(num_clusters, nodes)
