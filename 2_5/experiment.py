import numpy as np
from Tree import Tree, TreeMixture
from em_algorithm import EM_Algorithm
from matplotlib import pyplot as plt

seed_val = 12124

if __name__ == "__main__":
    tm = TreeMixture(2, 5)
    tm.simulate_pi(seed_val)
    tm.simulate_trees(seed_val)
    tm.sample_mixtures(100, seed_val=seed_val)

    samples = tm.samples

    em = EM_Algorithm(samples, 2, seed_val=seed_val)
    em.initialize(40, 5)
    loglikelihoods, topology_list, theta_list = em.optimize(max_num_iter=300)

    real_likelyhood = tm.likelihood_dataset(samples)
    learned_likelyhood = em.tree_mixture.likelihood_dataset(samples)

    print("Real likelyhood: "+str(real_likelyhood)+", learned "+str(learned_likelyhood))

    plt.plot(range(len(loglikelihoods)), loglikelihoods)
    plt.show()
