""" This file is created as a template for question 2.5 in DD2434 - Assignment 2.

    Please keep the fixed parameters in the function templates as is (in 2_5.py file).
    However if you need, you can add parameters as default parameters.
    i.e.
    Function template: def em_algorithm(seed_val, samples, k, num_iter=10):
    You can change it to: def em_algorithm(seed_val, samples, k, num_iter=10, new_param_1=[], new_param_2=123):

    You can write helper functions however you want.

    You do not have to implement the code for finding a maximum spanning tree from scratch. We provided two different
    implementations of Kruskal's algorithm and modified them to return maximum spanning trees as well as the minimum
    spanning trees. However, it will be beneficial for you to try and implement it. You can also use another
    implementation of maximum spanning tree algorithm, just do not forget to reference the source (both in your code
    and in your report)!

    We also provided an example regarding the Robinson-Foulds metric (see Phylogeny.py).

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py file),
    and modify them as needed. In addition to the sample files given to you, it is very important for you to test your
    algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Note that the sample files are tab delimited with binary values (0 or 1) in it.
    Each row corresponds to a different sample, ranging from 0, ..., N-1
    and each column corresponds to a vertex from 0, ..., V-1 where vertex 0 is the root.
    Example file format (with 5 samples and 4 nodes):
    1   0   1   0
    1   0   1   0
    1   0   0   0
    0   0   1   1
    0   0   1   1

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    After all, we will test your code with commands like this one:
    %run 2_5.py "data/example_tree_mixture.pkl_samples.txt" "data/example_result" 3 --seed_val 123
    where
    "data/example_tree_mixture.pkl_samples.txt" is the filename of the samples
    "data/example_result" is the base filename of results (i.e data/example_result_em_loglikelihood.npy)
    3 is the number of clusters for EM algorithm
    --seed_val is the seed value for your code, for reproducibility.

    For this assignment, we gave you three different trees
    (q_2_5_tm_10node_20sample_4clusters, q_2_5_tm_10node_50sample_4clusters, q_2_5_tm_20node_20sample_4clusters).
    As the names indicate, the mixtures have 4 clusters with varying number of nodes and samples.
    We want you to run your EM algorithm and compare the real and inferred results in terms of Robinson-Foulds metric
    and the likelihoods.
    """
import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import sys
import dendropy
from Tree import Tree, TreeMixture, Node
from Kruskal_v2 import maximum_spanning_tree
from tqdm import tqdm
from em_algorithm import EM_Algorithm
from Phylogeny import tree_to_newick_rec


def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)

def em_algorithm(seed_val, samples, num_clusters, max_num_iter=100):
    em = EM_Algorithm(samples, num_clusters, seed_val=seed_val)
    em.initialize(100, 5) # Sieving
    loglikelihood, topology_array, theta_array = em.optimize()
    return loglikelihood, topology_array, theta_array
    

def main():
    # Code to process command line arguments
    parser = argparse.ArgumentParser(description='EM algorithm for likelihood of a tree GM.')
    parser.add_argument('sample_filename', type=str,
                        help='Specify the name of the sample file (i.e data/example_samples.txt)')
    parser.add_argument('output_filename', type=str,
                        help='Specify the name of the output file (i.e data/example_results.txt)')
    parser.add_argument('num_clusters', type=int, help='Specify the number of clusters (i.e 3)')
    parser.add_argument('--seed_val', type=int, default=42, help='Specify the seed value for reproducibility (i.e 42)')
    parser.add_argument('--real_values_filename', type=str, default="",
                        help='Specify the name of the real values file (i.e data/example_tree_mixture.pkl)')
    # You can add more default parameters if you want.

    print("Hello World!")
    print("This file demonstrates the flow of function templates of question 2.5.")

    print("\n0. Load the parameters from command line.\n")

    args = parser.parse_args()
    print("\tArguments are: ", args)

    print("\n1. Load samples from txt file.\n")

    samples = np.loadtxt(args.sample_filename, delimiter="\t", dtype=np.int32)
    num_samples, num_nodes = samples.shape
    print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")

    loglikelihood, topology_array, theta_array = em_algorithm(args.seed_val,
                                                              samples,
                                                              num_clusters=args.num_clusters)

    print("\n3. Save, print and plot the results.\n")

    save_results(loglikelihood, topology_array, theta_array, args.output_filename)

    for i in range(args.num_clusters):
        print("\n\tCluster: ", i)
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    print("\n4. Retrieve real results and compare.\n")
    if args.real_values_filename != "":
        print("\tComparing the results with real values...")
        actual_tm = TreeMixture(args.num_clusters, num_nodes)
        actual_tm.load_mixture(args.real_values_filename)

        inferred_tm = TreeMixture(args.num_clusters, num_nodes)
        inferred_tm.load_mixture(args.output_filename)

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        diff = compute_tree_mix_diff(actual_tm, inferred_tm)
        print("Total Robinson-Foulds distance: "+str(diff))

        print("\t4.2. Make the likelihood comparison.\n")
        actual_lik = actual_tm.likelihood_dataset(samples)
        inferred_lik = inferred_tm.likelihood_dataset(samples)
        print("Log-Likelihood of actual tree: "+str(actual_lik)+", inferred tree: "+str(inferred_lik))

def compute_tree_mix_diff(actual_tm, inferred_tm):
    actual_newicks = [tree_to_newick_rec(tree.root) for tree in actual_tm.clusters]
    inferred_newicks = [tree_to_newick_rec(tree.root) for tree in inferred_tm.clusters]
    
    tns = dendropy.TaxonNamespace()
    actual_trees = [dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=tnx)
           for newick in actual_newicks]
    inferred_trees = [dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=tnx)
           for newick in inferred_newicks]
    
    assigned = [False for tree in actual_trees]
    
    # We assign each inferred tree to the actual tree with lowest symmetric distance and
    # compute the sum of the assigned distances
    tot_diff = 0
    for inf_tree in inferred_trees:
        best_diff = float('inf')
        best_diff_idx = -1
        for i, act_tree in enumerate(actual_trees):
            if assigned[act_tree]:
                continue
            cur_diff = dendropy.calculate.treecompare.symmetric_difference(act_tree, inf_tree)
            if cur_diff < best_diff:
                best_diff = cur_diff
                best_diff_idx = i
        tot_diff += best_diff
        assigned[best_diff_idx] = True
    return tot_diff

if __name__ == "__main__":
    main()
