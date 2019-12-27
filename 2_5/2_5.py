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
from Tree import Tree, TreeMixture, Node
from Kruskal_v2 import maximum_spanning_tree
from tqdm import tqdm


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
    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)

    You can change the function signature and add new parameters. Add them as parameters with some default values.
    i.e.
    Function template: def em_algorithm(seed_val, samples, k, max_num_iter=10):
    You can change it to: def em_algorithm(seed_val, samples, k, max_num_iter=10, new_param_1=[], new_param_2=123):
    """

    # Set the seed
    np.random.seed(seed_val)

    # TODO: Implement EM algorithm here.

    # Initialize random trees and distribution
    tree_mix = TreeMixture(num_clusters, samples.shape[1])
    tree_mix.simulate_pi(seed_val)
    tree_mix.simulate_trees(seed_val)

    loglikelihoods = []
    for iter_ in tqdm(range(max_num_iter)):
        resp = responsibilities(samples, tree_mix.clusters, tree_mix.pi)
        new_pi = np.sum(resp, axis=0) / samples.shape[0]

        new_trees = []
        # Creating graphs
        for k in range(num_clusters):
            graph = weighted_graph(samples, resp, k)
            max_spanning_tree = maximum_spanning_tree(graph)
            tree = to_tree_obj(max_spanning_tree, k, samples, resp)
            new_trees.append(tree)
        tree_mix.pi = new_pi
        tree_mix.clusters = new_trees
        lik = loglikelihood(samples, tree_mix)
        loglikelihoods.append(lik)

    topology_list = []
    theta_list = []
    for tree in tree_mix.clusters:
        topology_list.append(tree.get_topology_array())
        theta_list.append(tree.get_theta_array())

    loglikelihoods = np.array(loglikelihoods)
    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)
    return loglikelihoods, topology_list, theta_list

def responsibilities(samples, trees, pi):
    """Compute the responsibilities for each sample and each tree

    Parameters:
        samples -- Numpy array (num_samples, num_nodes) with samples on the rows
        trees   -- List of Tree objects
        pi      -- List of probabilities. pi[i] = probability of tree i
    
    Returns:
        A Numpy array (num_samples, num_trees) in which the element at i,j is the
        responsibility of tree j for sample i
    """
    resp = np.empty((samples.shape[0], len(trees)))
    for sample_idx, x in enumerate(samples):
        for tree_idx, tree in enumerate(trees):
            resp[sample_idx, tree_idx] = pi[tree_idx] * p_x_given_tree(x, tree)
        resp[sample_idx, :] /= p_x_marginal(x, trees, pi)
    return resp

def p_x_given_tree(x, tree):
    """ Compute the probability of x given the tree
    Parameters:
        x    --list of values for each node of the tree in bfs order
        tree -- Tree object, represents the graphical model
    
    Returns:
        Probability of x given the tree
    """
    p = 1
    visit_list = [tree.root]

    while len(visit_list) != 0:
        cur_node = visit_list[0]
        cur_node_idx = int(cur_node.name)
        visit_list = visit_list[1:] + cur_node.descendants
        
        if cur_node == tree.root:
            p *= cur_node.cat[x[cur_node_idx]]
        else:
            par_node_idx = int(cur_node.ancestor.name)
            par_k = x[par_node_idx]
            p *= cur_node.cat[par_k][x[cur_node_idx]]
    if p == 0:
        p = sys.float_info.epsilon
    return p

def p_x_marginal(x, trees, pi):
    """ Compute the marginal probability of x

    Parameters:
        x     -- List of values for each node of the tree in bfs order
        trees -- List of Tree objects
        pi    -- List of probabilities. pi[i] = probability of tree i
    
    Returns:
        Probability of x marginalised over all the trees
    """
    p = 0
    for p_tree, tree in zip(pi, trees):
        p += p_x_given_tree(x, tree) * p_tree
    return p

def weighted_graph(samples, resp, k):
    """Creates a fully connected graph where the weight between vertices
    s and t is the mutual information of the respective nodes in the samples,
    computed by assuming the k-th graphical model

    Parameters:
        samples -- Numpy array (num_samples, num_nodes) with samples on the rows
        resp    -- Responsibility matrix (num_samples, num_trees)
        k       -- Index of graphical model for which the graph has to be computed

    Returns:
        Fully connected graph described above as a dictionary
        { 'vertices' : list of indices of graphical model nodes
          'edges'    : list of tuples (v_i, v_j, weight)
        }
    """
    graph = {
        'vertices' : list(range(samples.shape[1])),
        'edges'    : [] 
    }

    for s in range(samples.shape[1]):
        for t in range(s + 1, samples.shape[1]):
            # Adding edge s->t with weight I_q_k(X_s, X_t)
            w = mutual_information(s, t, k, samples, resp)
            edge = (s, t, w)
            graph['edges'].append(edge)
    return graph

def mutual_information(s, t, k, samples, resp):
    """Compute the mutual information between nodes s and t in the samples assuming
       the graphical model is k

    Parameters:
        s, t    -- Indices of the two nodes for which to compute the mut. information
        k       -- Index of the graphical model assumed
        samples -- Numpy array (num_samples, num_nodes) with samples on the rows
        resp    -- Responsibility matrix (num_samples, num_trees)
    
    Returns:
        Mutual information between node s and node t computed from the samples
    """
    # We need q_k(X_s = a, X_t = b) joint probability
    # marginals: q_k(X_s = a) and q_k(X_s = b)

    mut_inform = 0
    for a, b in it.product([0, 1], repeat=2): # All variables are binary
        joint = q(k, s, t, a, b, samples, resp)
        marginal_s = q_marginal(k, s, a, samples, resp)
        marginal_t = q_marginal(k, t, b, samples, resp) 
        num = joint
        denom = marginal_s * marginal_t 
        if num == 0 or denom == 0:
            continue # Avoid log(0)
        mut_inform += joint * np.log(joint / (marginal_s * marginal_t))
    return mut_inform

def q(k, s, t, a, b, samples, resp):
    """Compute q distribution for GM k, vertices s and t with values a, b

    Parameters:
        k       -- Index of graphical model to use in computation
        s, t    -- Indices of vertices
        a, b    -- a = value of vertice s, b = value of vertice t
        samples -- Numpy array (num_samples, num_nodes) with samples on the rows
        resp    -- Responsibility matrix (num_samples, num_trees)

    Returns:
        Ratio of vertex pairs s,t with values respectively a, b over all possible
        combinations ov values for a and b, weighted by the respective responsibilities
    """
    count = lambda i, j : sum(resp[n, k] for n, x in enumerate(samples)
                              if x[s] == i and x[t] == j)

    num = count(a, b)
    denom = sum(count(i, j) for i, j in it.product([0,1], repeat=2))
    return num/denom

def q_marginal(k, node, value, samples, resp):
    """Compute q distribution for vertice at index node with given value.
    """
    count = lambda i : sum(resp[n, k] for n, x in enumerate(samples) if x[node] == i)
    num = count(value)
    denom = sum(count(j) for j in [0, 1])
    return num/denom

def to_tree_obj(spanning_tree, k, samples, resp):
    """Computes a Tree object from the spanning tree. Categorical distributions are
    computed from the samples and responsibilities

    Parameters:
        spanning_tree -- List of tuples (v_i, v_j weight)
        k             -- Index of tree we're updating
        samples       -- Numpy array (num_samples, num_nodes) with samples on the rows
        resp          -- Responsibility matrix (num_samples, num_trees)

    Returns:
        Tree object with first vertex in the spanning tree as root and remaining nodes 
        organised as a Tree, each node with its MLE q distribution
    """
    tree = Tree()
    tree.root = Node(spanning_tree[0][0], [])
    tree.k = 2
    tree.newick = tree.get_tree_newick()
    visit_list = [tree.root]
    remaining_nodes = spanning_tree

    while len(visit_list) != 0:
        tree.num_nodes += 1
        tree.num_leaves += 1
        cur_node = visit_list[0]
        # We're going to separate the remaining nodes in those that are connected
        # to the current and those that aren't
        connected = [edge for edge in remaining_nodes
                     if edge[0] == int(cur_node.name) or edge[1] == int(cur_node.name)]
        not_connected = [edge for edge in remaining_nodes
                     if edge[0] != int(cur_node.name) and edge[1] != int(cur_node.name)]

        # We create a Node for the connected nodes and add them as children to current node
        for edge in connected:
            other = edge[0] if edge[0] != int(cur_node.name) else edge[1]
            child = Node(other, [])
            child.ancestor = cur_node
            cur_node.descendants.append(child)
        visit_list = visit_list[1:] + cur_node.descendants
        remaining_nodes = not_connected # Keep only nodes that aren't connected yet
    
    visit_list = [tree.root]
    cont = 0
    while len(visit_list) != 0:
        cur_node = visit_list[0]
        cur_node.name = cont 
        cat = []
        if cur_node.ancestor == None:
            cat = [q_marginal(k, cont, 0, samples, resp),
                   q_marginal(k, cont, 1, samples, resp)]
        else:
            parent_idx = int(cur_node.ancestor.name)
            child_idx = int(cur_node.name)
            for p in [0, 1]: # For possible values of parent
                q_given_parent = [0, 0]
                for c in [0, 1]: # For possible values of child
                    q_given_parent[c] = q(k, parent_idx, child_idx, p, c, samples, resp)
                    q_given_parent[c] /= q_marginal(k, parent_idx, p, samples, resp)
                cat.append(q_given_parent)
        cont += 1
        cur_node.cat = cat
        visit_list = visit_list[1:] + cur_node.descendants

    return tree

def loglikelihood(samples, tree_mixture):
    # TODO How to compute log with sum inside?
    """Compute P(samples)

    Parameters:
        samples      -- Numpy array (num_samples, num_nodes) with samples on the rows
        tree_mixture -- TreeMixture object representing the graphical model mixture

    Returns:
        Probability of the samples
    """
    
    p = 0
    for tree, p_tree in zip(tree_mixture.clusters, tree_mixture.pi):
        p_data = p_tree
        for x in samples:
            p_data *= p_x_given_tree(x, tree)
        p += p_data
    return np.log(p)


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

    loglikelihood, topology_array, theta_array = em_algorithm(args.seed_val, samples, num_clusters=args.num_clusters)

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

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        # TODO: Do RF Comparison

        print("\t4.2. Make the likelihood comparison.\n")
        # TODO: Do Likelihood Comparison


if __name__ == "__main__":
    main()
