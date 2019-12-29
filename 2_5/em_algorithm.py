import numpy as np
import itertools as it
import sys
from Kruskal_v2 import maximum_spanning_tree
from Tree import Node, Tree, TreeMixture
from tqdm import tqdm
import copy

class EM_Algorithm:

    def __init__(self, samples, num_clusters, seed_val=None):
        self.samples = samples
        self.num_samples = samples.shape[0]
        self.num_nodes = samples.shape[1]
        self.num_clusters = num_clusters
        self.tree_mixture = None
        self.seed_val = seed_val
        if seed_val is not None:
            np.random.seed(seed_val)

    def optimize(self, max_num_iter=100):
        """Executes EM em_algorithm
        
        Parameters:
            max_num_iter -- Maximum number of iterations to perform
        
        Returns:
            loglikelyhoods -- Log likelihood of data for each iteration
            topology_list  -- List of tree topologies. Numpy array (num_clusters, num_nodes)
            theta_list     -- List of tree CDPs. Numpy array (num_clusters, num_nodes, 2)
        """
        if self.tree_mixture is None:
            raise Exception('EM has not been initialized. Call initialize() first')
        tm, loglikelihoods = self.__iter_optimize(max_num_iter)
        self.tree_mixture = tm
        topology_list = []
        theta_list = []
        for tree in tm.clusters:
            topology_list.append(tree.get_topology_array())
            theta_list.append(tree.get_theta_array())

        loglikelihoods = np.array(loglikelihoods)
        topology_list = np.array(topology_list)
        theta_list = np.array(theta_list)
        return loglikelihoods, topology_list, theta_list



    def __iter_optimize(self, max_num_iter, tree_mix=None, display_progress=True):
        """Performs the given number of iterations of the EM algorithm
        
        Parameters:
            max_num_iter -- Maximum number of iterations (can terminate before if converged)
            tree_mix     -- Initial TreeMixture. If None, self.tree_mixture is used

        Returns:
            TreeMixture optimized by EM algorithm
            List of likelihoods
        """
        print("Performing EM")
        if tree_mix is None:
            tree_mix = self.tree_mixture

        log_likelihoods = []
        for iter_ in tqdm(range(max_num_iter), disable=(not display_progress)):
            resp = self.responsibilities(tree_mix)
            new_pi = np.sum(resp, axis=0) / self.num_samples
            
            new_trees = []
            for k in range(len(tree_mix.clusters)):
                graph = self.weighted_graph(k, resp)
                max_span_tree = maximum_spanning_tree(graph)
                tree = self.to_tree_obj(max_span_tree, k, resp)
                new_trees.append(tree)
            tree_mix.clusters = new_trees
            tree_mix.pi = new_pi
            likelihood = tree_mix.likelihood_dataset(self.samples)
            if iter_ > 0 and likelihood < log_likelihoods[-1]:
                print("Likelyhood is decreasing")
            log_likelihoods.append(likelihood)
        return tree_mix, log_likelihoods

    
    def responsibilities(self, tree_mixture):
        """Compute the responsibility matrix

        Parameters:
            tree_mixture -- TreeMixture for which the responsibilities must be computed

        Returns:
            Numpy array (num_samples, num_clusters) with responsibilities for each
            sample and cluster
        """
        resp = np.empty((self.samples.shape[0], tree_mixture.num_clusters))
        for n, sample in enumerate(self.samples):
            for k, tree in enumerate(tree_mixture.clusters):
                resp[n, k] = tree_mixture.prob_observation_given_tree(sample, k)
            resp[n] /= tree_mixture.prob_observation(sample)
        resp *= tree_mixture.pi
        resp += sys.float_info.min
        row_sums = np.sum(resp, axis=1)
        resp /= row_sums[:, np.newaxis]
        return resp

    def weighted_graph(self, k, resp):
        """Creates a fully connected graph where the weight between vertices
        s and t is the mutual information of the respective nodes in the samples,
        computed by assuming the k-th graphical model

        Parameters:
            resp    -- Responsibility matrix (num_samples, num_trees)
            k       -- Index of graphical model for which the graph has to be computed

        Returns:
            Fully connected graph described above as a dictionary
            { 'vertices' : list of indices of graphical model nodes
              'edges'    : list of tuples (v_i, v_j, weight)
            }
        """
        graph = {
            'vertices' : list(range(self.samples.shape[1])),
            'edges'    : [] 
        }
        edges = []
        for s in range(self.num_nodes):
            for t in range(s + 1, self.num_nodes):
                w = self.mutual_information(s, t, k, resp)
                edge = (s, t, w)
                edges.append(edge)
        graph['edges'] = edges
        return graph

    def mutual_information(self, s, t, k, resp):
        I = 0
        for a, b in it.product([0, 1], repeat=2):
            q_s_t_a_b = self.q(k, s, t, a, b, resp)
            if q_s_t_a_b == 0:
                continue
            q_s_a = self.q_marginal(k, s, a, resp)
            q_t_b = self.q_marginal(k, t, b, resp)
            denom = q_s_a * q_t_b
            if denom == 0: # Handle precision loss in product
                denom = sys.float_info.min
            I += q_s_t_a_b * np.log(q_s_t_a_b / denom)
        return I

    def q(self, k, s, t, a, b, resp):
        num = np.sum(np.where((self.samples[:, s] == a) * (self.samples[:, t] == b),
                               resp[:, k], 0))
        denom = np.sum(resp[:, k])
        return num/denom
    
    def q_marginal(self, k, s, a, resp):
        num = np.sum(np.where(self.samples[:, s] == a, resp[:, k], 0))
        denom = np.sum(resp[:, k])
        return num/denom

    def initialize(self, sieving_tries=100, sieving_train=10):
        """Initializes the TreeMixtures using sieving.

        Parameters:
            sieving_tries -- Number of random initializations
            sieving_train -- Number of iterations for each initialization
        """
        print("Initializing EM ...")
        best_tree_mix = None
        best_likelihood = -float('inf')
        for t in tqdm(range(sieving_tries)):
            tree_mix = TreeMixture(self.num_clusters, self.num_nodes)
            tree_mix.simulate_trees(self.seed_val)
            tree_mix.simulate_pi(self.seed_val)

            tm, likelihoods = self.__iter_optimize(sieving_train, tree_mix=tree_mix, display_progress=False)
            if likelihoods[-1] > best_likelihood:
                best_likelihood = likelihoods[-1]
                best_tree_mix = tm
        self.tree_mixture = best_tree_mix

    def to_tree_obj(self, spanning_tree, k, resp):
        """Computes a Tree object from the spanning tree. Categorical distributions are
        computed from the samples and responsibilities

        Parameters:
            spanning_tree -- List of tuples (v_i, v_j weight)
            k             -- Index of tree we're updating
            resp          -- Responsibility matrix (num_samples, num_trees)

        Returns:
            Tree object with first vertex in the spanning tree as root and remaining nodes 
            organised as a Tree, each node with its MLE q distribution
        """
        tree = Tree()
        tree.root = Node(0, [])
        tree.k = 2
        visit_list = [tree.root]
        remaining_nodes = spanning_tree

        while len(visit_list) != 0:# and len(remaining_nodes):
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

        # Now we built a tree. We traverse it again to set the correct node names and
        # categorical distributions
        visit_list = [tree.root]
        cont = 0
        while len(visit_list) != 0:
            cur_node = visit_list[0]
            # TODO: Why does it work if I dont't rename the nodes?
            #cur_node.name = cont 
            cat = []
            if cur_node.ancestor == None:
                cat = [self.q_marginal(k, cur_node.name, 0, resp),
                       self.q_marginal(k, cur_node.name, 1, resp)]
            else:
                parent_idx = int(cur_node.ancestor.name)
                child_idx = int(cur_node.name)
                for p in [0, 1]: # For possible values of parent
                    q_given_parent = [0, 0]
                    for c in [0, 1]: # For possible values of child
                        q_given_parent[c] = self.q(k, parent_idx, child_idx, p, c, resp)
                    q_given_parent /= self.q_marginal(k, parent_idx, p, resp)
                    cat.append(q_given_parent)
            cont += 1
            cur_node.cat = cat
            visit_list = visit_list[1:] + cur_node.descendants
        return tree
