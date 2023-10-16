# General importations.
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import time
import itertools
import warnings
from scipy import stats

# Baselines.
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import chisq
from IPython.display import Image, display
import networkx as nx
import pydot


class LDPUtils():
    
    
    def get_ci(self,
               results: list,
               z: float = 1.96,
               return_mean: bool = True) -> tuple:
    
        '''
        Default is 95% confidence interval.
        '''

        mean = np.mean(results)
        se = stats.sem(results)
        ci = [mean - (z * se), mean + (z * se)]
        if return_mean:
            return mean, ci
        return ci
    

    def plot_nx(self,
                adjacency_matrix,
                labels,
                figsize = (10,10),
                dpi = 200,
                node_size = 800,
                arrow_size = 10):
        
        g = nx.from_numpy_array(adjacency_matrix, create_using = nx.DiGraph)
        plt.figure(figsize = figsize, dpi = dpi)  
        nx.draw_circular(g, 
                         node_size = node_size, 
                         labels = dict(zip(list(range(len(labels))), labels)), 
                         arrowsize = arrow_size,
                         with_labels = True)
        plt.show()
        plt.close()

        
    def view_pydot(self,
                   pdot, 
                   figsize = 500):
        
        img = Image(pdot.create_png(), width = 500)
        display(img)

        
    def plot_pydot_from_adjacency_matrix(self,
                                         adjacency_matrix, 
                                         labels, 
                                         undirected_edges: list = [("Z7", "X")],
                                         uncertain_edges: list = [("X", "Z2"), ("X", "Z6")]):

        # Convert adjacency matrix to networkx graph, then convert to pydot.
        g = nx.from_numpy_matrix(adjacency_matrix, create_using = nx.DiGraph)
        node_label_map = dict(zip(list(range(len(labels))), labels))
        g = nx.relabel_nodes(g, node_label_map, copy = True)
        p = nx.drawing.nx_pydot.to_pydot(g)

        # Replace X -> Y with dashed arrow.
        string = p.to_string()
        string = string.replace('X -> Y  [weight="1.0"]', 'X -> Y  [weight="1.0", style="dashed"]')
        p = pydot.graph_from_dot_data(string)[0]
        p.del_node('"\\n"')

        # Remove arrow heads from undirected edges.
        if undirected_edges is not None:
            string = p.to_string()
            for edge in undirected_edges:
                replace_from = '{} -> {}  [weight="1.0"]'.format(edge[0], edge[1])
                replace_to = '{} -> {}  [weight="1.0", dir=none]'.format(edge[0], edge[1])
                string = string.replace(replace_from, replace_to)
            p = pydot.graph_from_dot_data(string)[0]
            p.del_node('"\\n"')

        # Make uncertain edges dotted.
        if uncertain_edges is not None:
            string = p.to_string()
            for edge in uncertain_edges:
                replace_from = '{} -> {}  [weight="1.0"]'.format(edge[0], edge[1])
                replace_to = '{} -> {}  [weight="1.0", style="dotted"]'.format(edge[0], edge[1])
                string = string.replace(replace_from, replace_to)
            p = pydot.graph_from_dot_data(string)[0]
            p.del_node('"\\n"')

        # View graph.
        self.view_pydot(p)
        plt.close()


    def is_valid_adjustment_set(self,
                                adj_matrix: np.ndarray, 
                                x: int, 
                                y: int,
                                S: list,
                                verbose: bool = False) -> bool:
        
        '''
        Checks if `S` is a valid adjustment wrt `X` and `Y`.

        Parameters:
        ------------
        adj_matrix: The adjacency matrix for the DAG.
        x: The index of the `X` node.
        y: The index of the `Y` node.
        S: The indices of the `S` nodes (list[int]).

        Return:
        ---------
        Bool: Whether S is a valid adjustment wrt X and Y.
        '''

        S = set(S)
        dag = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

        # Set of nodes (including Y) on causal paths from X to Y.
        causal_nodes = set(p for path in nx.all_simple_paths(dag, source=x, target=y) for p in path if p != x)

        # The forbidden set contains nodes on causal paths from X to Y and their descendants.
        forbidden = set(causal_nodes)
        for n in causal_nodes:
            forbidden |= nx.descendants(dag, n)

        # A valid adjustment set cannot contain forbidden nodes.
        if len(forbidden & S) > 0:
            if verbose:
                print("Forbidden nodes in adjustment set:", S.intersection(forbidden))
            return False

        # Remove outgoing edges from X on causal paths to Y.
        edges_to_remove = [p[0] for p in nx.all_simple_edge_paths(dag, x, y)]
        dag.remove_edges_from(edges_to_remove)

        # check if S d-separates X and Y in the mutilated graph.
        return nx.d_separated(dag, {x}, {y}, S)