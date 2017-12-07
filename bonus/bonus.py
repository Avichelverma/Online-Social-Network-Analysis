#import networkx as nx
import numpy as np


# #To test this program, a sample graph is created.
# def example_graph():
#     g = nx.Graph()
#     g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'H'), ('B', 'D'), ('C', 'G'), ('C', 'D'), ('D', 'E'), ('D', 'F')])
#     return g


def jaccard_index(degrees, A, B):

    # Find the intersection of A & B
    intersection = A.intersection(B)

    # Calculate the numerator list
    numerator = [1/degrees[n] for n in intersection]

    # Calculate the denominator coming from A
    deno_a = [degrees[n] for n in A]

    # Calculate the denominator coming from B
    deno_b = [degrees[n] for n in B]

    # Return the modified jaccard index as per the new formula
    return np.sum(numerator) / ((1/np.sum(deno_a)) + (1/np.sum(deno_b)))


def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined above.
    Args:
    graph....a networkx graph
    node.....a node to score potential new edges for.
    Returns:
    A list of ((node, ni), score) tuples, representing the
              score assigned to edge (node, ni)
              (note the edge order)
    """
    # Find all nodes that are not already neighbours of node
    candidate_nodes = set(graph.nodes()) - set(graph.neighbors(node)) - {node}

    # get the degree of each node in the graph
    degrees = graph.degree()

    # get the neighbours of node
    neigh_node = set(graph.neighbors(node))

    # calculate the score for every pair of node & each element of candidate_nodes
    # and create a list of tuples as required
    output = [((node, ni), jaccard_index(degrees, neigh_node, set(graph.neighbors(ni)))) for ni in candidate_nodes]

    # return the sorted tuples based on the higher score first
    return sorted(output, key=lambda x: (-x[1], x[0][1]))


# #Main function is written to verify if this program works fine with respect to the requirement.
# if __name__ == '__main__':
#     g = example_graph()
#     print(jaccard_wt(g, 'A'))
