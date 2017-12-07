"""
Cluster.py

Dependencies:
==============
    Collect.py

Since huge data is involved in collect.py and runs for approx 15 hrs, and if the programmer does not wish to re-run the
collect.py, then this Cluster program will take the files that was run as a part of my initial test of collecting data.
As a part of Collect.py testing, I have collected 5 screen_ids, 250 followers of them and 300 followers of each 250
followers and ran for almost 15 hrs.

If the programmer running this file wishes to run collect.py first, then collect program will create files accordingly
for this program to pick up.

Note:
    No action required from the programmer. This program will take the files according to the programmer and process.

This file will cluster the screen_id and its friends.
"""

import pickle
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os.path


def read_data():
    """
    Read the pickle file collected from collect.py file.
    If
        collect.py is run, then it would have created the below files.
                    followers_followers_dict, ids_bk.pkl  - With 5 screen_ids, 250 followers and 300 followers_followers.
    else:
        If programmer is not running collect.py file, then this program would take the exisiting file tested as a part
        of my testing of collecting data.
                    followers_followers_dict, ids_bk.pkl  - With 5 screen_ids, 250 followers and 300 followers_followers.

    Params:
            followers_followers_dict....Name of the pickle file (Followers of Followers) to read.
            ids....Name of the pickle file (IDs file) to read.
    Returns:
            A list of strings, one per movie_name, in the order they are listed in the file.
    """

    if os.path.isfile('followers_followers_dict.pkl') and os.path.isfile('ids.pkl'):       # Created by the programmer

        pkl_file = open('followers_followers_dict.pkl', 'rb')
        followers_followers_dict = pickle.load(pkl_file)
        pkl_file.close()

        pkl_file = open('ids.pkl', 'rb')
        ids = pickle.load(pkl_file)
        pkl_file.close()

    else:

        pkl_file = open('followers_followers_dict_input.pkl', 'rb')           # Created by me as part of collect testing
        followers_followers_dict = pickle.load(pkl_file)
        pkl_file.close()

        pkl_file = open('ids_input.pkl', 'rb')
        ids = pickle.load(pkl_file)
        pkl_file.close()

    return followers_followers_dict, ids


def detect_friends(ids, followers_followers_dict):
    """
    Detect friends - People who follow both ways to create an undirected edge between the nodes in a graph
    Params:
        ids ................ ID's of the screen_names
        followers_followers_dict ............ Followers of followers of screen ID's.
            Note: Person A follows person B, this method will detect if Person B also follows person A. This way an
                undirected edge will be created between A and B for clustering.
    Returns:
        Friends_dict - A dict from the screen_id to its friends (Followers both ways)
    """

    friends_dict = defaultdict(list)
    search_ids = set(ids).union(set(followers_followers_dict.keys()))

    for key, values in followers_followers_dict.items():
        for value in list(values):
            if value in search_ids:
                friends_dict[value].append(key)
    return friends_dict


def create_graph(friends_dict):
    """
    Create a networkx undirected Graph, adding each screen_id and its friend as a node.

    Each candidate in the Graph will be represented by their screen_name, while each friend will be represented
    by their user id.

    Args:
        friends_dict........The dict mapping each friend to the screen_id who follow them and who is followed by them.
    Returns:
        A networkx Graph
    """

    graph = nx.Graph(friends_dict)

    for key, values in friends_dict.items():
        for value in values:
            if value in graph:
                graph.add_edge(key, value)
            else:
                graph.add_node(value)
                graph.add_edge(key, value)

    return graph


def draw_network(graph, filename):
    """
    Draw the network to a file.
    """

    position = nx.spring_layout(graph)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    nx.draw_networkx(graph, pos=position, with_labels=False, arrows=False, node_size=100, alpha=0.5, width=0.5,
                     node_color='b')
    plt.savefig(filename)


def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.
    """

    degreeList = [node for node, degree in dict(nx.degree(graph)).items() if degree >= min_degree]
    subgraph = graph.subgraph(degreeList)
    return subgraph


def community_detection(graph, length):
    """
    Use your approximate_betweenness algorithm implementation to partition a graph.

    That is, compute the approximate betweenness of all the edges in the graph and keep removing them until
    multiple components are created.

    Feature edge_betweenness_centrality in networkx is used to detect the communities.

    Returns:
        clusters - List of communities detected.
    """

    H = graph.copy()
    count = 0

    def find_best_edge(H0):
        result = nx.edge_betweenness_centrality(H0)
        return sorted(result.items(), key=lambda x: x[1], reverse=True)[0][0]

    clusters = [c for c in nx.connected_component_subgraphs(H)]

    while len(clusters) < length:
        edge_to_remove = find_best_edge(H)
        H.remove_edge(*edge_to_remove)
        clusters = [c for c in nx.connected_component_subgraphs(H)]
        count += 1

    return clusters


def get_clusters_info(clusters):
    """
    Get clusters information as below.

    Returns:
    num_of_communities ........... Prints the number of communities formed from the clusters info.
    Avg Users per communities .......... Compute the average number of users per community.

    """

    num_of_communities = len(clusters)
    n = 0
    for x in range(len(clusters)):
        n += len(clusters[x])
    avg_users_per_community = n / len(clusters)

    return num_of_communities, avg_users_per_community


def output_file(graph, subgraph, ids, clusters, num_of_communities, avg_users_per_community):
    """
    Params:
        graph, ids, clusters, num_of_communities, avg_users_per_community
    Returns:
        Nothing. Writes the output to Users.txt file.
    """

    num_of_users = graph.number_of_nodes() - len(ids)

    f = open('cluster_results.txt', 'w', encoding='utf-8')
    f.write("******************************************************************************************************")
    f.write("\n                    NETWORK INFORMATION                                                            \n")
    f.write("******************************************************************************************************")
    f.write("\n\nNumber of Users collected :%d" % num_of_users)
    f.write("\nThe Original Graph has %d nodes and %d edges " % (graph.order(), graph.number_of_edges()))
    f.write('\n\nThe Subgraph (MinFreq >= 3) has %d nodes and %d edges ' % (subgraph.order(), subgraph.number_of_edges()))

    f.write("\n\nThe number of communities discovered  : %d \n" % num_of_communities)
    f.write("Average number of users per community : %d\n" % avg_users_per_community)

    f.write("\n******************************************************************************************************")
    f.write("\n                    CLUSTERS INFORMATION                                                            \n")
    f.write("*******************************************************************************************************")

    f.write("\n\nNumber of clusters \t\t Corresponding Number of Nodes/Followers")
    f.write("\n****************** \t\t  ***********************************\n")
    for i in range(0, len(clusters)):
        f.write("%d \t\t\t\t\t\t\t %d \n" % (i, len(clusters[i])))
    f.write("\n")
    f.close()


def main():
    """
    Main Method
    """
    print("\nCollecting required pkl files for processing ..............")

    followers_followers_dict, ids = read_data()
    print("\nDetecting friends - People who follow a person and vice versa -  "
          " These are determined by getting the followers of followers of the screen_id...  \n")

    friends_dict = detect_friends(ids, followers_followers_dict)

    graph = create_graph(friends_dict)
    draw_network(graph, 'network_original.png')
    print("Original Graph is plotted and the corresponding network is drawn to ----> network_original.png file.....\n")

    subgraph = get_subgraph(graph, 3)
    draw_network(subgraph, 'network_pruned.png')
    print("Pruned SubGraph (minfreq >= 3) - after removing the outliers are plotted and the corresponding network "
          "is drawn to  ----------> network_pruned.png file....\n")

    print("Detecting communities (using pruned graph) in progress................\n")
    clusters = community_detection(subgraph, 4)
    print("Communities are detected via Edge-betweeness centrality & results are captured in ----> "
          "cluster_results.txt file")
    num_of_communities, avg_users_per_community = get_clusters_info(clusters)
    output_file(graph, subgraph, ids, clusters, num_of_communities, avg_users_per_community)
    print("\n************************************** END OF PROCESSING*********************************************")


if __name__ == '__main__':
    main()
