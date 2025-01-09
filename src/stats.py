import os
import networkx as nx
import json
import numpy as np
import logging
import community as community_louvain
import random
from sklearn.metrics.pairwise import rbf_kernel

from util import timing

logging.basicConfig(level=logging.INFO)


def compute_mmd(X, Y, kernel="rbf", gamma=1.0):
    if kernel == "rbf":
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)
    else:
        raise ValueError("Only 'rbf' is supported.")

    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd


def calculate_mmd_between_graphs(G1, G2, num_nodes, gamma=1.0):
    try:
        sampled_nodes = random.sample(list(G1.nodes()), num_nodes)
        G1_sampled = G1.subgraph(sampled_nodes)
        G2_sampled = G2.subgraph(sampled_nodes)

        X = nx.to_numpy_array(G1_sampled)
        Y = nx.to_numpy_array(G2_sampled)

        mmd = compute_mmd(X, Y, kernel="rbf", gamma=gamma)
        logging.info(f"MMD calculated: {mmd}")
        return mmd
    except Exception as e:
        logging.error(f"Error: {e}")
        return None


def calculate_avgSim_assortativity(graph):
    metrics = {}
    for node in graph.nodes():
        if "avgSim_u" not in graph.nodes[node] or graph.nodes[node]["avgSim_u"] is None:
            graph.nodes[node]["avgSim_u"] = 0.0

    try:
        assortativity = nx.attribute_assortativity_coefficient(graph, "avgSim_u")
        if np.isnan(assortativity):
            logging.warning("avgSim assortativity resulted in NaN, setting to 0.0")
            assortativity = 0.0
    except Exception as e:
        logging.error(f"Error: {e}")
        assortativity = None

    metrics["avgSim_assortativity"] = assortativity

    return metrics


def calculate_degree_and_clustering(graph):
    metrics = {}
    metrics["degree_assortativity"] = nx.degree_assortativity_coefficient(graph)
    metrics["clustering_coefficient"] = nx.average_clustering(graph)
    metrics["density"] = nx.density(graph)
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    metrics["degree_distribution"] = degree_sequence[:10]
    metrics["degree_mean"] = np.mean(degree_sequence)
    metrics["degree_std"] = np.std(degree_sequence)
    metrics["number_of_user_nodes"] = sum(
        1 for _, attr in graph.nodes(data=True) if attr.get("type") == "user"
    )
    metrics["number_of_text_nodes"] = sum(
        1 for _, attr in graph.nodes(data=True) if attr.get("type") == "text"
    )

    return metrics


def calculate_modularity(graph):
    metrics = {}
    if isinstance(graph, nx.DiGraph):
        graph = graph.to_undirected()

    partition = community_louvain.best_partition(graph)
    metrics["modularity"] = community_louvain.modularity(partition, graph)

    return metrics


@timing
def compute_metrics(G1, G2, epsilon, dataset_name, output_folder="output/"):

    original_graph_path = os.path.join(
        output_folder, dataset_name, f"{dataset_name}_global_graph.gml.gz"
    )
    noisy_graph_path = os.path.join(
        output_folder,
        dataset_name,
        f"combined_synthesized_global_graph_{epsilon}.gml.gz",
    )
    output_path = os.path.join(
        output_folder, dataset_name, f"combined_metrics_{epsilon}.json"
    )

    if os.path.exists(output_path):
        print(f"{output_path} exists")
        return

    # Graphs
    if G1 is None:
        G1 = nx.read_gml(original_graph_path)
    if G2 is None:
        G2 = nx.read_gml(noisy_graph_path)
    assert G1.number_of_nodes() == G2.number_of_nodes(), (
        G1.number_of_nodes(),
        G2.number_of_nodes(),
    )

    # Metrics
    metrics = {}
    # TOCHECK: if we only release nodes and edges, how is this possible?
    # metrics.update(calculate_avgSim_assortativity(G2))
    metrics.update(calculate_degree_and_clustering(G2))
    metrics.update(calculate_modularity(G2))

    # MMD
    # TOCHECK: this needs to be improved, now it is just an approximation
    num_sample_nodes = min(1000, G1.number_of_nodes())
    metrics["mmd"] = calculate_mmd_between_graphs(G1, G2, num_sample_nodes)

    # JSON file
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"metrics saved to {output_path}")
