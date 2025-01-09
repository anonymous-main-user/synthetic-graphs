import os
import networkx as nx
import numpy as np
from util import (
    parallel_for_balanced,
    sanitize_filename,
)

def laplace_noise(scale):
    return np.random.laplace(0, scale)

def signed_laplace_noise(value, scale):
    noise = np.random.laplace(0, scale)
    return (
        max(0, value + noise)
        if value > 0
        else min(0, value + noise) if value < 0 else noise
    )

def save_noised_ego_graphs(ego_graphs, base_path):
    for user, ego_data in ego_graphs.items():
        ego_graph, ego_vectors, user_neighbors, text_neighbors = ego_data

        sanitized_user = sanitize_filename(user)
        gml_path = os.path.join(base_path, f"ego_graph_{sanitized_user}.gml.gz")
        assert not os.path.exists(gml_path)
        nx.write_gml(ego_graph, gml_path)

def apply_exponential_mechanism(
    ego_graph,
    full_graph,
    user,
    avgSim_u,
    epsilon,
    sensitivity,
    max_avgSim,
    compatibility_scores,
    total_compatibility_scores,
):
    if ego_graph.degree(user) == 0:
        return ego_graph

    d_u = full_graph.degree(user)
    edge_utilities = {}
    for node in full_graph.nodes:
        if node != user:
            h_ij = compatibility_scores.get(node, 0)
            edge_utilities[node] = (h_ij * d_u) / total_compatibility_scores

    utilities = {}

    for node in full_graph.nodes:
        if node != user:
            edge_in_ego = (user, node) in ego_graph.edges
            edge_utility = edge_utilities.get(node, 0) * (
                1 - 1 / len(ego_graph.edges)
                if edge_in_ego
                else 1 / len(ego_graph.edges)
            )

            avgSim_preservation_penalty = abs(
                avgSim_u
                - min(full_graph.nodes[node].get("noised_avgSim", 0.0), max_avgSim)
            )

            utilities[node] = edge_utility - avgSim_preservation_penalty

    exp_mech_probs = {
        node: np.exp((epsilon * utility) / (2 * sensitivity))
        for node, utility in utilities.items()
    }
    sum_exp_mech_probs = sum(exp_mech_probs.values())
    normalized_probs = {
        node: prob / sum_exp_mech_probs for node, prob in exp_mech_probs.items()
    }

    selected_nodes = np.random.choice(
        list(normalized_probs.keys()),
        p=list(normalized_probs.values()),
        size=len(ego_graph.nodes) - 1,
        replace=False,
    )

    new_ego_graph = nx.DiGraph()

    new_ego_graph.add_node(user)
    for node in selected_nodes:
        new_ego_graph.add_edge(user, node)

    return new_ego_graph

def compute_ego_graph_utility(ego_graph, new_ego_graph, user, avgSim_u, max_avgSim):

    original_edges = set(ego_graph.edges)
    new_edges = set(new_ego_graph.edges)
    all_edges = original_edges.union(new_edges)

    edge_weights = {
        edge: (
            (1 - 1 / len(original_edges))
            if edge in original_edges
            else (1 / len(original_edges))
        )
        for edge in all_edges
    }

    edge_utilities = sum(edge_weights[edge] for edge in all_edges)

    similarity_preservation = -sum(
        abs(
            avgSim_u
            - min(
                float(new_ego_graph.nodes[node].get("noised_avgSim", 0.0)), max_avgSim
            )
        )
        for node in new_ego_graph.nodes
    )

    return edge_utilities + similarity_preservation

def get_candidate_nodes(ego_graph, full_graph, user, candidate_limit=None):

    candidate_nodes = set(ego_graph.neighbors(user))
    for neighbor in candidate_nodes:
        candidate_nodes.update(full_graph.neighbors(neighbor))
    if candidate_limit:
        all_nodes = list(full_graph.nodes)
        candidate_nodes.update(
            np.random.choice(all_nodes, candidate_limit, replace=False)
        )
    candidate_nodes.discard(user)
    return list(candidate_nodes)

def apply_exponential_mechanism_with_w_s(
    ego_graph,
    full_graph,
    user,
    avgSim_u,
    epsilon,
    sensitivity,
    max_avgSim,
    candidate_limit=100,
):

    candidate_nodes = get_candidate_nodes(ego_graph, full_graph, user, candidate_limit)
    compatibility_scores = {
        node: np.exp(-((avgSim_u - full_graph.nodes[node]["noised_avgSim"]) ** 2))
        for node in candidate_nodes
    }
    d_u = full_graph.degree(user)
    edge_utilities = {
        node: (compatibility_scores[node] * d_u)
        / sum(compatibility_scores[k] * full_graph.degree(k) for k in candidate_nodes)
        for node in candidate_nodes
    }
    exp_mech_probs = {
        node: np.exp((epsilon * utility) / (2 * sensitivity))
        for node, utility in edge_utilities.items()
    }
    sum_exp_mech_probs = sum(exp_mech_probs.values())
    normalized_probs = {
        node: prob / sum_exp_mech_probs for node, prob in exp_mech_probs.items()
    }
    num_edges = len(ego_graph.nodes) - 1
    selected_nodes = np.random.choice(
        list(normalized_probs.keys()),
        p=list(normalized_probs.values()),
        size=num_edges,
        replace=False,
    )
    new_ego_graph = nx.DiGraph()
    new_ego_graph.add_nodes_from(ego_graph.nodes)
    for node in selected_nodes:
        new_ego_graph.add_edge(user, node)
    utility = compute_ego_graph_utility(ego_graph, new_ego_graph, avgSim_u, max_avgSim)
    return new_ego_graph, utility

def process_single_ego_graph(
    gml_file_path,
    full_graph,
    epsilon_laplace,
    epsilon_exponential,
    sensitivity,
    scale,
    max_avgSim,
    compatibility_scores,
    total_compatibility_scores,
):
    try:
        ego_graph = nx.read_gml(gml_file_path)
    except Exception as e:
        print(f"Error loading ego graph from {gml_file_path}: {e}")
        return None

    user_nodes = [n for n, d in ego_graph.nodes(data=True) if d.get("type") == "user"]
    if len(user_nodes) == 0:
        print(f"Empty ego graph detected: {gml_file_path}")
        return None

    user = ego_graph.graph["main_node"]

    try:
        avgSim_u = float(ego_graph.nodes[user].get("avgSim_u", 0.0))
        noised_avgSim_u = signed_laplace_noise(avgSim_u, scale)
        ego_graph.nodes[user]["noised_avgSim"] = float(noised_avgSim_u)

        new_ego_graph = apply_exponential_mechanism(
            ego_graph,
            full_graph,
            user,
            noised_avgSim_u,
            epsilon_exponential,
            sensitivity,
            max_avgSim,
            compatibility_scores,
            total_compatibility_scores,
        )

        if new_ego_graph is None or len(new_ego_graph.nodes) == 0:
            print(
                f"Failed to create noised ego graph for {user}. Return original graph."
            )
            return {user: (ego_graph, None, None, None)}

        ego_vectors = {
            n: ego_graph.nodes[str(n)].get("vector", []) if n in ego_graph.nodes else []
            for n in new_ego_graph.nodes
        }
        user_neighbors = [
            n
            for n in new_ego_graph.neighbors(user)
            if new_ego_graph.nodes[n].get("type") == "user"
        ]
        text_neighbors = [
            n
            for n in new_ego_graph.neighbors(user)
            if new_ego_graph.nodes[n].get("type") == "text"
        ]

        processed_ego_graphs = {
            user: (new_ego_graph, ego_vectors, user_neighbors, text_neighbors)
        }
        return processed_ego_graphs

    except Exception as e:
        print(f"error during processing for user {user} in {gml_file_path}: {e}")
        print("return original ego graph.")
        return {user: (ego_graph, None, None, None)}

def process_single_ego_graph_par_bal(
    queue,
    iss,
    gml_file_paths,
    full_graph,
    epsilon_laplace,
    epsilon_exponential,
    sensitivity,
    scale,
    noised_base_path,
    max_avgSim,
    compatibility_scores,
    total_compatibility_scores,
):
    for i in iss:
        result = process_single_ego_graph(
            gml_file_paths[i],
            full_graph,
            epsilon_laplace,
            epsilon_exponential,
            sensitivity,
            scale,
            max_avgSim,
            compatibility_scores,
            total_compatibility_scores,
        )
        if result:
            save_noised_ego_graphs(result, noised_base_path)
        queue.put((i, None))
    queue.put(None)

def process_ego_graphs(G, epsilon, dataset_name="enron", output_folder="output/"):
    input_base_path = os.path.join(output_folder, dataset_name, "ego_graphs")
    noised_base_path = os.path.join(
        output_folder, dataset_name, f"noisy_ego_graphs_{epsilon}"
    )
    if os.path.exists(noised_base_path):
        print("process_ego_graphs: noised_base_path exists")
        return

    sensitivity = 2.0
    epsilon_laplace = epsilon / 2
    epsilon_exponential = epsilon / 2
    scale = max(0.01, sensitivity / epsilon_laplace)

    gml_files = sorted(
        [f for f in os.listdir(input_base_path) if f.endswith(".gml.gz")]
    )
    gml_file_paths = [os.path.join(input_base_path, f) for f in gml_files]

    if G is None:
        global_graph_path = os.path.join(
            output_folder, dataset_name, f"{dataset_name}_global_graph.gml.gz"
        )
        G = nx.read_gml(global_graph_path)

    if not os.path.exists(noised_base_path):
        os.makedirs(noised_base_path)

    print("Computing noised_ego_graphs with parallel_for_balanced")

    for node in G.nodes:
        avgSim_v = float(G.nodes[node].get("avgSim_u", 0.0))
        noised_avgSim_v = signed_laplace_noise(avgSim_v, scale)
        G.nodes[node]["noised_avgSim"] = float(noised_avgSim_v)

    max_avgSim = max(float(G.nodes[node].get("noised_avgSim", 0.0)) for node in G.nodes)

    compatibility_scores = {
        node: np.exp(
            -(
                (
                    float(G.nodes[node].get("avgSim_u", 0.0))
                    - G.nodes[node]["noised_avgSim"]
                )
                ** 2
            )
        )
        for node in G.nodes
    }

    total_compatibility_scores = sum(
        h_ik * G.degree(k) for k, h_ik in compatibility_scores.items()
    )

    parallel_for_balanced(
        process_single_ego_graph_par_bal,
        (
            gml_file_paths,
            G,
            epsilon_laplace,
            epsilon_exponential,
            sensitivity,
            scale,
            noised_base_path,
            max_avgSim,
            compatibility_scores,
            total_compatibility_scores,
        ),
        range(len(gml_file_paths)),
        DEBUG=True,
    )

    print(f"Noised ego graphs saved at {noised_base_path}")
