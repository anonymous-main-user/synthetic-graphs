#! /usr/bin/env python3

import os
import networkx as nx

from util import sanitize_filename, parallel_for_balanced


def save_ego_graph(G, node, output_folder):
    ego_graph = nx.ego_graph(G, node)
    sanitized_node = sanitize_filename(node)
    ego_graph.graph["main_node"] = node
    ego_graph_path = os.path.join(output_folder, f"{sanitized_node}_ego_graph.gml.gz")
    assert not os.path.exists(ego_graph_path)
    nx.write_gml(ego_graph, ego_graph_path)

    # TOCHECK: this is not used anywhere

    # nodes_data = {n: G.nodes[n] for n in ego_graph.nodes}
    # attributes_path = os.path.join(output_folder, f"{sanitized_node}_attributes.txt")
    # with open(attributes_path, "w") as f:
    #     for n, attrs in nodes_data.items():
    #         f.write(f"Node: {n}\n")
    #         for attr, value in attrs.items():
    #             f.write(f"  {attr}: {value}\n")
    #         f.write("\n")


def save_ego_graph_par(queue, iis, user_nodes, G, output_folder_graphs):
    for i in iis:
        save_ego_graph(G, user_nodes[i], output_folder_graphs)
        queue.put((i, None))
    queue.put(None)


def create_ego_graphs(G, dataset_name="enron", output_folder="output/"):
    global_graph_path = os.path.join(
        output_folder, dataset_name, f"{dataset_name}_global_graph.gml.gz"
    )
    output_folder_graphs = os.path.join(output_folder, dataset_name, "ego_graphs")
    if os.path.exists(output_folder_graphs):
        print("create_ego_graphs: output_folder_graphs exists")
        return
    os.makedirs(output_folder_graphs, exist_ok=True)
    if G is None:
        G = nx.read_gml(global_graph_path)
    user_nodes = [node for node in G.nodes if G.nodes[node].get("type") == "user"]


    for n in sorted(user_nodes):
        sanitize_filename(n)

    print("Creating ego graphs")
    parallel_for_balanced(
        save_ego_graph_par,
        (user_nodes, G, output_folder_graphs),
        range(len(user_nodes)),
        DEBUG=True,
    )


    print("Ego graphs extraction completed.")
