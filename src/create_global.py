#! /usr/bin/env python3

import os

import networkx as nx
import pandas as pd

from util import timing


@timing
def create_global_graph(dataset_name="enron", output_folder="output/"):
    nodes_csv_path = os.path.join(
        output_folder, dataset_name, f"{dataset_name}_nodes.csv.gz"
    )
    edges_csv_path = os.path.join(
        output_folder, dataset_name, f"{dataset_name}_edges.csv.gz"
    )
    output_graph_path = os.path.join(
        output_folder, dataset_name, f"{dataset_name}_global_graph.gml.gz"
    )
    if os.path.exists(output_graph_path):
        print(f"{output_graph_path} exists")
        return None
    G = nx.Graph()

    nodes_df = pd.read_csv(nodes_csv_path)
    edges_df = pd.read_csv(edges_csv_path)

    for index, row in nodes_df.iterrows():
        node_id = str(row["node"])
        G.add_node(node_id)
        if "type" in row and pd.notna(row["type"]):
            G.nodes[node_id]["type"] = str(row["type"])
        if "vector" in row and pd.notna(row["vector"]):
            G.nodes[node_id]["vector"] = str(eval(row["vector"]))
        if "avgSim_u" in row and pd.notna(row["avgSim_u"]):
            G.nodes[node_id]["avgSim_u"] = str(row["avgSim_u"])
        if "user_neighbors" in row and pd.notna(row["user_neighbors"]):
            G.nodes[node_id]["user_neighbors"] = str(eval(row["user_neighbors"]))
        if "text_nodes" in row and pd.notna(row["text_nodes"]):
            G.nodes[node_id]["text_nodes"] = str(eval(row["text_nodes"]))

    nodes_to_remove = [node for node in G.nodes if G.nodes[node].get("type") != "user"]
    G.remove_nodes_from(nodes_to_remove)

    for _index, row in edges_df.iterrows():
        u, v = str(row["source"]), str(row["target"])
        if u in G.nodes and v in G.nodes:
            G.add_edge(u, v)

    print(f"total number of nodes: {G.number_of_nodes()}")

    nx.write_gml(G, output_graph_path)
    print(f"Updated graph saved to {output_graph_path}")
    return G
