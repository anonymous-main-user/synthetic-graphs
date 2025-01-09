import os
import networkx as nx
from tqdm import tqdm


def load_noised_ego_graphs(base_path):
    combined_graph = nx.DiGraph()
    gml_files = sorted(
        [
            os.path.join(base_path, f)
            for f in os.listdir(base_path)
            if f.endswith(".gml.gz")
        ]
    )
    for gml_file_path in tqdm(gml_files, desc="Loading noised ego graphs"):
        ego_graph = nx.read_gml(gml_file_path)
        combined_graph.add_nodes_from(ego_graph.nodes)
        for edge in ego_graph.edges:
            combined_graph.add_edge(edge[0], edge[1])
    return combined_graph


def compute_synthesized_global_graph(
    epsilon, dataset_name="enron", output_folder="output/"
):
    noised_base_path = os.path.join(
        output_folder, dataset_name, f"noisy_ego_graphs_{epsilon}"
    )
    combined_graph_path = os.path.join(
        output_folder,
        dataset_name,
        f"combined_synthesized_global_graph_{epsilon}.gml.gz",
    )
    if os.path.exists(combined_graph_path):
        print("compute_synthesized_global_graph: combined_graph_path exists")
        return
    combined_synthesized_graph = load_noised_ego_graphs(noised_base_path)
    nx.write_gml(combined_synthesized_graph, combined_graph_path)
    print(f"Combined synthesized global graph saved  {combined_graph_path}")
