#! /usr/bin/env python3

import json
import os
import gzip
import pandas as pd
import networkx as nx
from textblob import TextBlob
from textblob.sentiments import PatternAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import ijson
import community.community_louvain as community_louvain


from util import timing, calculate_avg_path_length, parallel_for_balanced


def analyser_email_details(message):
    headers = {}
    try:
        parts = message.split("\n\n", 1)
        body = parts[1] if len(parts) > 1 else ""
        for line in parts[0].split("\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                headers[key] = value.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None
    return headers.get("From"), headers.get("To", ""), body.strip()


def attribute_assortativity_coefficient(G, attribute):
    try:
        nodes_with_attr = [
            node
            for node in G.nodes
            if attribute in G.nodes[node] and G.nodes[node][attribute] is not None
        ]

        if len(nodes_with_attr) < 2:
            return None

        for node in nodes_with_attr:
            if isinstance(G.nodes[node][attribute], float):
                G.nodes[node][attribute] = int(round(G.nodes[node][attribute]))

        return nx.numeric_assortativity_coefficient(G, attribute)
    except Exception as e:
        print(f"Error avgSim_u: {e}")
        return None


def save_nodes_to_csv(G, file_path):
    nodes_data = []
    for node in G.nodes:

        if "Pref_u" in G.nodes[node]:
            pref_u_value = G.nodes[node]["Pref_u"]
            if isinstance(pref_u_value, np.ndarray):
                pref_u_value = pref_u_value.tolist()
        else:
            pref_u_value = None

        node_data = {
            "node": node,
            "type": G.nodes[node]["type"],
            "vector": json.dumps(pref_u_value) if pref_u_value is not None else None,
            "avgSim_u": G.nodes[node].get("avgSim_u", None),
        }
        nodes_data.append(node_data)

    nodes_df = pd.DataFrame(nodes_data)
    nodes_df.to_csv(file_path, index=False)


def save_edges_to_csv(G, file_path):
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({"source": u, "target": v})
    edges_df = pd.DataFrame(edges_data)
    edges_df.to_csv(file_path, index=False)


def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


def get_sentiment_par_bal(queue, iis, texts, P):
    for i in iis:
        ans = P.analyze(texts[i]).polarity
        queue.put((i, ans))
    queue.put(None)


def get_avgSim_u_par_bal(queue, iis, user_nodes, G):
    for i in iis:
        user = user_nodes[i]
        neighbors = [n for n in G.neighbors(user) if G.nodes[n]["type"] == "user"]
        ans = 0.0
        if neighbors:
            similarities = [
                cosine_similarity(
                    G.nodes[user]["Pref_u"].reshape(1, -1),
                    G.nodes[n]["Pref_u"].reshape(1, -1),
                )[0][0]
                for n in neighbors
                if "Pref_u" in G.nodes[n]
            ]

            if similarities:
                avgSim_u = np.mean(similarities)
                ans = float(avgSim_u)
        queue.put((i, ans))
    queue.put(None)


def calculate_user_clustering(graph):
    user_nodes = [
        node for node, attr in graph.nodes(data=True) if attr.get("type") == "user"
    ]
    user_graph = graph.subgraph(user_nodes)

    if isinstance(user_graph, nx.DiGraph):
        user_graph = user_graph.to_undirected()

    return nx.average_clustering(user_graph)


def process_enron_small(datasets_folder, dataset_filename):
    dataset_path = os.path.join(datasets_folder, dataset_filename)
    print("Computing graph for Enron small dataset")
    data = pd.read_csv(dataset_path, names=["file", "message"], skiprows=1)
    G = nx.DiGraph()
    for index, row in tqdm(data.iterrows()):
        sender, receivers, text = analyser_email_details(row["message"])
        if sender and receivers and text:
            if sender not in G:
                G.add_node(sender, type="user")
            text_node_id = f"text_{index}"
            G.add_node(text_node_id, type="text", content=text)
            G.add_edge(sender, text_node_id, relation="sent")
            for receiver in receivers.split(","):
                receiver = receiver.strip()
                if receiver:
                    if receiver not in G:
                        G.add_node(receiver, type="user")
                    G.add_edge(text_node_id, receiver, relation="received")
    return G


def process_enron(datasets_folder, dataset_filename):
    dataset_path = os.path.join(datasets_folder, dataset_filename)
    print("Computing graph for Enron dataset")
    data = pd.read_csv(dataset_path, names=["file", "message"], skiprows=1)
    G = nx.DiGraph()
    for index, row in tqdm(data.iterrows()):
        sender, receivers, text = analyser_email_details(row["message"])
        if sender and receivers and text:
            if not sender.endswith("@enron.com"):
                continue
            if sender not in G:
                G.add_node(sender, type="user")
            text_node_id = f"text_{index}"
            G.add_node(text_node_id, type="text", content=text)
            G.add_edge(sender, text_node_id, relation="sent")
            for receiver in receivers.split(","):
                receiver = receiver.strip()
                if receiver and receiver.endswith("@enron.com"):
                    if receiver not in G:
                        G.add_node(receiver, type="user")
                    G.add_edge(text_node_id, receiver, relation="received")
    return G


def process_yelp(datasets_folder, dataset_filename):

    dataset_path_users = os.path.join(datasets_folder, dataset_filename["users"])
    dataset_path_reviews = os.path.join(datasets_folder, dataset_filename["reviews"])
    dataset_path_businesses = os.path.join(
        datasets_folder, dataset_filename["businesses"]
    )
    dataset_path_tips = os.path.join(datasets_folder, dataset_filename["tips"])

    G = nx.DiGraph()

    print("Processing Yelp users..")
    with gzip.open(dataset_path_users, "r") as f:
        for line in tqdm(f):
            user = json.loads(line.strip())
            user_id = user["user_id"]
            G.add_node(user_id, type="user", name=user.get("name", "Unknown"))

    print("Processing Yelp businesses..")
    with gzip.open(dataset_path_businesses, "r") as f:
        for line in tqdm(f):
            business = json.loads(line.strip())
            business_id = business["business_id"]
            G.add_node(
                business_id, type="business", name=business.get("name", "Unknown")
            )

    business_to_users = defaultdict(set)

    print("Processing Yelp reviews..")
    with gzip.open(dataset_path_reviews, "r") as f:
        for line in tqdm(f):
            try:
                review = json.loads(line.strip())
                review_id = review["review_id"]
                user_id = review["user_id"]
                business_id = review["business_id"]
                text_content = review["text"]

                G.add_node(review_id, type="text", content=text_content)

                if user_id in G:
                    G.add_edge(user_id, review_id, relation="wrote")

                if business_id in G:
                    G.add_edge(business_id, review_id, relation="reviewed_in")

                business_to_users[business_id].add(user_id)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")
                continue

    print("Processing Yelp tips..")
    with gzip.open(dataset_path_tips, "r") as f:
        for line in tqdm(f):
            tip = json.loads(line.strip())
            user_id = tip["user_id"]
            business_id = tip["business_id"]
            text_content = tip["text"]

            tip_node_id = f"tip_{user_id}_{business_id}"
            G.add_node(tip_node_id, type="tip", content=text_content)

            if user_id in G:
                G.add_edge(user_id, tip_node_id, relation="gave_tip")

            if business_id in G:
                G.add_edge(business_id, tip_node_id, relation="tipped_at")

    print("Creating user-user connections based on shared businesses..")
    for business_id, users in business_to_users.items():
        if len(users) > 1:
            user_list = list(users)
            for i, user1 in enumerate(user_list):
                for user2 in user_list[i + 1 :]:
                    if not G.has_edge(user1, user2):
                        G.add_edge(user1, user2, relation="shared_business")

    remove_isolated_nodes(G)

    return G


def process_dblp(datasets_folder, dataset_filename):
    dataset_file = os.path.join(datasets_folder, dataset_filename)
    print("Processing DBLP dataset")

    G = nx.DiGraph()

    with gzip.open(dataset_file, "r") as infile:
        paper_iterator = ijson.items(infile, "item")
        paper_count = 0
        for paper in tqdm(paper_iterator, desc="Processing papers"):
            paper_count += 1

            paper_id = paper.get("title")
            if not paper_id:
                continue

            title = paper.get("title", "No title")
            authors = paper.get("authors", [])
            abstract = paper.get("abstract", "No abstract")

            G.add_node(paper_id, type="text", content=title, abstract=abstract)

            for author in authors:
                author_name = author.get("name", "Unknown Author")
                if author_name not in G:
                    G.add_node(author_name, type="user")
                G.add_edge(author_name, paper_id, relation="authored")

            for i, author in enumerate(authors):
                author_name = author.get("name", "Unknown Author")
                for j in range(i + 1, len(authors)):
                    co_author_name = authors[j].get("name", "Unknown Author")
                    if not G.has_edge(author_name, co_author_name):
                        G.add_edge(author_name, co_author_name, relation="coauthored")

    print(f"Total papers processed: {paper_count}")
    remove_isolated_nodes(G)

    return G


def remove_isolated_nodes(G):
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated nodes.")


@timing
def compute_tfidf_matrix(texts):
    print("Starting Tfidf")
    tfidf_vectoriser = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectoriser.fit_transform(texts)
    return tfidf_matrix


@timing
def compute_lda(tfidf_matrix):
    # This is random, to reproduce should be fixed
    # TOCHECK: n_components is a tunable parameter
    print("Starting LDA")
    lda = LatentDirichletAllocation(n_components=10, n_jobs=-1)
    lda_matrix = lda.fit_transform(tfidf_matrix)
    return lda_matrix


@timing
def process_data(
    dataset_name="enron",
    dataset_filename=None,
    datasets_folder="datasets/",
    output_folder="output/",
):
    output_folder_path = os.path.join(output_folder, dataset_name)
    os.makedirs(output_folder_path, exist_ok=True)

    initial_graph_path = os.path.join(
        output_folder_path, f"{dataset_name}_initial_graph.gml.gz"
    )
    dataset_path_nodes = os.path.join(
        output_folder_path, f"{dataset_name}_nodes.csv.gz"
    )
    dataset_path_edges = os.path.join(
        output_folder_path, f"{dataset_name}_edges.csv.gz"
    )

    if os.path.exists(initial_graph_path):
        print(f"{initial_graph_path} exists")
        return None

    if dataset_name == "enron_small":
        G = process_enron_small(datasets_folder, dataset_filename)
    elif dataset_name.startswith("yelp"):
        G = process_yelp(datasets_folder, dataset_filename)
    elif dataset_name == "enron":
        G = process_enron(datasets_folder, dataset_filename)
    elif dataset_name.startswith("dblp"):
        G = process_dblp(datasets_folder, dataset_filename)
    else:
        raise ValueError("Unknown dataset")

    # user_nodes = [node for node, attr in G.nodes(data=True) if attr.get("type", "").lower() == "user"]
    # user_subgraph = G.subgraph(user_nodes)
    # subgraph_undirected = user_subgraph.to_undirected()
    UG = G.to_undirected()
    partition = community_louvain.best_partition(UG)
    # partition = community_louvain.best_partition(subgraph_undirected, resolution=0.02)
    nx.set_node_attributes(G, partition, "community")

    number_of_nodes = len([node for node in G.nodes if G.nodes[node]["type"] == "user"])
    print(f"total number of nodes: {number_of_nodes}")

    texts = [
        G.nodes[node]["content"] for node in G.nodes if G.nodes[node]["type"] == "text"
    ]

    tfidf_matrix = compute_tfidf_matrix(texts)
    lda_matrix = compute_lda(tfidf_matrix)

    print("Computing sentiments")
    P = PatternAnalyzer()
    sentiments = parallel_for_balanced(
        get_sentiment_par_bal, (texts, P), range(len(texts)), DEBUG=True
    )
    sentiments = [sentiments[i] for i in range(len(texts))]

    V_t = np.hstack([lda_matrix, np.array(sentiments).reshape(-1, 1)])
    text_nodes = [node for node in G.nodes if G.nodes[node]["type"] == "text"]
    for node, vector in zip(text_nodes, V_t):
        G.nodes[node]["vector"] = vector

    user_nodes = [node for node in G.nodes if G.nodes[node]["type"] == "user"]

    for user in user_nodes:
        connected_text_nodes = [
            n for n in G.neighbors(user) if G.nodes[n]["type"] == "text"
        ]
        if connected_text_nodes:
            topic_sentiment_pairs = [G.nodes[n]["vector"] for n in connected_text_nodes]
            topics = np.array([pair[:-1] for pair in topic_sentiment_pairs])
            sentiments = np.array([pair[-1] for pair in topic_sentiment_pairs])

            avg_topic_vector = np.mean(topics, axis=0)
            topic_sums = np.sum(topics, axis=1)
            weighted_sentiment = np.average(sentiments, weights=topic_sums)

            G.nodes[user]["Pref_u"] = np.hstack([avg_topic_vector, weighted_sentiment])
        else:
            # TOCHECK: is it better to make this random?
            G.nodes[user]["Pref_u"] = np.zeros(lda_matrix.shape[1] + 1)

    for user1 in user_nodes:
        user_text_neighbors = [
            n for n in G.neighbors(user1) if G.nodes[n]["type"] == "text"
        ]
        for text_node in user_text_neighbors:
            connected_users = [
                n for n in G.neighbors(text_node) if G.nodes[n]["type"] == "user"
            ]
            for user2 in connected_users:
                if not G.has_edge(user1, user2):
                    G.add_edge(user1, user2)

    G.remove_nodes_from([n for n in G.nodes if G.nodes[n]["type"] != "user"])
    print(f"Total number of nodes {G.number_of_nodes()}")
    print(f"Total number of edges {G.number_of_edges()}")

    for user in user_nodes:
        neighbors = [n for n in G.neighbors(user) if G.nodes[n]["type"] == "user"]
        G.nodes[user]["num_user_neighbors"] = len(neighbors)

    print("Computing avgSim_u")
    avgSim_u = parallel_for_balanced(
        get_avgSim_u_par_bal, (user_nodes, G), range(len(user_nodes)), DEBUG=True
    )
    for i in range(len(user_nodes)):
        G.nodes[user_nodes[i]]["avgSim_u"] = avgSim_u[i]

    convert_numpy_to_list(G)

    save_nodes_to_csv(G, dataset_path_nodes)
    save_edges_to_csv(G, dataset_path_edges)

    nx.write_gml(G, initial_graph_path)
    return G


def convert_numpy_to_list(G):
    for node in G.nodes:
        if isinstance(G.nodes[node].get("Pref_u"), np.ndarray):
            G.nodes[node]["Pref_u"] = G.nodes[node]["Pref_u"].tolist()
        if isinstance(G.nodes[node].get("vector"), np.ndarray):
            G.nodes[node]["vector"] = G.nodes[node]["vector"].tolist()


def compute_local_clustering_coefficients(G):
    try:
        clustering_coefficients = nx.clustering(G)
        clustering_values = list(clustering_coefficients.values())
        summary = {
            "mean_clustering": np.std(clustering_values),
            # "std_clustering": np.mean(clustering_values),
            # "max_clustering": np.max(clustering_values),
        }
        print(f"Local clustering coefficient summary: {summary}")
        return summary
    except Exception as e:
        print(f"Error in computing local clustering coefficients: {e}")
        return None


def compute_initial_metrics(G, dataset_name="enron", output_folder="output/"):
    dataset_path_stats = os.path.join(
        output_folder, dataset_name, f"{dataset_name}_stats.json"
    )
    if os.path.exists(dataset_path_stats):
        print(f"{dataset_path_stats} exists")
        return
    if G is None:
        initial_graph_path = os.path.join(
            output_folder, dataset_name, f"{dataset_name}_initial_graph.gml.gz"
        )
        if not os.path.exists(initial_graph_path):
            assert False, "G must be computed or exist in the file"
        G = nx.read_gml(initial_graph_path)

    UG = G.to_undirected()
    density = nx.density(UG)
    print(f"density: {density}")

    avg_path_length = calculate_avg_path_length(UG)
    if avg_path_length is not None:
        print(f"avg path length: {avg_path_length}")

    assortativity = nx.degree_assortativity_coefficient(UG)
    print(f"assortativity coeff for degree: {assortativity}")

    assortativity_avgSim_u = attribute_assortativity_coefficient(UG, "avgSim_u")
    print(f"assortativity coeff for avgSim_u: {assortativity_avgSim_u}")

    local_clustering_coefficients = compute_local_clustering_coefficients(UG)
    print(f"Local clustering coefficients summary: {local_clustering_coefficients}")

    degree_sequence = sorted([d for n, d in UG.degree()], reverse=True)
    degree_mean = np.mean(degree_sequence)
    degree_std = np.std(degree_sequence)
    print(f"Degree distribution: {degree_sequence[:10]}")
    print(f"Degree mean: {degree_mean}")
    print(f"Degree standard deviation: {degree_std}")

    partition = community_louvain.best_partition(UG)
    modularity = community_louvain.modularity(partition, UG)
    print(f"Modularity: {modularity}")

    # save metrics
    data_stats = {
        "density": density,
        "avg_path_length": avg_path_length,
        "degree_assortativity": assortativity,
        "avg_similarity_assortativity": assortativity_avgSim_u,
        "local_clustering_coefficient": local_clustering_coefficients,
        "degree_distribution_top_10": degree_sequence[:10],
        "degree_mean": degree_mean,
        "degree_std": degree_std,
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "modularity": modularity,
    }

    with open(dataset_path_stats, "w") as json_file:
        json.dump(data_stats, json_file)
