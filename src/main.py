#! /usr/bin/env python3

import argparse

from original_graph_metrics import process_data, compute_initial_metrics
from create_global import create_global_graph
from to_ego_graphs import create_ego_graphs
from noisy_ego_graphs import process_ego_graphs
from synthesized_graph import compute_synthesized_global_graph
from stats import compute_metrics


datasets_root_folder = "datasets"
output_root_folder = "output"

data_filenames = {
    "enron_small": "emails.csv",
    "enron": "emails.csv.gz",
    "yelp": {
        "users": "yelp_academic_dataset_user.json.gz",
        "reviews": "yelp_academic_dataset_review.json.gz",
        "tips": "yelp_academic_dataset_tip.json.gz",
        "businesses": "yelp_academic_dataset_business.json.gz",
    },
    "yelp_filtered_nv": {
        "users": "yelp_filtered_user_nv.json.gz",
        "reviews": "yelp_filtered_review_nv.json.gz",
        "tips": "yelp_filtered_tip_nv.json.gz",
        "businesses": "yelp_filtered_business_nv.json.gz",
    },
    "dblp": "dblp_v14.json.gz",
    "dblp_filtered": "dblp_filtered_topic.json.gz",
}


def run_pipeline(dataset, epsilon=2.0):
    print(f"Running pipeline for dataset: {dataset} with epsilon: {epsilon}")

    dataset_folder = f"{datasets_root_folder}/{dataset}"

    G = process_data(
        dataset_name=dataset,
        dataset_filename=data_filenames[dataset],
        datasets_folder=dataset_folder,
        output_folder=output_root_folder,
    )

    compute_initial_metrics(G, dataset, output_root_folder)

    G = create_global_graph(dataset, output_root_folder)

    create_ego_graphs(G, dataset, output_root_folder)

    process_ego_graphs(G, epsilon, dataset, output_root_folder)

    Gs = compute_synthesized_global_graph(epsilon, dataset, output_root_folder)

    compute_metrics(G, Gs, epsilon, dataset, output_root_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("d", type=str, help="database name")
    args = parser.parse_args()
    if args.d:
        run_pipeline(args.d, epsilon=2.0)
    else:
        run_pipeline("enron_small", epsilon=2.0)  # Enron small
        run_pipeline("enron", epsilon=2.0)  # Enron
        run_pipeline("dblp", epsilon=2.0)  # DBLP
        run_pipeline("yelp", epsilon=2.0)
