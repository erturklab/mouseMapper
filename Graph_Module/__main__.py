import os
import json
import argparse
from graph_extraction import graph_extraction


if __name__ == "__main__":
    path_config = "./config.json"

    config = {}

    with open(path_config, "rb") as file:
        config = json.load(file)

    print("Performing graph extraction...")
    graph_extraction(config)
    print(f"Graph extraction done; Saved to {config['graph_extraction']['path_out']}")
