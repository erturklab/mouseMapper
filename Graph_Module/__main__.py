import os
import json
import argparse
from graph_extraction import graph_extraction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph extraction module")
    parser.add_argument("config", metavar="config", type=str, nargs="*", default="config.json", help="Path for the config file; default is in the same folder as the __main__.py file (./config.json)")

    args = parser.parse_args()

    config_location = args.config

    if type(config_location) == type([]):
        config_location = config_location[0]

    # Load settings
    config = {}
    with open(config_location,"r") as file:
        print(f"Loading {config_location}")
        config = json.loads(file.read())

    # path_config = "./config.json"

    # config = {}

    # with open(path_config, "rb") as file:
    #     config = json.load(file)

    print("Performing graph extraction...")
    graph_extraction(config)
    print(f"Graph extraction done; Saved to {config['graph_extraction']['path_out']}")
