import os
import json
import argparse

from organ_segmentation.organ_segmentation import organ_segmentation
from tissue_segmentation.tissue_segmentation import tissue_segmentation
from inflammation_segmentation.inflammation_segmentation import inflammation_segmentation
from nerve_segmentation.nerve_segmentation import nerve_segmentation
from graph_extraction.graph_extraction import graph_extraction


if __name__ == "__main__":
    #TODO parse arguments
    #TODO Load config
    #TODO Check flags

    path_config = ""

    config = {}

    with open(path_config, "rb") as file:
        config = json.load(file)

    flags = config["FLAGS"]

    if flags["perform_organ_segmentation"]:
        print("Performing organ segmentation...")
        organ_segmentation(config)
        print(f"Organ segmentation done; Saved to {config['organ_segmentation']['path_out']}")
        print("Please proceed to select the relevant patches")

    if flags["perform_tissue_segmentation"]:
        print("Performing tissue segmentation...")
        tissue_segmentation(config)
        print(f"Tissue segmentation done; Saved to {config['tissue_segmentation']['path_out']}")
        print("Please proceed to select the relevant patches")

    if flags["perform_inflammation_segmentation"]:
        print("Performing inflammation/CD68 segmentation...")
        inflammation_segmentation(config)
        print(f"Inflammation segmentation done; Saved to {config['inflammation_segmentation']['path_out']}")

    if flags["perform_nerve_segmentation"]:
        print("Performing nerve/UCHL1 segmentation...")
        nerve_segmentation(config)
        print(f"Nerve segmentation done; Saved to {config['nerve_segmentation']['path_out']}")

    if flags["perform_graph_extraction"]:
        print("Performing graph extraction...")
        graph_extraction(config)
        print(f"Graph extraction done; Saved to {config['graph_extraction']['path_out']}")
