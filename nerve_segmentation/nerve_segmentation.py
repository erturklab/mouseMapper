import os
import sys

def nerve_segmentation(config):
    pi2_loc = config["graph_extraction"]["parameters"]["path_pi2"]
    sys.path.append(pi2_loc)

