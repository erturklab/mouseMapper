import os
import numpy as np
import pyvista as pv
from tqdm import tqdm

"""
Small script to merge chunked curliness graphs
"""

path_subgraphs  = ""
path_out        = ""

graph           = pv.read(os.path.join(path_graphs, os.listdir(path_graphs)[0]))
graph_name      = os.listdir(path_graphs)[0].split(".")[0] + "_curliness.vtp" # Fix the ugly names we got from chunking

# We already loaded the first graph, so start with 1
for sub_graph in tqdm(os.listdir(path_subgraphs)[1:]):
    subgraph_polydata = pv.read(os.path.join(path_subgraphs, sub_graph))
    intersection = len(np.intersect1d(np.where(graph["curliness"] > 0), np.where(subgraph_polydata["curliness"] > 0))) # Calculate the amount of vertices with more than one curliness value
    assert(intersection == 0) # If this throws an error there are multiple curliness values for one vertix - somethings not right

    graph["curliness_low"] += subgraph_polydata["curliness_low"]
    graph["curliness"] += subgraph_polydata["curliness"]
    graph["curliness_high"] += subgraph_polydata["curliness_high"]

graph.save(os.path.join(path_out, graph_name))
