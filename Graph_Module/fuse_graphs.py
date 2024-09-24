import os
import dill
import numpy as np
from tqdm import tqdm
from math import sqrt




#TODO : Multiple thresholds? 
def get_adjacent_graph_by_coords(graph_dict, adjacent_graph_coords):
    """Find the relevant graph by coordinates
    """
    result_graph = {}
    for k in graph_dict.keys():
        if graph_dict[k]["coords"] == adjacent_graph_coords:
            result_graph = graph_dict[k]
    return result_graph

def distance(node_1, node_2):
    """
    Calculate euclidian distance between two vertices
    """
    distance = sqrt(
                (node_1[0]-node_2[0])**2 +
                (node_1[1]-node_2[1])**2 +
                (node_1[2]-node_2[2])**2
                )
    return distance

def add_zyx_offset(block_coords, block_size, graph):
    """
    Add offset to block coordinates, block metadata comes from file names
    Since python is pass-by-reference, there is no need to return the graph
    In order to ease future calculations, offset in pixel is still returned
    Args:
        block_coords    : 
        block_size      :
        graph           :
    Returns:
        block_offset
    """
    for vertex_i, vertex in enumerate(graph["vertices"]):
        print("-"*30)
        print(f"Len vertices:\t{len(graph['vertices'])}")
        try:
            print(f"Vertex\t{vertex}\nCoords\t{block_coords}\nSize\t{block_size}")
            vertex[0] += block_coords[0] * block_size[0]  
            vertex[1] += block_coords[1] * block_size[1]  
            vertex[2] += block_coords[2] * block_size[2]  
        except IndexError as ie:
            print(f"{ie} for\n Vertex\t{vertex}\nCoords\t{block_coords}\nSize\t{block_size}")
            #TODO Update edges, too
            graph["vertices"] = np.delete(graph["vertices"], vertex_i)

    return [block_coords[0] * block_size[0],block_coords[1] * block_size[1],block_coords[2] * block_size[2]]

def get_relevant_nodes(block_coordinates, max_size=[2,2,2]):
    """Retrieve adjacent nodes to given coordinates to decrease search space
    """
    relevant_nodes = []
    relevant_nodes.append([i + j for i,j in zip(block_coordinates, [-1, 0, 0])])
    relevant_nodes.append([i + j for i,j in zip(block_coordinates, [1, 0, 0])])
    relevant_nodes.append([i + j for i,j in zip(block_coordinates, [0, -1, 0])])
    relevant_nodes.append([i + j for i,j in zip(block_coordinates, [0, 1, 0])])
    relevant_nodes.append([i + j for i,j in zip(block_coordinates, [0, 0, -1])])
    relevant_nodes.append([i + j for i,j in zip(block_coordinates, [0, 0, 1])])

    final_nodes = []
    for r_n in relevant_nodes:
        # kick out edge cases
        final_node = []
        for coord_i, coord in enumerate(r_n):
            if coord >= 0 and coord <= max_size[coord_i]:
                final_node.append(coord)
                
        if len(final_node) > 2:
            final_nodes.append(final_node)

    # for z in [-1, 0, 1]:
    #     for y in [-1, 0, 1]:
    #         for x in [-1, 0, 1]:
    #             if not z == 0 and y == 0 and x == 0 and not(z**2 + y**2 + x**2 ):
    #                 neighbour = [block_coordinates[0] + z, block_coordinates[1] + y, block_coordinates[2] + x]
    #                 relevant_nodes.append(neighbour)
    # print(f"Block {block_coordinates} Relevant nodes:\n {final_nodes}")
    return final_nodes
                
def fuse_side(graphs, threshold, side):
    """
    Fuse graph along a given side
    Args:
        - side      : Side from the source node; -/+ indicate before or after, 1 = z, 2 = y, 3 = x
    """
    
    print(f"Fusing side {side} for graphs source {graphs[0]['coords']} target {graphs[1]['coords']}")
    new_edges = []

    # Zip coordinates and edges together to know which coordinate belongs to which numbered vertex
    node_source = zip(graphs[0]["graph"]["vertices"], range(np.amin(graphs[0]["graph"]["edges"]), np.amax(graphs[0]["graph"]["edges"])))
    node_target = zip(graphs[1]["graph"]["vertices"], range(np.amin(graphs[1]["graph"]["edges"]), np.amax(graphs[1]["graph"]["edges"])))

    # All graphs have their coords transformed, use block coords here
    minus = side > 0
    real_side = abs(side) - 1
    offset_source = graphs[0]["offset"][real_side]
    offset_target = graphs[1]["offset"][real_side]
    
    #TODO Formalize this
    block_size = [2000, 2000, 2000]


    if minus:
        node_source = [n for n in node_source if n[0][real_side] - offset_source < threshold]
        node_target = [n for n in node_target if n[0][real_side] - offset_target > block_size[real_side] - threshold]
    else:
        node_source = [n for n in node_source if n[0][real_side] - offset_source > block_size[real_side] - threshold]
        node_target = [n for n in node_target if n[0][real_side] - offset_target < threshold]

    
    # Add new edges if distance between nodes is below threshold
    for n_s in tqdm(node_source, desc="Traversing nodes..."):
        for n_t in tqdm(node_target, leave=False):
            node_distance = distance(n_s[0], n_t[0]) 
            if  node_distance < threshold:
                print(f"Found new edge between two nodes, distance {node_distance}",end="\r",flush=True)
                new_edge = [n_s[1], n_t[1]]
                new_edges.append(new_edge)

    new_edges = np.array(new_edges)
    return new_edges


def fuse_edges(graphs, threshold):
    """
    Fuse two graphs together given a threshold of how far the graphs can be apart
    Args:
        - graphs : list of graphs
        - threshold : maximum distance threshold
    """
    coords_source   = graphs[0]["coords"]
    coords_target   = graphs[1]["coords"]

    vertices_source = graphs[0]["graph"]["vertices"]
    vertices_target = graphs[1]["graph"]["vertices"]


    edges = graphs[0]["graph"]["edges"]

    # Update edge count for target edges
    graphs[1]["graph"]["edges"] += len(graphs[0]["graph"]["vertices"])

    # Check which side of the block has to be traversed
    side = [s -t for s, t in zip(coords_source, coords_target)]

    new_edges = []

    # Which side to check?
    if side[0] > 0:
        new_edges = fuse_side(graphs, threshold, 1)
    elif side[0] < 0:
        new_edges = fuse_side(graphs, threshold, -1)
    elif side[1] > 0:
        new_edges = fuse_side(graphs, threshold, 2)
    elif side[1] < 0:
        new_edges = fuse_side(graphs, threshold, -2)
    elif side[2] > 0:
        new_edges = fuse_side(graphs, threshold, 3)
    elif side[1] < 0:
        new_edges = fuse_side(graphs, threshold, -3)

    RED = '\033[91m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    END = '\033[0m'
    # Append new edges
    if len(new_edges) > 0:
        edges = np.concatenate([edges, new_edges])
        print(f"\nAppending {GREEN}{BOLD}{len(new_edges)} new edges{END}\n")
    else:
        print(f"\nAppending {RED}{BOLD}{len(new_edges)} new edges{END}\n")
    # Append untouched edges
    edges = np.concatenate([edges, graphs[1]["graph"]["edges"]])

    # Concat vertices
    vertices = np.concatenate([vertices_source, vertices_target])

    return vertices, edges

#TODO Don't remove small isolated sub graphs if you want to check all nodes...Maybe do that in post?
#TODO Rename already existing graphs (coords are in .raw file)
def fuse_graphs(path_graphs, path_out, threshold = 20):
    """
    Fuse multiple graphs from one image processed in blocks together into one graph

    """
    
    print(f"Fusing graphs in {path_graphs}")
    # graph dict
    graph_dict = {}

    max_blocks = [0, 0, 0]

    # First iteration to find out the maximum amount of blocks
    for graph_name in [x for x in os.listdir(path_graphs) if ".pickledump" in x]:
        print(f"Graph name:\n{graph_name}")
        graph_info = graph_name.split(".")[0]
        print(f"Graph info:\n{graph_info}")
        block_coords = [
                    int(graph_info.split("_")[2]),
                    int(graph_info.split("_")[3]),
                    int(graph_info.split("_")[4])
                ]
        if block_coords[0] > max_blocks[0]:
            max_blocks[0] = block_coords[0]

        if block_coords[1] > max_blocks[1]:
            max_blocks[1] = block_coords[1]

        if block_coords[2] > max_blocks[2]:
            max_blocks[2] = block_coords[2]

    print(f"Max blocks: {max_blocks}")

    # Load all graphs
    for graph_name in [x for x in os.listdir(path_graphs) if ".pickledump" in x]:
        print(f"Reading {graph_name}",end="\r",flush=True)
        graph = {}

        with open(os.path.join(path_graphs, graph_name), "rb") as file:
            graph = dill.load(file)

        graph_info = graph_name.split(".")[0]
        block_coords = [
                    int(graph_info.split("_")[2]),
                    int(graph_info.split("_")[3]),
                    int(graph_info.split("_")[4])
                ]
        block_size = [
                    2000, # graph_info.split("_")[6],
                    2000, # graph_info.split("_")[7],
                    2000 # graph_info.split("_")[8]
                ]
        
        # TODO block_size is wrong here, it should be the block_size of every block before that in the given axis. For now, we can use 2000 here for HFD blocks.
        block_offset = add_zyx_offset(block_coords, block_size, graph)

        # max_blocks = config["max_blocks"]

        adjacent_graphs = get_relevant_nodes(block_coords, max_blocks)

        graph_dict[graph_name]              = {}
        graph_dict[graph_name]["graph"]     = graph
        graph_dict[graph_name]["adjacent"]  = adjacent_graphs
        graph_dict[graph_name]["coords"]    = block_coords
        graph_dict[graph_name]["offset"]    = block_offset
    
    print("Read all subgraphs" + " "*20)
    
    checked_pairs = []
    final_edge, final_vertex = [], []

    # Now that all graphs are loaded, we can traverse them one by one 
    for graph in graph_dict.keys():
        for adjacent_graph_coords in graph_dict[graph]["adjacent"]:
            if not [graph_dict[graph]["coords"], adjacent_graph_coords] in checked_pairs and not [adjacent_graph_coords, graph_dict[graph]["coords"]] in checked_pairs:
                print(f"Fusing graphs\nSource graph\t{graph_dict[graph]['coords']}\tTarget graph coords\t{adjacent_graph_coords}")
                adjacent_graph = get_adjacent_graph_by_coords(graph_dict, adjacent_graph_coords)

                fused_vertices, fused_edges = fuse_edges([graph_dict[graph], adjacent_graph], threshold)
                final_vertex.extend(fused_vertices)
                final_edge.extend(fused_edges)

                checked_pairs.append([graph_dict[graph]["coords"]])

    final_graph_name = ("_").join(graph_info.split("_")[:2])
    final_graph = {}
    final_graph["vertices"]  = final_vertex
    final_graph["edges"]     = final_edge
    with open(os.path.join(path_out, final_graph_name) + ".pickledump", "xb") as file:
        dill.dump(final_graph, file)

