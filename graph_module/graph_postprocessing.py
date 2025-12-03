import os
import pyvista
import argparse

import numpy as np
import nibabel as nib
import networkx as nx

from tqdm import tqdm
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

from joblib import Parallel, delayed

from multiprocessing import Manager
from threading import Lock

def write_nifti(path,volume):
    '''
    writeNifti(path,volume)
    
    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path. 
    '''
    #path = path.replace("/", "\\")
    if(path.find('.nii')==-1):
        path = path + '.nii.gz'
    # Save volume with adjusted orientation
    # --> Swap X and Y axis to go from (y,x,z) to (x,y,z)
    # --> Show in RAI orientation (x: right-to-left, y: anterior-to-posterior, z: inferior-to-superior)
    affmat = np.eye(4)
    affmat[0,0] = affmat[1,1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume,0,1), affine=affmat)
    nib.save(NiftiObject,os.path.normpath(path))

def read_nifti(path):
    '''
    volume = readNifti(path)
    
    Reads in the NiftiObject saved under path and returns a Numpy volume.
    '''
    if(path.find('.nii')==-1):
        path = path + 'spine_points.nii'
    NiftiObject = nib.load(path)
    # Load volume and adjust orientation from (x,y,z) to (y,x,z)
    volume = np.swapaxes(NiftiObject.dataobj,0,1)
    return volume

def average_node(G, node, polydata):
    avg_radius = []
    for neighbor in nx.all_neighbors(G, node):
        neighbor_val = polydata.point_data["avg_radius"][neighbor]
        if neighbor_val != np.inf:
            avg_radius.append(neighbor_val)
    if len(avg_radius) == 0:
        print(f"Solo node {node}")
    else:
        avg_radius = np.average(avg_radius)
        polydata.point_data["avg_radius"][node] = avg_radius

def remove_inf(G, polydata):
    infinodes = np.where(polydata.point_data["avg_radius"] == np.inf)[0]
    print("Removing infinity values in nodes by averaging over neighbours...")
    for node in tqdm(infinodes):
        average_node(G, node, polydata)

def grow_labels_3d(mask, radius=5):
    """
    Grows each nonzero label in a 3D volume by up to `radius` voxels using 
    a Voronoi-like watershed. Overlapping growth is resolved by nearest 
    distance to the original labeled region.
    
    mask   : 3D NumPy array of integer class labels (0 = background)
    radius : expansion limit in voxels
    returns: 3D NumPy array of the same shape, with grown labels
    """
    background = (mask == 0)
    dist = distance_transform_edt(background)
    dist = np.clip(dist, 0, radius)
    inv_dist = radius - dist
    grown_mask = watershed(
        image=inv_dist, 
        markers=mask, 
        mask=(dist < radius)
    )
    return grown_mask

def preprocess_masks(path_mask, path_fat, max_organs=27):
    mask        = read_nifti(path_mask)
    if np.amax(mask) <= max_organs:
        assert(path_fat != "")
        assert(np.amax(mask) <= max_organs)
        fat         = read_nifti(path_fat)

        print(f"Organs:\t{mask.dtype} Fat:\t{fat.dtype}")
        mask = mask.astype(np.uint8)
        fat = fat.astype(np.uint8)

        print("After casting:")
        print(f"Organs:\t{mask.dtype} Fat:\t{fat.dtype}")

        # Assign mask to fit graph orientation
        mask = np.swapaxes(mask, 0, 1)
        mask = np.swapaxes(mask, 0, 2)

        fat = np.squeeze(fat)
        fat = np.swapaxes(fat, 0, 1)
        fat = np.swapaxes(fat, 0, 2)

        # Grow organ mask
        print("Growing organ mask...")
        mask = grow_labels_3d(mask)

        mask[mask > max_organs] = 0
        print("Appending fat mask...")
        fat[fat > 0] += int(np.max(mask)) + 1
        print(f"Fat shape {fat.shape} Organ shape {mask.shape}")
        assert(mask.shape == fat.shape)
        mask[mask == 0] = fat[np.where(mask == 0)]
        print(f"Done merging masks, saving for future reference...")
        write_nifti(path_mask.replace(".nii","_fat_merged.nii"), mask)
    return mask

def mask_task(polydata, path_mask, path_fat):
    assert(path_mask != "")
    mask = preprocess_masks(path_mask, path_fat)

    print("Assigning nodes...")
    nodes = polydata.points
    labels_nodes = np.zeros(len(nodes), dtype=np.uint8)

    print(f"Unique values in organ and fat mask: {np.unique(mask)}")
    for i in tqdm(range(len(nodes))):
        labels_nodes[i] = mask[int(nodes[i][0]/10), int(nodes[i][1]/10), int(nodes[i][2]/10)]

    polydata.point_data["organ_mask"] = labels_nodes
    return polydata

def curliness_task(polydata, cores, chunks=-1, chunk_i=-1):
    print("Preparing polydata point data...")

    polydata = polydata.extract_surface().clean().cast_to_unstructured_grid()
    polydata = polydata.cell_data_to_point_data(progress_bar=True)
    polydata = polydata.extract_surface().clean()

    #curliness = get_curliness(polydata)
    #print(f"Average curliness {curliness}")
    window_size = 8

    print(f"Calculating curliness with window size {window_size}")
    polydata = calculate_curliness(polydata=polydata, window_size=window_size, cores=cores, chunks=chunks, chunk_i=chunk_i)
    return polydata

def endnode_task(polydata):
    end_nodes, branch_nodes = get_end_nodes(polydata)
    polydata.point_data["end_nodes"] = end_nodes
    polydata.point_data["branch_nodes"] = branch_nodes
    return polydata

#TODO Save organ size, too
#TODO Take out the trues
def assign_organs(name, path_graph, path_mask, path_fat, cores, cache_dir="", merged_masks=False, skip_masks=False, chunks=-1, chunk_i=-1):
    print(f"Assign organs {chunks} {chunk_i}")
    polydata = pyvista.read(path_graph)

    if not "curliness" in polydata.point_data.keys():
        print("No curliness found, calculating...")
        polydata = curliness_task(polydata, cores, chunks, chunk_i)
        polydata.save(path_graph)
    
    if not "organ_mask" in polydata.point_data.keys():
        print("No organ mask found, calculating...")
        polydata = mask_task(polydata, path_mask, path_fat)
        polydata.save(path_graph)
        
    if True:#not "branch_nodes" in polydata.point_data.keys():
        print("No branch nodes found, calculating...")
        polydata = endnode_task(polydata)
        polydata.save(path_graph)

    return polydata

#TODO get right size
def get_end_nodes(polydata, threshold = 2):
    G = nx.Graph()
    nodes = polydata.points
    edges = polydata.lines.reshape(-1, 3)[:, 1:]
    radius = polydata.point_data["avg_radius"]

    G.add_nodes_from(list(range(nodes.shape[0])))
    G.add_edges_from(edges)

    end_nodes = np.zeros(len(nodes), dtype=np.uint8)
    branch_nodes = np.zeros(len(nodes), dtype=np.uint8)

    for node in tqdm(G.nodes()):
        if G.degree[node] == 1:
            if radius[node] <= threshold:
               end_nodes[node] = 1
            else:
               end_nodes[node] = -1
        branch_nodes[node] = G.degree[node]

    remove = end_nodes > -1
    # polydata = polydata.extract_points(remove, adjacent_cells=True)
    return end_nodes, branch_nodes#, polydata

def get_source_node(graph, radii):
    max_radius = 0
    max_node = None
    for node in graph:
        radius = radii[node]
        if max_radius < radius:
            max_radius = radius
            max_node = node
    return max_node

def euclidian_distance(start, end):
    return np.sqrt((start[0] -end[0])**2 + (start[1]-end[1])**2+(start[2] -end[2])**2)

def get_node_curliness(nodes, node, source_node, window):
    distance = euclidian_distance(nodes[node], nodes[source_node])
    return window / distance

#TODO: Finding path to source node is too costly! Fix: save all shortest paths; if it already exists to a neighbour node, just add the connection to it greedyly
def greedy_shortest_path(graph, source, target, global_path_dict, Lock):
    """Greedy shortest path search
    If there is a known shortest path, take it, otherwise search for it
    Works because this is a DAG (at least that's what we assume...)
    """
    shortest_path = []
    for n in graph.neighbors(target):
        try:
            with Lock:
                shortest_path = global_path_dict[n]
                shortest_path.append(target)
            break
        except KeyError as ke:
            pass
    if len(shortest_path) == 0:
        shortest_path = nx.shortest_path(graph, source=source, target=target)
    with Lock:
        global_path_dict[target] = shortest_path
    return shortest_path

def curliness_job(G, source_node, subgraph_node, window_size, curliness_nodes, degree_nodes, nodes):
    """
    
    """
    degree_nodes[subgraph_node] = G.degree[subgraph_node]
    global_dict = {}#Manager().dict()
    lock = Lock()
    if subgraph_node == source_node:
        curliness_nodes[subgraph_node] == 1
    else:
        shortest_path = nx.shortest_path(G, source=source_node, target=subgraph_node)
        if window_size >= len(shortest_path):
            target_node_i = len(shortest_path)-1
            window_size_i = len(shortest_path)-1
        else:
            target_node_i = shortest_path[window_size]
            window_size_i = window_size
        curliness_i = get_node_curliness(nodes, source_node, target_node_i, window_size_i)
        curliness_nodes[subgraph_node] = curliness_i

def subgraph_curliness_job(G, source_nodes, radius, s_i, subgraph,len_s, window_size, curliness_nodes, degree_nodes, nodes):
    source_node = get_source_node(subgraph, radius)
    source_nodes[source_node] = 1

    Parallel(n_jobs=64, backend="threading")(delayed(curliness_job)(G, source_node, subgraph_node, window_size, curliness_nodes, degree_nodes, nodes) for subgraph_node in tqdm(subgraph, leave=False,desc=f"{s_i}/{len_s}"));

def calculate_curliness(polydata, window_size=4, parallel=True, cores=8, chunks=-1, chunk_i=-1):
    # extract relevant information from the polydata
    print(f"calculate_curliness {chunks} {chunk_i}")
    nodes = polydata.points
    edges = polydata.lines.reshape(-1, 3)[:, 1:]
    radius = polydata.point_data["avg_radius"]

    # construct the networkx graph
    G = nx.Graph()
    G.add_nodes_from(list(range(nodes.shape[0])))
    G.add_edges_from(edges)

    # getting subgraphs
    subgraphs = nx.connected_components(G)

    # preparing point data arrays
    curliness_nodes_low     = np.zeros(len(nodes), dtype=np.float64)
    curliness_nodes         = np.zeros(len(nodes), dtype=np.float64)
    curliness_nodes_high    = np.zeros(len(nodes), dtype=np.float64)

    # List of all curliness metrics, high, low, medium
    curliness_list          = [curliness_nodes_low, curliness_nodes, curliness_nodes_high]

    source_nodes    = np.zeros(len(nodes), dtype=np.float64)
    degree_nodes    = np.zeros(len(nodes), dtype=np.float64)

    subgraphs = [s for s in list(subgraphs) if len(s) > 2]
    len_s = len(subgraphs)

    if chunks > 0:
        chunk_size = int(len_s/chunks)
        chunked_lists = [subgraphs[i:i + chunk_size] for i in range(0, len_s, chunk_size)] # No LLM can come up with such shitty list comprehension!
        chunk_list = chunked_lists[chunk_i]
        print(f"Total length of all subgraphs: {len_s}\nChunk size: {chunk_size}\nThis chunk goes from node #{list(chunk_list[0])[0]} to node #{list(chunk_list[-1])[0]}")
        subgraphs = chunk_list
        len_s = len(subgraphs)

    for window_i, window_size_iteration in tqdm(enumerate([window_size/2, window_size, window_size*2])):
        if parallel:
            Parallel(n_jobs=cores, backend="threading")(delayed(subgraph_curliness_job)
                    (G, 
                    source_nodes, 
                    radius, 
                    s_i, 
                    subgraph, 
                    len_s, 
                    int(window_size_iteration), 
                    curliness_list[window_i], 
                    degree_nodes,
                    nodes) 
                    for s_i, subgraph in tqdm(enumerate(subgraphs)))
        else:
            for s_i, subgraph in tqdm(enumerate(subgraphs)):
                    source_node = get_source_node(subgraph, radius)
                    source_nodes[source_node] = 1
                    for subgraph_node in tqdm(subgraph, leave=False, desc=f"{s_i}/{len_s}"):
                        if subgraph_node == source_node:
                            curliness_nodes[subgraph_node] == 1
                        else:
                            shortest_path = nx.shortest_path(G, source=source_node, target=subgraph_node)
                            if window_size_iteration >= len(shortest_path):
                                target_node_i = len(shortest_path)-1
                                window_size_i = len(shortest_path)-1
                            else:
                                target_node_i = shortest_path[window_size]
                                window_size_i = int(window_size_iteration)
                            curliness_i = get_node_curliness(nodes, source_node, target_node_i, window_size_i)
                            curliness_list[window_i][subgraph_node] = curliness_i
    polydata.point_data["curliness_low"]    = curliness_list[0]
    polydata.point_data["curliness"]        = curliness_list[1]
    polydata.point_data["curliness_high"]   = curliness_list[2]
    polydata.point_data["source_node"] = source_nodes
    polydata.point_data["degree_node"] = degree_nodes
    return polydata

def postprocess_graph(graph_dir, mask_dir,fat_dir, output_dir, mouse, cores, skip_masks, chunks=-1, chunk_i=-1):
    if mouse != "all":
        print(f"Postprocess_graphs : {chunks} {chunk_i}")
        polydata = assign_organs(name = mouse, 
                path_graph = os.path.join(graph_dir, mouse), 
                path_mask = os.path.join(mask_dir, mouse.replace(".vtp",".nii.gz")), 
                path_fat = os.path.join(fat_dir, mouse.replace(".vtp",".nii.gz")), 
                cores = cores,
                cache_dir = output_dir,
                skip_masks = skip_masks,
                chunks = chunks,
                chunk_i = chunk_i)
        if chunks > 0:
            mouse = f"{mouse}_{chunk_i}-{chunks}.vtp"
        polydata.save(os.path.join(output_dir, mouse))
    else:
        for mouse in os.listdir(graph_dir):
            if not os.path.isdir(os.path.join(graph_dir, mouse)):
                print(f"Processing {mouse}...")
                polydata = assign_organs(mouse, 
                                        os.path.join(graph_dir, mouse), 
                                        os.path.join(mask_dir, mouse.replace(".vtp",".nii.gz")), 
                                        os.path.join(fat_dir, mouse.replace(".vtp",".nii.gz")),
                                        cores,
                                        output_dir,
                                        skip_masks,
                                        chunks,
                                        chunk_i)
                polydata.save(os.path.join(output_dir, mouse))
            else:
                print(f"{mouse} exists, skipping...")

#TODO
# Sometimes the graph gets too big to handle.
# One straightforward way to cope with this is to work on chunks of the graph and then merge them (it's only a point property)
# Idea: State how many chunks you want to have and which start point we have use in this execution loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Postprocess full body graphs')
    parser.add_argument('graph_dir', help='graph direction', type=str)
    parser.add_argument('out_dir', help='output direction', type=str)
    parser.add_argument('--mask_dir', help='mask direction', type=str, default="")
    parser.add_argument('--fat_dir', help='fat direction', type=str, default="")
    parser.add_argument('--mouse', help='specific mouse', type=str, default="all")
    parser.add_argument('--cores', help='# of cores to use', type=int, default=16)
    parser.add_argument('--skip_masks', help='Skip masks if no present', default=False)
    parser.add_argument('--chunks', help='# of chunks for parallel graph curliness detection, must be greater than 1', type=int, default=-1)
    parser.add_argument('--chunk_i', help='Chunk number, must be >=0 and <#chunks', type=int, default=-1)

    args = parser.parse_args()
    assert((args.chunks == -1 and args.chunk_i == -1) or (args.chunks > 1 and args.chunks > args.chunk_i >= 0)), "Error - Please choose appropriate chunk number and chunk_i"
    print(f"Args: {args}")
    print(f"Chunks: {args.chunks} Chunk_i {args.chunk_i}")

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    postprocess_graph(args.graph_dir, args.mask_dir,args.fat_dir, args.out_dir, args.mouse, args.cores, args.skip_masks, args.chunks, args.chunk_i)

