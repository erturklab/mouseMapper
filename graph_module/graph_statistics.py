import os
import pyvista
import numpy as np
import nibabel as nib
import networkx as nx
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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

def get_organ_dict():
    organ_dict = {
            0:"background",
            1:"spleen",
            2:"kidney",
            3:"lungs",
            4:"heart",
            5:"brain",
            6:"gallbladder",
            7:"adrenal gl",
            8:"liver",
            9:"thymus",
            10:"LN",
            11:"stomach",
            12:"gut",
            14:"diaphragm",
            15:"pancreas",
            16:"abd_wall",
            17:"spinal_c",
            18:"testes",
            19:"prep_gland",
            20:"peyer_p",
            21:"vesicular_gl",
            22:"bladder",
            23:"Submandibular gland",
            24:"Sublingual gland",
            25:"Partoid gland",
            26:"Extraorbital lacrimal gland",
            27:"Orbital lacrimal gland",
            28:"???",
            29:"subFat",
            30:"vsFat",
            31:"muscle",
            32:"bone",
            33:"bone marrow"
        }
    return organ_dict

def get_organ_name(organ):
    organ_dict = get_organ_dict()
    return organ_dict[organ]

def get_graph(polydata):
    nodes = polydata.points
    edges = polydata.lines.reshape(-1, 3)[:, 1:]
    print("Generating graph")
    G = nx.Graph()
    G.add_nodes_from(list(range(nodes.shape[0])))
    G.add_edges_from(edges)
    print("Getting endnodes")
    endnodes = [node for node, degree in G.degree() if degree == 1 and polydata["avg_radius"][node] < 0.8]
    return G, endnodes

def get_single_organ_stat(polydata, graph, endnodes, name,diet, mask, organ, organs, degree, curliness, radius, organ_list):
    organ_nodes = [n for n in range(polydata.points.shape[0]) if organs[n] == organ]

    organ_sum   = np.sum([mask == organ])
    if organ > 0:
        organ_sum /= organ

    mean_curliness_low = np.mean(curliness[0][organs == organ])
    mean_curliness = np.mean(curliness[1][organs == organ])
    mean_curliness_high = np.mean(curliness[2][organs == organ])

    mean_radius = np.mean(radius[organs == organ])

    normalized_vertices = len(organ_nodes) / organ_sum

    normalized_edges = len(graph.edges(organ_nodes)) / organ_sum

    normalized_end_nodes = len([n for n in endnodes if organs[n] == organ]) / organ_sum
    normalized_branch_nodes = len([n for n in degree if organs[n] == organ and n > 2]) / organ_sum

    single_organ_df = pd.DataFrame({"Mouse Name":[name],
                "Diet":[diet],
                "Organ":[get_organ_name(organ)],
                "Curliness Low":[mean_curliness_low],
                "Curliness":[mean_curliness],
                "Curliness High":[mean_curliness_high],
                "Radius": [mean_radius],
                "Vertices": [normalized_vertices],
                "Edges": [normalized_edges],
                "End nodes":[normalized_end_nodes],
                "Branch nodes":[normalized_branch_nodes]
                })
    organ_list.append(single_organ_df)

def average_node(G, node, polydata):
    avg_radius = []
    inf_radius = []
    for neighbor in nx.all_neighbors(G, node):
        neighbor_val = polydata.point_data["avg_radius"][neighbor]
        if neighbor_val != np.inf:
            avg_radius.append(neighbor_val)
        else:
            inf_radius.append(neighbor)
    if len(avg_radius) == 0 and len(inf_radius) == len(list(nx.all_neighbors(G, node))):
        print(f"infinity circle for {node}")
        #TODO: What do we do here? We cant just exclude it from the Graph, we must modify the polydata, too...
        # for neighbor in list(nx.all_neighbors(G, node)):
        #     G.remove_node(neighbor)
        #     polydata.point_data["avg_radius"][neighbor] = 1
        polydata.point_data["avg_radius"][node] = 1
        G.remove_node(node)
    else:
        avg_radius = np.average(avg_radius)
        polydata.point_data["avg_radius"][node] = avg_radius
    return G

def remove_inf(G, polydata):
    infinodes = np.where(polydata.point_data["avg_radius"] == np.inf)[0]
    print("Removing infinity values in nodes by averaging over neighbours...")
    for node in tqdm(infinodes):
        average_node(G, node, polydata)

def get_missing_organs(unique_list):
    missing_organs = []
    organ_dict = get_organ_dict()
    for i in organ_dict.keys():
        if i not in unique_list:
            missing_organs.append(organ_dict[i])
    return missing_organs

def get_organ_statistics(polydata, name, mask, parallel=True):
    graph, endnodes = get_graph(polydata)

    # Remove isolated nodes
    graph.remove_nodes_from(list(nx.isolates(graph)))

    # Remove nodes with erroneous values
    remove_inf(graph, polydata)

    organs      = polydata.point_data["organ_mask"]
    print(f"Amount organs: \t {name} \t {len(np.unique(organs))}")
    print(f"Missing organs:\n{get_missing_organs(np.unique(organs))}\n")
    radius      = polydata.point_data["avg_radius"]
    curliness   = [polydata.point_data["curliness_low"], polydata.point_data["curliness"], polydata.point_data["curliness_high"]]
    degree      = polydata.point_data["branch_nodes"]
    if "hfd" in name:
        diet = "hfd"
    else:
        diet = "chow"
    
    organ_list  = []

    if parallel:
        Parallel(n_jobs = len(np.unique(organs)), backend="threading")(delayed(get_single_organ_stat)
                (polydata = polydata,
                graph = graph,
                endnodes = endnodes,
                name = name,
                diet = diet,
                mask = mask,
                organ = organ,
                organs = organs,
                curliness = curliness,
                degree = degree,
                radius = radius,
                organ_list = organ_list) for organ in tqdm(np.unique(organs), leave=False))
    else:
        for organ in tqdm(np.unique(organs), leave=False, desc=f"{name} {get_organ_name(organ)}"):
            get_single_organ_stat(polydata = polydata,
                graph = graph,
                endnodes = endnodes,
                name = name,
                diet = diet,
                mask = mask,
                organ = organ,
                organs = organs,
                curliness = curliness,
                radius = radius,
                degree = degree,
                organ_list = organ_list)

    organ_df = pd.concat(organ_list, ignore_index=True)

    return organ_df

def get_merged_organ_stats(path_graph, path_mask, item, df_list):
    polydata = pyvista.read(path_graph + item)
    mask = read_nifti(os.path.join(path_mask, item.replace(".vtp",".nii.gz")))
    df = get_organ_statistics(polydata, item.split(".")[0], mask)
    df_list.append(df)

def main():
    """
    This script generates a dataframe based on the organ annotated graphs.
    """
    path_graph = ""
    path_mask  = ""

    df_list = []

    for item in tqdm(os.listdir(path_graph)):
        get_merged_organ_stats(path_graph, path_mask, item, df_list)

    df = pd.concat(df_list, ignore_index=True)

    df.to_csv("organ_statistics.csv")

if __name__ == "__main__":
    main()
