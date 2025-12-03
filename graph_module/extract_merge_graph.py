import os
import json
import numpy as np
import nibabel as nib
from voreen_vesselgraphextraction import extract_vessel_graph
import pandas as pd
from scipy.spatial.distance import cdist
import gzip
from tqdm import tqdm
import pyvista as pv


def patchify_voxel(volume_shape, patch_size, pad):
    p_h, p_w, p_d = patch_size
    pad_h, pad_w, pad_d = pad

    p_h = p_h -2*pad_h
    p_w = p_w -2*pad_w
    p_d = p_d -2*pad_d
    
    v_h, v_w, v_d = volume_shape

    # Calculate the number of patch in ach axis
    n_w = np.ceil(1.0*(v_w-p_w)/p_w+1)
    n_h = np.ceil(1.0*(v_h-p_h)/p_h+1)
    n_d = np.ceil(1.0*(v_d-p_d)/p_d+1)

    n_w = int(n_w)
    n_h = int(n_h)
    n_d = int(n_d)

    pad_1 = (n_h - 1) * p_h + p_h - v_h
    pad_2 = (n_w - 1) * p_w + p_w - v_w
    pad_3 = (n_d - 1) * p_d + p_d - v_d

    # volume = np.pad(volume, ((pad_h, pad_1+pad_h), (pad_w, pad_2+pad_w), (pad_d, pad_3+pad_d)), mode='reflect')
    h, w, d= v_h+2*pad_h+pad_1, v_w+2*pad_w+pad_2, v_d+2*pad_d+pad_3
    
    x_ = np.int32(np.linspace(pad_h, h-p_h-pad_h, n_h))
    y_ = np.int32(np.linspace(pad_w, w-p_w-pad_w, n_w))
    z_ = np.int32(np.linspace(pad_d, d-p_d-pad_d, n_d))
    
    ind = np.meshgrid(x_, y_, z_, indexing='ij')
    
    patch_list = []
    start_ind = []
    seq_ind = []
    for i, start in enumerate(list(np.array(ind).reshape(3,-1).T)):
        patch = [(start[0]-2*pad_h,start[0]+p_h), (start[1]-2*pad_w,start[1]+p_w), (start[2]-2*pad_d,start[2]+p_d)]
        patch_list.append(patch)
        start_ind.append(start)
        seq_ind.append([i//(y_.shape[0]*z_.shape[0]), (i%(y_.shape[0]*z_.shape[0]))//z_.shape[0], (i%(y_.shape[0]*z_.shape[0]))%z_.shape[0]])
        
    return patch_list, start_ind, seq_ind


def vvg_to_df(vvg_path):

    if vvg_path[-3:] == ".gz":
        with gzip.open(vvg_path, "rt") as gzipped_file:
            # Read the decompressed JSON data
            json_data = gzipped_file.read()
            data = json.loads(json_data)

    else:
        f = open(vvg_path)
        data = json.load(f)
        f.close()

    id_col = []
    pos_col = []
    node1_col = []
    node2_col = []
    minDistToSurface_col = []
    maxDistToSurface_col = []
    avgDistToSurface_col = []
    numSurfaceVoxels_col = []
    volume_col = []
    nearOtherEdge_col = []

    node_id_col = []
    node_voxel_col = []
    node_radius_col = []

    for i in data["graph"]["nodes"]:
        node_id_col.append(i["id"])
        node_voxel_col.append(i["voxels_"])
        node_radius_col.append(i["radius"])

    d_nodes = {'id': node_id_col,'voxel_pos' : node_voxel_col, "radius": node_radius_col}
    df_nodes = pd.DataFrame(d_nodes)
    df_nodes.set_index('id')

    for i in data["graph"]["edges"]:
        positions = []
        minDistToSurface = []
        maxDistToSurface = []
        avgDistToSurface = []
        numSurfaceVoxels = []
        volume = []
        nearOtherEdge = []

        try:
            i["skeletonVoxels"]
        except KeyError:
            #print("fail vessel")
            continue

        id_col.append(i["id"])
        node1_col.append(i["node1"])
        node2_col.append(i["node2"])

        for j in i["skeletonVoxels"]:
            positions.append(np.asarray(j["pos"]))
            minDistToSurface.append(j["minDistToSurface"])
            maxDistToSurface.append(j["maxDistToSurface"])
            avgDistToSurface.append(j["avgDistToSurface"])
            numSurfaceVoxels.append(j["numSurfaceVoxels"])
            volume.append(j["volume"])
            nearOtherEdge.append(j["nearOtherEdge"] )
        
        pos_col.append(positions)
        minDistToSurface_col.append(minDistToSurface)
        maxDistToSurface_col.append(maxDistToSurface)
        avgDistToSurface_col.append(avgDistToSurface)
        numSurfaceVoxels_col.append(numSurfaceVoxels)
        volume_col.append(volume)
        nearOtherEdge_col.append(nearOtherEdge)

    d = {'id': id_col,'pos' : pos_col, "node1" : node1_col, "node2" : node2_col, "minDistToSurface": minDistToSurface_col,"maxDistToSurface":maxDistToSurface_col, "avgDistToSurface":avgDistToSurface_col, "numSurfaceVoxels":numSurfaceVoxels_col, "volume":volume_col,"nearOtherEdge":nearOtherEdge_col }
    df_edge = pd.DataFrame(d)
    df_edge.set_index('id')
    return df_edge, df_nodes


############################################# Extract & Save Graphs #########################################
def process_crop(crop_idx, start_, seq_, zarr_data, config):
    image_name = f"seg_{str(seq_).replace(', ', '_').replace('[', '').replace(']', '')}"
    nii_path = os.path.join(config.output_dir, f'{image_name}.nii')
    graph_path = nii_path.replace(".nii", "_graph.vvg") 
    vtp_path    = nii_path.replace(".nii", "_graph.vtp")
    
    (start1,stop1), (start2,stop2), (start3,stop3) = crop_idx

    patch_size = config.patch_size
    pad = config.pad
    shape_ = zarr_data.shape

    # clip start and stop values
    start1_ = max(0, start1)
    start2_ = max(0, start2)
    start3_ = max(0, start3)
    stop1_ = min(zarr_data.shape[0], stop1)
    stop2_ = min(zarr_data.shape[1], stop2)
    stop3_ = min(zarr_data.shape[2], stop3)

    print("Loading data...")
    try:
        if not os.path.exists(graph_path) and not os.path.exists(vtp_path):
            if os.path.exists(nii_path):
                print(f"{nii_path} exists, loading...")
                niftiobj = nib.load(nii_path)
                print("Loaded nifti, getting fdata")
                small_data = niftiobj.get_fdata()

            else:
                print(f"{nii_path} doesn't exist, cropping from zarr...")
                small_data = zarr_data[start1_:stop1_, start2_:stop2_, start3_:stop3_]
            print("Data loaded.")

            pad1 = (start1_-start1, stop1-stop1_)
            pad2 = (start2_-start2, stop2-stop2_)
            pad3 = (start3_-start3, stop3-stop3_)

            seg_ = np.pad(small_data, (pad1, pad2, pad3), mode='constant')

            # check if the patch is all zeros
            if np.sum(seg_[pad[0]:patch_size[0]-pad[0], pad[1]:patch_size[1]-pad[1], pad[2]:patch_size[2]-pad[2]]) < 2.0:
                print('skipped patch', image_name, flush=True)
                return
        # else:
            # create a nifti image from the patch
            img_nii = nib.Nifti1Image(seg_.astype(np.float32), np.eye(4))

            if not os.path.exists(nii_path):
                nib.save(img_nii, nii_path)
            else:
                #TODO: Only works if output path is unique for each mouse, please ensure somewhere (current target is ambigous")
                print(f"{nii_path} exists, skipping save...")

            
            extract_vessel_graph(nii_path, 
                config.output_dir,
                config.tempdir,
                config.cachedir,
                config.bulge_size,
                config.workspace_file,
                config.voreen_tool_path,
                name=image_name,
                generate_graph_file=True,
                verbose=False,
            )
        else:
            print(f"{graph_path} exists, skipping Voreen step...")
            # print('Voreen Finished', flush=True)
            # graph_path = nii_path.replace(".nii", "_graph.vvg") 
        
        if not os.path.exists(vtp_path):
            cnt_edges_df, cnt_nodes_df = vvg_to_df(graph_path)
            nodes = np.array([np.array(item_).mean(0) for item_ in cnt_nodes_df['voxel_pos'].values])
            cnt_edges = np.stack([cnt_edges_df['node1'].values,cnt_edges_df['node2'].values], axis=1)
            # # cnt_radius = cnt_edges_df['radius'].values
            cnt_lines = cnt_edges_df['pos'].values
            avg_radius = cnt_edges_df['avgDistToSurface'].values
            min_radius = cnt_edges_df['minDistToSurface'].values
            max_radius = cnt_edges_df['maxDistToSurface'].values
            node_id = np.arange(len(nodes))

            edges = []
            avg_radii = []
            min_radii = []
            max_radii = []

            for cnt_, edge_, avg_rad_, min_rad_, max_rad_ in zip(cnt_lines, cnt_edges, avg_radius, min_radius, max_radius):

                new_node_id = np.arange(len(node_id), len(node_id)+len(cnt_))
                nodes = np.concatenate([nodes, cnt_], axis=0)
                node_id = np.concatenate((node_id, new_node_id))
                node_0, node_1 = node_id[edge_[0]], node_id[edge_[1]]
                node_pairs = [node_0]+list(new_node_id)+[node_1]
                edge_ = list(map(list, zip(*[node_pairs[:-1], node_pairs[1:]])))

                edges.extend(np.array(edge_))
                avg_rad_ = np.array(avg_rad_)
                min_rad_ = np.array(min_rad_)
                max_rad_ = np.array(max_rad_)

                # make a running avg of two consecutive avg_rad_ values, consider list size of 1
                if len(avg_rad_) > 1:
                    avg_rad_ = (avg_rad_[:-1] + avg_rad_[1:])/2
                    min_rad_ = (min_rad_[:-1] + min_rad_[1:])/2
                    max_rad_ = (max_rad_[:-1] + max_rad_[1:])/2
                    # append the first element of avg_rad_ in the begining and the last element of avg_rad_ at the end
                    avg_radii.extend(np.array([avg_rad_[0]]+list(avg_rad_)+[avg_rad_[-1]]))
                    min_radii.extend(np.array([min_rad_[0]]+list(min_rad_)+[min_rad_[-1]]))
                    max_radii.extend(np.array([max_rad_[0]]+list(max_rad_)+[max_rad_[-1]]))
                else:
                    avg_radii.extend(np.array([avg_rad_[0]]+[avg_rad_[-1]]))
                    min_radii.extend(np.array([min_rad_[0]]+[min_rad_[-1]]))
                    max_radii.extend(np.array([max_rad_[0]]+[max_rad_[-1]]))

            nodes = np.array(nodes)
            edges = np.array(edges)
            avg_radii = np.array(avg_radii)
            min_radii = np.array(min_radii)
            max_radii = np.array(max_radii)

            if nodes.shape[0] > 0 and edges.shape[0] > 0:
                # concatenate 2 at the start of each edge
                edges = np.concatenate((np.int32(2 * np.ones((edges.shape[0], 1))), edges), 1)

                # print("Nodes: ", nodes.shape)
                # print("Edges: ", edges.shape)
            
                # create a graph from the nodes and edges using pv
                graph = pv.PolyData()
                graph.points = nodes
                graph.lines = edges

                # put the avg_radii, min_radii, max_radii in the edge data
                graph.cell_data["avg_radius"] = avg_radii
                graph.cell_data["min_radius"] = min_radii
                graph.cell_data["max_radius"] = max_radii

                graph = graph.extract_surface().clean()
                graph = graph.clip_box(bounds=(pad[0], patch_size[0]-pad[0], pad[1], patch_size[1]-pad[1], pad[2],  patch_size[2]-pad[2]), invert=False)

                shift_h = start_[0] - 2*pad[0]
                shift_w = start_[1] - 2*pad[1]
                shift_d = start_[2] - 2*pad[2]
                graph = graph.translate((shift_h, shift_w, shift_d))
                graph = graph.clip_box(bounds=(0, shape_[0], 0, shape_[1], 0,  shape_[2]), invert=False)
                graph = graph.extract_surface().clean()

                # check if the graph is empty
                if graph.n_points == 0 or graph.n_cells == 0:
                    print('skipped patch', image_name, flush=True)
                    if os.path.exists(nii_path):
                        os.remove(nii_path)
                    return
                else:
                    # save the graph
                    # graph.save(nii_path.replace(".nii", "_graph.vtp"))
                    graph.save(vtp_path)
                    print('saved patch', image_name, flush=True)
                    os.remove(nii_path)
                    return
            
            else:
                os.remove(nii_path)
                print('skipped patch', image_name, flush=True)
                return
        
    except Exception as e:
        print('extract_merge_graph.py', str(e), flush=True)
        return

############################################# Merge Graphs #############################################

def match_features_one_to_one(features1, features2, threshold):
    """
    Matches features from two sets based on least Euclidean distance, ensuring one-to-one matching,
    and returns matches and unmatched features.

    Parameters:
        features1 (numpy.ndarray): First feature set, shape (N, 3).
        features2 (numpy.ndarray): Second feature set, shape (M, 3).
        threshold (float): Maximum distance for a match.

    Returns:
        matched1 (numpy.ndarray): Matched features from the first set.
        matched2 (numpy.ndarray): Matched features from the second set.
        unmatched1 (numpy.ndarray): Unmatched features from the first set.
        unmatched2 (numpy.ndarray): Unmatched features from the second set.
    """
    # Compute pairwise distances between features
    distances = cdist(features1, features2, metric='euclidean')
    
    # Create arrays to track matched features
    matched1_indices = []
    matched2_indices = []
    
    # Flatten the distance matrix and sort by distance
    all_distances = []
    for i, row in enumerate(distances):
        for j, dist in enumerate(row):
            all_distances.append((dist, i, j))
    all_distances.sort(key=lambda x: x[0])
    
    # Process matches in order of increasing distance
    for dist, i, j in all_distances:
        if dist >= threshold:
            continue
        if i in matched1_indices or j in matched2_indices:
            continue
        matched1_indices.append(i)
        matched2_indices.append(j)
    
    # Extract matched features
    matched1 = features1[matched1_indices]
    matched2 = features2[matched2_indices]
    
    # Find unmatched features
    unmatched1 = np.delete(features1, matched1_indices, axis=0)
    unmatched2 = np.delete(features2, matched2_indices, axis=0)
    
    return matched1_indices, matched2_indices, matched1, matched2, unmatched1, unmatched2


def merge_graphs(vtp_files, output_dir, span):
    edge_list = {}
    avg_radius_list = {}
    min_radius_list = {}
    max_radius_list = {}
    for file_ in tqdm(vtp_files, desc='Gathering All Edges', ncols=100):
        # get the file index from the file name
        file_index = file_.replace('seg_', '').replace('_graph.vtp', '')
        # convert file_index to a list of integers
        a = [int(i) for i in file_index.split('_')]
        
        # increase the value of each element of the file_index by 1 at a time
        candidate_index = [[a[0]+1, a[1], a[2]], [a[0], a[1]+1, a[2]], [a[0], a[1], a[2]+1]]

        # check if the file corresponding to the candidate_index exists
        for b in candidate_index:
            file_name = f"seg_{str(b).replace(', ', '_').replace('[', '').replace(']', '')}_graph.vtp"
            # check if the file exists in the vtp_files
            if file_name in vtp_files:
                # load graph corresponding to a
                graph_a = pv.read(os.path.join(output_dir, file_))

                # clean the graph
                graph_a = graph_a.extract_surface().clean()
            
                # load graph corresponding to b
                graph_b = pv.read(os.path.join(output_dir, file_name))
                # clean the graph
                graph_b = graph_b.extract_surface().clean()

                # get their nodes
                nodes_a = graph_a.points
                nodes_b = graph_b.points

                # get their edges
                edges_a = np.array(graph_a.lines).reshape(graph_a.n_lines,-1)[:,1:]
                edges_b = np.array(graph_b.lines).reshape(graph_b.n_lines,-1)[:,1:]

                # check if edge_b has duplicate edges
                edges_b = np.unique(edges_b, axis=0)

                # get the avg_radius, min_radius, max_radius
                avg_radius_a = graph_a.cell_data['avg_radius']
                avg_radius_b = graph_b.cell_data['avg_radius']
                min_radius_a = graph_a.cell_data['min_radius']
                min_radius_b = graph_b.cell_data['min_radius']
                max_radius_a = graph_a.cell_data['max_radius']
                max_radius_b = graph_b.cell_data['max_radius']

                # get in which entry a and b is exactly same
                dim = np.where(np.array(a) != np.array(b))[0][0]

                idx_ = np.max([a[dim], b[dim]])

                # get nodes which dim is within 5 voxel distance
                id_a = np.where(np.abs(nodes_a[:, dim] - idx_*span[dim]) < 5) #FIXME: Hardcoded threshold
                id_b = np.where(np.abs(nodes_b[:, dim] - idx_*span[dim]) < 5) #FIXME: Hardcoded threshold

                id_a_, id_b_, matched1, matched2, unmatched1, unmatched2 = match_features_one_to_one(nodes_a[id_a], nodes_b[id_b], 6) #FIXME: Hardcoded threshold

                # print('############################################################################')

                # if id_a_ and id_b_ are not empty
                if len(id_a_) > 0 and len(id_b_) > 0:
                    id_a = id_a[0][id_a_]
                    id_b = id_b[0][id_b_]
                    edge_list[str(a)+'+'+str(b)] = list(zip(id_a, id_b))

                    # for each element in id_a check in which row it is present in edges_a
                    edge_a = [np.where(np.isin(edges_a, id_a_))[0] for id_a_ in id_a]
                    edge_b = [np.where(np.isin(edges_b, id_b_))[0] for id_b_ in id_b]

                    avg_radius_list[str(a)+'+'+str(b)] = [(np.mean(avg_radius_a[edge1_])+np.mean(avg_radius_b[edge2_]))/2.0  for edge1_, edge2_ in zip(edge_a, edge_b)]
                    min_radius_list[str(a)+'+'+str(b)] = [(np.min(min_radius_a[edge1_])+np.min(min_radius_b[edge2_]))/2.0  for edge1_, edge2_ in zip(edge_a, edge_b)]
                    max_radius_list[str(a)+'+'+str(b)] = [(np.max(max_radius_a[edge1_])+np.max(max_radius_b[edge2_]))/2.0  for edge1_, edge2_ in zip(edge_a, edge_b)]


    # create empty nodes and edges
    nodes = []
    edges = []
    avg_radii = []
    min_radii = []
    max_radii = []
    start_nodes = {}
    num_node = 0
    for file_ in tqdm(vtp_files, desc='Merging Graphs', ncols=100):
        # get the file index from the file name
        file_index = file_.replace('seg_', '').replace('_graph.vtp', '')
        # convert file_index to a list of integers
        a = [int(i) for i in file_index.split('_')]
        graph = pv.read(os.path.join(output_dir, file_))
        nodes.extend(list(graph.points))
        edges.extend(list(np.array(graph.lines).reshape(graph.n_lines,-1)[:,1:]+num_node))
        avg_radii.extend(list(graph.cell_data['avg_radius']))
        min_radii.extend(list(graph.cell_data['min_radius']))
        max_radii.extend(list(graph.cell_data['max_radius']))
        start_nodes[str(a)] = num_node
        num_node += len(graph.points)

    # add edges from the edge_list
    for key, edges_ in tqdm(edge_list.items(), desc='Accumulation New Edges', ncols=100):
        a, b = key.split('+')
        avg_radii_ = avg_radius_list[key]
        min_radii_ = min_radius_list[key]
        max_radii_ = max_radius_list[key]
        for v, avg_radius_, min_radius_, max_radius_ in zip(edges_, avg_radii_, min_radii_, max_radii_):
            edges.append([v[0]+start_nodes[str(a)], v[1]+start_nodes[str(b)]])
            avg_radii.append(avg_radius_)
            min_radii.append(min_radius_)
            max_radii.append(max_radius_)


    nodes = np.array(nodes)
    edges = np.array(edges)
    print(f"Edges shape {edges.shape}")
    edges = np.concatenate((np.int32(2 * np.ones((edges.shape[0], 1))), edges), 1)
    avg_radii = np.array(avg_radii)
    min_radii = np.array(min_radii)
    max_radii = np.array(max_radii)


    # create a graph from the nodes and edges using pv
    graph = pv.PolyData()
    graph.points = nodes
    graph.lines = edges

    # graph = pv.UnstructuredGrid(edges.flatten(), np.array([4] * len(edges)), nodes)
    # put the avg_radii, min_radii, max_radii in the edge data
    print(f"Cell size {graph.n_cells}\t|\tRadii:\
            \n\t\t\t Average : {avg_radii.shape}\
            \n\t\t\t Minimum : {min_radii.shape}\
            \n\t\t\t Maximum : {max_radii.shape}\
            ")
    graph.cell_data["avg_radius"] = avg_radii[:graph.n_cells]
    graph.cell_data["min_radius"] = min_radii[:graph.n_cells]
    graph.cell_data["max_radius"] = max_radii[:graph.n_cells]

    graph = graph.extract_surface().clean()

    return graph
