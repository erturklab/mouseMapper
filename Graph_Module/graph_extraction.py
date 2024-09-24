import os
import sys

import dill

import dask
import dask.array as da

import pandas as pd
import numpy as np
import nibabel as nib

from dask.diagnostics import ProgressBar
from dask.array.image import imread

from matplotlib import pyplot as plt
from pathlib import Path

from fuse_graphs import fuse_graphs

def read_nifti(path):
    '''
    volume = readNifti(path)

    Reads in the NiftiObject saved under path and returns a Numpy volume.
    '''
    if(path.find('.nii')==-1):
        path = path + '.nii'
    NiftiObject = nib.load(path)
    # Load volume and adjust orientation from (x,y,z) to (y,x,z)
    volume = np.swapaxes(NiftiObject.dataobj,0,1)
    return volume

def _get_bb(patch):
    """Get the bounding box for a cell in a binary matrix
    """
    mip_z = da.max(patch, axis=0)
    mip_x = da.max(patch, axis=1)
    mip_y = da.max(patch, axis=2)

    mip_z.compute()
    mip_x.compute()
    mip_y.compute()

    mip_z = np.nonzero(mip_z)
    mip_x = np.nonzero(mip_x)
    mip_y = np.nonzero(mip_y)


    bb = ((
        np.amin(mip_z[0]),
        np.amin(mip_x[0]), 
        np.amin(mip_y[0])
        ),(
        np.amax(mip_z[1]), 
        np.amax(mip_z[1]), 
        np.amax(mip_z[1])
        ))
    return bb

def from_npy(array):
    print(f"dtype {array.dtype}")
    array = array.astype(np.uint8)
    print(f"After casting {array.dtype}")
    bb = _get_bb(array)
    print(f"Bounding box {bb}")
    print("Creating new pi image")
    img = pi.newimage(array.dtype, array.shape[0], array.shape[1], array.shape[2])
    print("Setting data...")
    img.set_data(array)
    return img

def to_npy(img):
    return(img.get_data())

def load_binary(path_in):
    """
    Reading image depending on input type (NIFTI/TIFF stack)
    Args:
        path_in (str) : Input path
    """
    print("Reading image...")
    if ".nii.gz" in path_in:
        print("Reading nifti image...")
        binary = da.from_array(read_nifti(path_in))
    else:
        # Read input data
        print("Reading TIFF stack...")
        paths_in = Path(path_in, "*.tif*")
        if len(os.listdir(path_in)) > 0:
            with ProgressBar():
                binary = imread(str(paths_in))
        else:
            print(f"Path {path_in} empty, skipping...")
            return
    print("Read image, converting to uint8...")
    # binary = binary.astype(np.uint8)
    return binary

def fuse_measurements(path_out, mouse):
    """
    Fuse measurements of cut volumes together into one volume
    """
    fuse_graphs(path_out, os.path.join(path_out, mouse.replace(".nii.gz","")))

def graph_extraction(config):
    """
    Generate graphs of binary nerve segmentations and analyzes them afterwards
    """
    pi2location = config["graph_extraction"]["parameters"]["path_pi2"]
    sys.path.append(pi2location)

    lib = __import__('pi2py2')
    globals().update({k: getattr(lib, k) for k in dir(lib) if not k.startswith('_')})

    # from pi2py2 import *
    global pi
    pi = Pi2()

    path_in = config["graph_extraction"]["path_nerve_mask"]
    path_out = config["graph_extraction"]["path_out"]

    ProgressBar().register()

    if not os.path.exists(path_out):
        os.mkdir(path_out)

    for mouse in os.listdir(path_in):
        binary = load_binary(os.path.join(path_in, mouse))
        if config["graph_extraction"]["parameters"]["cut_volume"]:
            sub_shapes = [
                    int(binary.shape[0] / 2), 
                    int(binary.shape[1] / 4), 
                    int(binary.shape[2] / 4)
                    ]
            # binary = binary.rechunk({0: 2000, 1: 5000, 2: 5000})
            binary = binary.rechunk({0: sub_shapes[0], 1: sub_shapes[1], 2: sub_shapes[2]})
            blocks_shape = binary.blocks.shape
            block_shape = "not initialized"
            for z in range(blocks_shape[0]):
                for y in range(blocks_shape[1]):
                    for x in range(blocks_shape[2]):
                        if not os.path.exists(os.path.join(path_out, mouse, f"{mouse}_{z}_{y}_{x}.vtk")):
                            print(f"Skeletonizing {mouse} z {z}/{blocks_shape[0]} y {y}/{blocks_shape[1]} x {x}/{blocks_shape[2]}")
                            print("Computing..")
                            with ProgressBar():
                                block = binary.blocks[z, y, x].compute()
                                block_shape = block.shape
                            if da.max(block) > 0:
                                print(f"Computed, block has shape {block.shape}, starting pipeline...")
                                skeletonize_measurements(block, os.path.join(path_out, mouse), f"{mouse}_{z}_{y}_{x}_{block_shape[0]}_{block_shape[1]}_{block_shape[2]}", config, cut_processing=True)
                            else:
                                print(f"Block {z} {y} {x} of shape {block_shape} is empty, skipping...")
                        else:
                            print(f"Block {z} {y} {x} exists, skipping...")
            fuse_measurements(path_out, mouse)
        else:
            skeletonize_measurements(binary, os.path.join(path_out, mouse), mouse, config)

    analyze_measurements(path_out, path_out)

def skeletonize_measurements(binary, path_out, output_name, config, cut_processing=False):
    """
    Skeletonize a binary volumetric mask, postprocess the graph and save a distance-map enhanced graph.
    This script is based on the pi2 example given at https://pi2-docs.readthedocs.io/en/latest/examples/ex_vessel_graph.html#vessel-graph-example
    Args:
        binary (np.array): binary segmentation array
        path_out (str): Path of the output folder, will be created if not existing. Raw skeleton, measurements csv and vtk file is saved there
    """
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    if os.path.exists(os.path.join(path_out, f"{output_name}.vtk")):
        print("Volume exists, skipping..")
    else:
        print("Sanity checks")
        print(f"Shape {binary.shape}")
        if cut_processing:
            print(f"Min/Max {da.min(binary).compute()} {da.max(binary).compute()}")
        img = from_npy(binary)
        print("Converted to pi2 image")

        # Skeletonize
        print("Generating skeleton...")
        pi.surfaceskeleton(img, False)

        #intermediate_path = "./skeletonization_steps/skel_intermediate.raw"
        intermediate_path = os.path.join(path_out, f"{output_name}_skeleton_intermediate.raw")
        pi.writeraw(img, intermediate_path)

        # Save for later
        skel = to_npy(img)

        # Create a distance map
        print("Calculating distance map...")
        img = from_npy(binary)
        dmap = pi.newimage(ImageDataType.FLOAT32)

        pi.dmap(img, dmap)

        dmap_data = to_npy(dmap)

        # Trace skeleton
        print("Tracing skeleton...")
        skeleton = from_npy(skel)

        smoothing_sigma  = 2
        max_displacement = 2

        vertices         = pi.newimage(ImageDataType.FLOAT32)
        edges            = pi.newimage(ImageDataType.UINT64)
        measurements     = pi.newimage(ImageDataType.FLOAT32)
        points           = pi.newimage(ImageDataType.INT32)

        pi.tracelineskeleton(skeleton, vertices, edges, measurements, points, True, 1, smoothing_sigma, max_displacement)

        if config["graph_extraction"]["parameters"]["prune_skeleton"]:
            # Graph pruning
            pruning_threshold = config["graph_extraction"]["parameters"]["pruning_threshold"]
            pi.pruneskeleton(vertices, edges, measurements, points, pruning_threshold, False, True)

        # Convert to vtk format in order to get radius for each point and line
        print("Generating vtk image...")
        vtkpoints = pi.newimage()
        vtklines = pi.newimage()
        pi.getpointsandlines(vertices, edges, measurements, points, vtkpoints, vtklines)

        # Get radius for each point
        points_data = to_npy(vtkpoints)
        print(f"points_data shape {points_data.shape}")
        radius_points = np.zeros([points_data.shape[0]])
        if points_data.shape == (3,):
            points_data = np.array([points_data])
            print(points_data.shape)
        for i in range(0, points_data.shape[0]):
            p = points_data[i, :]
            r = dmap_data[int(p[1]), int(p[0]), int(p[2])]
            radius_points[i] = r

        # Next, we will remove all edges that has at least one free and and whose L/r < 2.
        # First, get edges, vertices, and branch length as NumPy arrays.
        old_edges   = to_npy(edges)
        vert_coords = to_npy(vertices)
        # The tracelineskeleton measures branch length by anchored convolution and returns it in the
        # measurements image.
        meas_data = to_npy(measurements)
        print(f"Measurements shape:\t{meas_data.shape}")
        print(f"Measurements:\n{meas_data}")
        try:
            if meas_data.shape == (3,):
                meas_data = np.array([meas_data])
            length_data = meas_data[:, 1]
        except IndexError as ie:
            print(f"{ie}:\n {binary.shape}\n {skel.shape}")
            print(f"Measurements:\n{meas_data}")
            print(f"Points:\n{points_data}")
            print(f"Lines:\n{to_npy(vtklines)}")
            exit()
        # Calculate degree of each vertex
        deg = {}
        for i in range(0, vert_coords.shape[0]):
            deg[i] = 0

        try:
            print(f"Old edges shape {old_edges.shape}")
            if old_edges.shape == (2,):
                old_edges = np.array([old_edges])
            for i in range(0, old_edges.shape[0]):
                deg[old_edges[i, 0]] += 1
                deg[old_edges[i, 1]] += 1
        except IndexError as ie:
            print(f"{ie}")
            exit()

        if config["graph_extraction"]["parameters"]["remove_small_nodes"]:
            # Determine which edges should be removed
            remove_flags = []
            for i in range(0, old_edges.shape[0]):
                    n1 = old_edges[i, 0]
                    n2 = old_edges[i, 1]

                    # Remove edge if it has at least one free end point, and if L/r < 2, where
                    # r = max(r_1, r_2) and r_1 and r_2 are radii at the end points or the edge.
                    should_remove = False
                    if deg[n1] == 1 or deg[n2] == 1:

                            p1 = vert_coords[n1, :]
                            p2 = vert_coords[n2, :]

                            r1 = dmap_data[int(p1[1]), int(p1[0]), int(p1[2])]
                            r2 = dmap_data[int(p2[1]), int(p2[0]), int(p2[2])]

                            r = max(r1, r2)
                            L = length_data[i]

                            if L < 2 * r:
                                    should_remove = True

                    ## Remove very short isolated branches, too.
                    if deg[n1] == 1 and deg[n2] == 1:
                            L = length_data[i]
                            if L < 5 / 0.75: # (5 um) / (0.75 um/pixel)
                                    should_remove = True

                    remove_flags.append(should_remove)

            remove_flags = np.array(remove_flags).astype(np.uint8)
            print(f"Before dynamic pruning: {old_edges.shape[0]} edges")

            # This call adjusts the vertices, edges, and measurements images such that
            # the edges for which remove_flags entry is True are removed from the graph.
            # Disable distributed processing for this - not yet implemented

            #intermediate_path = "./skeletonization_steps/skel_intermediate.raw"
            intermediate_path = os.path.join(path_out, f"{output_name}_intermediate.raw")
            pi.writeraw(img, intermediate_path)

            #img = pi.newimage
            #pi.readraw(img, intermediate_path)
            if np.sum(remove_flags > 0):
                print(f"Removing {np.sum(remove_flags)} edges")
                pi.removeedges(vertices, edges, measurements, points, remove_flags, True, True)
            else:
                print("No edge candidates detected, proceeding...")

        # Get average radius for each branch
        # Notice that the vtklines image has a special format that is detailed in
        # the documentation of getpointsandlines function.
        lines_data = to_npy(vtklines)
        radius_lines = []
        i = 0
        edge_count = lines_data[i]
        i += 1
        for k in range(0, edge_count):
            count = lines_data[i]
            i += 1

            R = 0
            for n in range(0, count):
                index = lines_data[i]
                i += 1
                p = points_data[index, :]
                R += dmap_data[int(p[1]), int(p[0]), int(p[2])]
            R /= count

            radius_lines.append(R)

        radius_lines = np.array(radius_lines)


        # Convert to vtk format again, now with smoothing the point coordinates to get non-jagged branches.
        vtkpoints = pi.newimage()
        vtklines = pi.newimage()
        pi.getpointsandlines(vertices, edges, measurements, points, vtkpoints, vtklines, smoothing_sigma, max_displacement)


        # Write to file
        #TODO: Could be empty, then array needs to be restructured
        print("Saving vtk image...")
        print(f"vtkpoints\t{to_npy(vtkpoints).shape}")
        print(f"vtklines\t{to_npy(vtklines).shape}")
        print(f"radius_points\t{radius_points.shape}")
        print(f"radius_lines\t{radius_lines.shape}\n{radius_lines}")
        if radius_lines.shape == (1,):
            radius_lines =  np.array([1., 1.])
        print(f"radius_lines\t{radius_lines.shape}\n{radius_lines}")
        pi.writevtk(vtkpoints, vtklines, os.path.join(path_out, f"{output_name}.vtk") , "radius", radius_points, "radius", radius_lines)


        # # Generate and save figures
        # print("Saving figures...")
        # plt.hist(radius_points, bins="auto")
        # plt.savefig(os.path.join(path_out, f"{output_name}_radius_points.png"))
        # plt.hist(radius_lines, bins="auto")
        # plt.savefig(os.path.join(path_out, f"{output_name}_radius_lines.png"))
        # plt.hist(deg.values(),bins="auto")
        # plt.savefig(os.path.join(path_out, f"{output_name}_degree.png"))

        # Save properties dict
        properties_dict = {
                "vertices":to_npy(vertices),
                "edges":to_npy(edges),
                "measurements":to_npy(measurements),
                "points":to_npy(points),
                "radius_points":radius_points,
                "radius_lines":radius_lines,
                "vtkpoints":to_npy(vtkpoints),
                "vtklines":to_npy(vtklines),
                "deg":deg.values()
                }
        with open(os.path.join(path_out, f"{output_name}_properties.pickledump"), "wb") as handle:
            dill.dump(properties_dict, handle)
        # Clean up
        if not config["FLAGS"]["save_raw"]:
            for item in os.listdir(path_out):
                if ".raw" in item:
                    os.remove(os.path.join(path_out, item))


def analyze_measurements(path_graphs, path_out):
    """
    Analyzing the skeleton properties
    Args:
        path_in (str): Path of the skeleton measurements
        path_out (str): Path of the output folder where the csv will be saved to
    """
    properties = {}
    columns = ["Mouse","Nerve Endings", "Avg Thickness (points)", "Max Thickness (points)", "Avg Thickness (line)", "Max Thickness (line)", "Vertices", "Edges","Avg Degree", "Max Degree", "Avg branching size", "Max branching size"]
    df = pd.DataFrame(columns=columns)
    for item in os.listdir(path_graphs):
        if os.path.isdir(f"{path_graphs}/{item}"):
            with open(f"{path_graphs}/{item}/{item}_properties.pickledump", "rb") as handle:
                properties[item] = dill.load(handle)

    print("Calculating nerve endings...")
    for item in sorted(properties.keys()):
        vertices = properties[item]["vertices"]
        edges    = properties[item]["edges"]
        deg = {}
        for i in range(0, len(properties[item]["vertices"])):
                deg[i] = 0

        for i in range(0, edges.shape[0]):
                deg[edges[i, 0]] += 1
                deg[edges[i, 1]] += 1

        leaf_nodes = []
        for i in range(0, edges.shape[0]):
                n1 = edges[i, 0]
                n2 = edges[i, 1]
                leaf_node = False
                if deg[n1] == 1 or deg[n2] == 1:
                    leaf_node = True
                leaf_nodes.append(leaf_node)

        df = pd.concat([df,
                        pd.DataFrame({
                            columns[0]:item,
                            columns[1]:[np.sum(leaf_nodes)],
                            columns[2]:[np.nan],
                            columns[3]:[np.nan],
                            columns[4]:[np.nan],
                            columns[5]:[np.nan],
                            columns[6]:[np.nan],
                            columns[7]:[np.nan],
                            columns[8]:[np.average(list(deg.values()))],
                            columns[9]:[np.max(list(deg.values()))],
                            columns[10]:[np.average([x for x in list(deg.values()) if x > 1])],
                            columns[11]:[np.max([x for x in list(deg.values()) if x > 1])],
                        })], ignore_index=True)

    print("Calculating average thickness (point based)")
    for item in sorted(properties.keys()):
        df.loc[df[columns[0]] == item, columns[2]] = np.average(properties[item]['radius_points'])
        df.loc[df[columns[0]] == item, columns[3]] = np.max(properties[item]['radius_points'])

    print("Calculating average thickness (line based)")
    for item in sorted(properties.keys()):
        df.loc[df[columns[0]] == item, columns[4]] = np.average(properties[item]['radius_lines'])
        df.loc[df[columns[0]] == item, columns[5]] = np.max(properties[item]['radius_lines'])

    print("Counting number of vertices")
    for item in sorted(properties.keys()):
        df.loc[df[columns[0]] == item, columns[6]] = np.max(properties[item]['vertices'])

    print("Calculating number of edges")
    for item in sorted(properties.keys()):
        df.loc[df[columns[0]] == item, columns[7]] = np.max(properties[item]['edges'])
        print(f"{item}\t{len(properties[item]['edges'])}")

    print("Saving...")
    df.to_csv(path_out + "/nerve_properties.csv")
