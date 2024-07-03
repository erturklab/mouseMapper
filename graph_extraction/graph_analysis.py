import os
import sys

import dill

import dask
import dask.array as da

import numpy as np
import nibabel as nib

from dask.diagnostics import ProgressBar
from dask.array.image import imread

from matplotlib import pyplot as plt
from pathlib import Path


# PI2 Import
sys.path.append("/home/rami/Documents/pi2/bin-linux64/release-nocl/")
from pi2py2 import *
pi = Pi2()
# pi.distribute(Distributor.LOCAL)

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

def check(array):
    if type(array) == "np.ndarray" or issubclass(np.ndarray, type(array)) or issubclass(da.core.Array, type(array)):
        return array
    else:
        return to_npy(array)

def from_npy(array):
    img = pi.newimage(array.dtype, array.shape[0], array.shape[1], array.shape[2])
    img.set_data(array)
    return img

def to_npy(img):
    return(img.get_data())
    
def show_mip(array, savepath=""):
    array = check(array)
    fig, axis = plt.subplots(ncols=3)
    for i in range(len(array.shape)):
        mip = np.amax(array, axis=i)
        axis[i].imshow(mip)
    if savepath == "":
        plt.show()
    else:
        plt.savefig(savepath)
    
def show_sum(array, savepath=""):
    array = check(array)
    fig, axis = plt.subplots(ncols=3)
    for i in range(len(array.shape)):
        mip = np.sum(array, axis=i)
        axis[i].imshow(mip)
    if savepath == "":
        plt.show()
    else:
        plt.savefig(savepath)

def skeletonize_measurements(path_in, path_out, output_name):
    """
    Skeletonize a binary volumetric mask, postprocess the graph and save a distance-map enhanced graph.
    Args:
        path_in (str): Path of the input volumetric image, folder of TIFF images
        path_out (str): Path of the output folder, will be created if not existing. Raw skeleton, measurements csv and vtk file is saved there
    """
    if not os.path.exists(path_out):
        os.mkdir(path_out)


    print("Reading image...")
    if ".nii.gz" in path_in:
        print("Reading nifti image...")
        binary = read_nifti(path_in)
    else:
        # Read input data
        print("Reading TIFF stack...")
        path_in = Path(path_in, "*.tif*")
        with ProgressBar():
            binary = imread(str(path_in))
    binary = binary.astype(np.uint8)
    img = from_npy(binary)

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

    # Graph pruning should go here
    pi.pruneskeleton(vertices, edges, measurements, points, 40, False, True)

    # Convert to vtk format in order to get radius for each point and line
    print("Generating vtk image...")
    vtkpoints = pi.newimage()
    vtklines = pi.newimage()
    pi.getpointsandlines(vertices, edges, measurements, points, vtkpoints, vtklines)
    
    # Get radius for each point
    points_data = to_npy(vtkpoints)
    radius_points = np.zeros([points_data.shape[0]])
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
    length_data = meas_data[:, 1]

    # Calculate degree of each vertex
    deg = {}
    for i in range(0, vert_coords.shape[0]):
        deg[i] = 0

    for i in range(0, old_edges.shape[0]):
        deg[old_edges[i, 0]] += 1
        deg[old_edges[i, 1]] += 1

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
            #if deg[n1] == 1 and deg[n2] == 1:
            #        L = length_data[i]
            #        if L < 5 / 0.75: # (5 um) / (0.75 um/pixel)
            #                should_remove = True

            remove_flags.append(should_remove)

    remove_flags = np.array(remove_flags).astype(np.uint8)
    print(f"Before dynamic pruning: {old_edges.shape[0]} edges")
    print(f"Removing {np.sum(remove_flags)} edges")

    # This call adjusts the vertices, edges, and measurements images such that
    # the edges for which remove_flags entry is True are removed from the graph.
    # Disable distributed processing for this - not yet implemented
    #pi.distribute(Distributor.LOCAL)
    #pi.distribute()

    #intermediate_path = "./skeletonization_steps/skel_intermediate.raw"
    intermediate_path = os.path.join(path_out, f"{output_name}_intermediate.raw")
    pi.writeraw(img, intermediate_path)

    #img = pi.newimage
    #pi.readraw(img, intermediate_path)
    pi.removeedges(vertices, edges, measurements, points, remove_flags, True, True)

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
    print("Saving vtk image...")
    pi.writevtk(vtkpoints, vtklines, os.path.join(path_out, f"{output_name}.vtk") , "radius", radius_points, "radius", radius_lines)


    # Generate and save figures
    print("Saving figures...")
    plt.hist(radius_points, bins="auto")
    plt.savefig(os.path.join(path_out, f"{output_name}_radius_points.png"))
    plt.hist(radius_lines, bins="auto")
    plt.savefig(os.path.join(path_out, f"{output_name}_radius_lines.png"))
    plt.hist(deg.values(),bins="auto")
    plt.savefig(os.path.join(path_out, f"{output_name}_degree.png"))

    # Save properties dict
    properties_dict = {
            "vertices":to_npy(vertices),
            "edges":to_npy(edges),
            "measurements":to_npy(measurements),
            "points":to_npy(points),
            "radius_points":radius_points,
            "radius_lines":radius_lines,
            "deg":deg.values()
            }
    with open(os.path.join(path_out, f"{output_name}_properties.pickledump"), "wb") as handle:
        dill.dump(properties_dict, handle)

# path_in = "/media/10TB/Projects/UCHL1 HFD/trigeminal_seg/segmentation_cutout/"
# path_out = "/media/10TB/Projects/UCHL1 HFD/trigeminal_seg/segmentation_cutout_graphs/"
path_in     = "/media/10TB/Projects/UCHL1 HFD/full body seg/original/image/"
path_out    = "/media/10TB/Projects/UCHL1 HFD/full body seg/original/graph/"
for item in os.listdir(path_in):
    # if len(os.listdir(path_in + item)) > 0:
    skeletonize_measurements(os.path.join(path_in, item), os.path.join(path_out, item), item)
