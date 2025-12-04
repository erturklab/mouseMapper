import os
import numpy as np
import cv2
import nibabel as nib
from shutil import copy
from cc3d import connected_components
import pickle



print('Utils loaded without issues')


def readNifti(path,reorient=None):
    '''
    volume = readNifti(path)
    
    Reads in the NiftiObject saved under path and returns a Numpy volume.
    This function can also read in .img files (ANALYZE format).
    '''
    if(path.find('.nii')==-1 and path.find('.img')==-1):
        path = path + '.nii'
    print(path)
    if(os.path.isfile(path)):    
        NiftiObject = nib.load(path)
    elif(os.path.isfile(path + '.gz')):
        NiftiObject = nib.load(path + '.gz')
    else:
        raise Exception("No file found at: "+path)
    # Load volume and adjust orientation from (x,y,z) to (y,x,z)
    volume = np.swapaxes(NiftiObject.dataobj,0,1)
    if(reorient=='uCT_Rosenhain' and path.find('.img')):
        # Only perform this when reading in raw .img files
        # from the Rosenhain et al. (2018) dataset
        #    y = from back to belly
        #    x = from left to right
        #    z = from toe to head
        volume = np.swapaxes(volume,0,2) # swap y with z
        volume = np.flip(volume,0) # head  should by at y=0
        volume = np.flip(volume,2) # belly should by at x=0
    return volume

def get_blobs(volume):
    '''
    bloblist = get_blobs(volume)
    
    This function returns a list of dictionaries, in which each dictionary
    represents one blob in the given 'searchvolume'. A blob is defined as 
    a set of connected points. The 'searchvolume' is expected to be a 
    p-dimensional Numpy array of zero and non-zero values. All neighboring
    non-zero values will be treated as connected points, i.e. a blob.
    
    Each blob dictionary in the list 'blobs' has the following entries:
        * blob['id'] - Number of blob in searchvolume, starting with 0
        * blob['points'] - List of points in this blob. Each point is a 1D Numpy array with p coordinates (one per dimension)
        * blob['offset'] - Offset from bounding box to global coordinate system
        * blob['boundingbox'] - Size of 3D box enclosing the entire blob
        * blob['volume'] - Number of voxels in blob
        * blob['CoM'] - Center of Mass (within bounding box)
        * blob['max_dist'] - Largest distance between any two points of blob
        * blob['characterization'] - Dict of further characterizations
        
    NB: The runtime of this function is largely independent of size of the 
    searchvolume, but grows with the number as well as the size of blobs.
    For busy 3D volumes, get_blobs_fast() can >100 times faster (but might
    falsly merge two almost-overlapping points in rare cases)
    
    This version is using an external library for connected components (26-connectedness)
    that was not available at the beginning of Project Leo. Please see:
        https://github.com/seung-lab/connected-components-3d
    '''
    volume = volume.astype(bool)
    labeled_volume = connected_components(volume, connectivity=18)
    labels = [ x for x in np.unique(labeled_volume) if x != 0 ]
    bloblist = []
    for label in labels:
        allpoints = np.asarray(np.where(labeled_volume == label)).T.tolist() # returns list of pointers; slow for large vols
        blob = {}
        blob['id'] = len(bloblist)
        blob['points'] = allpoints
        blob = characterize_blob(blob)
        bloblist.append(blob)
    return bloblist

def characterize_blob(blob,reduced=False):
    ''' 
    blob = characterize_blob(blob,reduced=False)
    
    This takes a dictonary 'blob' as an input, calculates various metrics
    to characterize the blob, and adds these metrics to the dictionary before
    returning it.
    
    For the input dictionary, only the field "points" must be given. It 
    should be a list of points in 3D space representing the blob. The points 
    must be given in absolute coordinates
    
    The returned dictionary will comprise the following metrics:
        * blob['offset'] - Offset from bounding box to global coordinate system
        * blob['boundingbox'] - Size of 3D box enclosing the entire blob
        * blob['volume'] - Number of voxels in blob
        * blob['CoM'] - Center of Mass (within bounding box)
        * blob['max_dist'] - Largest distance between any two points of blob
        * blob['characterization']['stringness'] - Defined as "1-sphereness"; approaches 1 for string-like shapes
    '''
    # Crop to relevant region
    if(len(blob['points'])==0):
        print('WARNING: Blob is empty')
        blob['volume'] = 0
        blob['CoM'] = None
        blob['stringness'] = None
        return blob
    boundmin = np.min(blob['points'],axis=0,keepdims=True)
    boundmax = np.max(blob['points'],axis=0,keepdims=True)
    boundingbox = (boundmax-boundmin+1).flatten()
    relpointers = (blob['points'] - boundmin)
    blob['offset'] = boundmin.flatten()
    blob['boundingbox'] = boundingbox
    if(len(blob['points'][0])<3):
        #print('2D blobs are only partially characterized in current implementation.')
        return blob
    if(reduced):
        blob['volume'] = len(blob['points'][1])
        return blob 
    canvas = np.zeros(boundingbox,bool)
    canvas[relpointers[:,0],relpointers[:,1],relpointers[:,2]] = 1
    # Volume
    volume = np.sum(canvas)
    blob['volume'] = volume
    if(volume==1):
        blob['CoM'] = relpointers[0]
        blob['max_dist'] = 0
        blob['stringness'] = None
        return blob
    # Center of Mass
    CoM = np.uint32(np.round(np.mean(relpointers,axis=0,keepdims=True)).flatten())
    blob['CoM'] = CoM
    # Maximum distance between any two points of blob
    dist_to_MOP = 0
    for point in relpointers:
        dist = point_dist(CoM,point)
        if(dist>dist_to_MOP):
            dist_to_MOP = dist
            MOP = point
    max_dist = 0
    for point in relpointers:
        dist = point_dist(MOP,point)
        if(dist>max_dist):
            max_dist = dist
    blob['max_dist'] = max_dist
    # Stringness/elongation
    d_min = 2*(volume/(4/3*np.pi))**(1/3) # diameter of a sphere with same volume
    stringness = 1-d_min/max_dist
    stringness = np.clip(stringness,0,1) # clip to 0 and 1 for rounding errors (discrete vs. continuous geometry)
    blob['stringness'] = stringness*100
    
    return blob


def find_blobs_slow_in_folder(folder_patches,path_to_save_predictions):

    all_patches = sorted(os.listdir(folder_patches))
    all_patches = [patch for patch in all_patches if patch.find('.nii.gz')!=-1]
    for patch in all_patches:
        allblobs = []
        patch_name = patch.replace('.nii.gz','')
        volume = readNifti(folder_patches+patch_name)
        
        
        blobs = get_blobs(volume)

        for blob in blobs:
            blob['abs_loc']=blob['offset']
            allblobs.append(blob)

        print(len(allblobs))
        psave(path_to_save_predictions+'prediction_'+patch, allblobs)
        allblobs = []
        

#%%
def point_dist(p1,p2):
    ''' 
    dist = point_dist(p1,p2)
    
    Returns the distance (scalar) between two points, defined as their vector norm. 
    Points p1 and p2 need to be in p-dimensional vector format, i.e. a 1-D array 
    containing the coordinates of the corresponding pixel/voxel in the respective 
    p-dimensional space (e.g., an image or a volume)
    '''
    ndim = len(p1)
    distsum = 0
    for dim in range(0,ndim):
        distsum = distsum + abs(p1[dim]-p2[dim])**2
    dist = distsum**(1/2)
    return dist

def psave(path, variable):
    '''
    psave(path, variable)
    
    Takes a variable (given as string with its name) and saves it to a file as specified in the path.
    The path must at least contain the filename (no file ending needed), and can also include a 
    relative or an absolute folderpath, if the file is not to be saved to the current working directory.

    '''
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    #path = path.replace('\\','/')
    #cwd = os.getcwd().replace('\\','/')
    #if(path[0:2] != cwd[0:2] and path[0:5] != '/mnt/'):
    #    path = os.path.abspath(cwd + '/' + path) # If relatice path was given, turn into absolute path
    folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    if(os.path.isdir(folderpath) == False):
        os.makedirs(folderpath) # create folder(s) if missing so far.
    file = open(path, 'wb')
    pickle.dump(variable,file,protocol=4)
    print('Variable saved to: '+path)

def pload(path):
    '''
    variable = pload(path)
    
    Loads a variable from a file that was specified in the path. The path must at least contain the 
    filename (no file ending needed), and can also include a relative or an absolute folderpath, if 
    the file is not to located in the current working directory.
    
    The function returns the variable that was saved in the file.'''
    file = open(path, 'rb')
    return pickle.load(file)
