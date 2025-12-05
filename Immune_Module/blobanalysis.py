import numpy as np
import scipy.ndimage
from cc3d import connected_components
from collections import defaultdict



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


#%%
def point_offset(old_pointlist,offsetpoint,mode):
    ''' 
    new_pointlist = point_offset(old_pointlist,offsetpoint,mode)
    
    Adds or subtracts offsetpoint from/to each point in old_pointlist. This can be
    used to switch between absolute and relative coordinates
    '''
    if(mode=='add'):
        new_pointlistarray = np.asarray(old_pointlist) + np.asarray(offsetpoint)
    elif(mode=='subtract'):
        new_pointlistarray = np.asarray(old_pointlist) - np.asarray(offsetpoint)
    return new_pointlistarray.tolist()

#%%
def delete_points(old_list,points):
    ''' 
    new_list = delete_points(old_list,points)
    
    Returns a list of all points in 'old_list' that are not part of the list 'points'.
    (The 'old_list' will not be modified by this function.)
    '''
    new_list = old_list.copy()
    for point in points:
        try: 
            new_list.remove(point)
        except:
            pass
    return new_list


#%%
def add_points(old_list,points):
    ''' 
    new_list = add_points(old_list,points)
    
    Returns a list of all points in 'old_list' plus those in 'points' that are not already in old_list
    (The 'old_list' will not be modified by this function.)
    '''
    new_list = old_list.copy()
    for point in points:
        if(point not in new_list): new_list.append(point)
    return new_list


#%%
def get_overlap(pointlist1,pointlist2):
    ''' 
    overlappingpoints = get_overlap(pointlist1,pointlist2)
    
    Returns a list of all points in 'pointlist2' that are overlapping with the list of
    points provided in 'pointlist1'.
    '''
    overlappingpoints = []
    for point in pointlist2:
        if(point in pointlist1): overlappingpoints.append(point)
    return overlappingpoints


#%%
def test_overlap(pointlist1,pointlist2):
    ''' 
    test_result = test_overlap(pointlist1,pointlist2)
    
    Checks whether any point in pointlist1 is also in pointlist2
    '''
    # First check trivial cases with fast methods
    pa1 = np.asarray(pointlist1)
    pa2 = np.asarray(pointlist2)
    ndims = pa1.shape[1]
    if(np.min(pa1[:,0]) > np.max(pa2[:,0])): return False
    if(np.min(pa2[:,0]) > np.max(pa1[:,0])): return False
    if(np.min(pa1[:,1]) > np.max(pa2[:,1]) and ndims >= 2): return False
    if(np.min(pa2[:,1]) > np.max(pa1[:,1]) and ndims >= 2): return False
    if(np.min(pa1[:,2]) > np.max(pa2[:,2]) and ndims >= 3): return False
    if(np.min(pa2[:,2]) > np.max(pa1[:,2]) and ndims >= 3): return False
    # Then check point by point overlap
    test_result = False
    maxind = len(pointlist1) - 1
    ind = 0
    while(test_result == False and ind <= maxind):
        if(pointlist1[ind] in pointlist2): test_result = True
        ind += 1
    return test_result


#%%
def get_neighbors_of_blob(pointlist1,pointlist2):
    ''' 
    touchingpoints = get_neighbors_of_blob(pointlist1,pointlist2)
    
    Returns a list of all points in 'pointlist2' that are direct neighbors to any of
    the points in the list 'pointlist1'. This includes points that are part of the blob
    
    ### Idea for speedup ###
    The current implementation calls get_overlap far more often than needed (at least for large 
    pointlists), as the 26 versions of currentpointlist are typically largely overlapping. However,
    the overhead involved in avoiding this may outweigh the benefits (1 implementation tested, but was not faster)
    '''
    if(len(pointlist2[0])!=3): raise ValueError('Please provide 3D inputs')
    touchingpoints = []
    pointlista = np.asarray(pointlist2)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                #shift points relative to pointlist1 & get overlapping points
                currentpointlist = (pointlista + [dy,dx,dz]).tolist()
                overlappingpoints = get_overlap(pointlist1,currentpointlist)
                # shift them back & add to list
                if(len(overlappingpoints)>0): overlappingpoints = (np.asarray(overlappingpoints) - [dy,dx,dz]).tolist()
                touchingpoints = add_points(touchingpoints,overlappingpoints)
    return touchingpoints


#%%
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
    
    labels = np.unique(labeled_volume)
    labels = labels[labels != 0]

    # single pass: gather indices per label
    coords = np.argwhere(labeled_volume > 0)
    blobs = defaultdict(list)
    for x, y, z in coords:
        blobs[labeled_volume[x, y, z]].append((int(x), int(y), int(z)))

    bloblist = []
    for i, label in enumerate(labels):
    
        blob = {}
        blob['id'] = len(bloblist)
        blob['points'] = blobs[label]
        blob = characterize_blob(blob)
        bloblist.append(blob)
        
    return bloblist

#%%
def get_single_blob(volume):
    ''' 
    blob = get_single_blob(volume)
    
    Returns blob-dict with all points in given volume. Only valid for
    volumes for which we already know to only contain a single blob.
    Result would be the same with get_blobs() but it's faster.
    '''
    allpoints = np.asarray(np.where(volume.astype(bool) == True)).T.tolist() # returns list of pointers for all non-zero values
    blob = {}
    blob['id'] = 0
    blob['points'] = allpoints
    blob = characterize_blob(blob)
    return blob


#%%
def get_blobs_2D(searcharea):
    '''
    TEST
    '''
    assert(len(searcharea.shape) == 2)
    connectedpixel = np.ones([3,3]) # Mask of pixels that count as connected to center pixel
    labeled,n_blobs = scipy.ndimage.label(searcharea,structure=connectedpixel) #very fast for large 2D, not for large 3D
    bloblist = []
    for b in range(0,n_blobs):
        blob = {}
        blob['points'] = (np.asarray(np.where(labeled==b+1)).T).tolist()
        blob = characterize_blob(blob,reduced=True) # only add minimal information
        bloblist.append(blob)
    return bloblist


#%%
def get_blobs_fast(volume):
    ''' 
    blobs = get_blobs_fast(volume)
    
    Much faster version of get_blobs() that solves 3D blob detection via 2D projections.
    Will be >100 times faster in busy 3D volumes but may potentially yield slightly wrong
    results. This will happen in cases where 2 neighboring blobs are so close and of such 
    a shape that their local 2D projections will always overlap (intertwined blobs)
    '''
    blobs = []
    volume = volume.astype(np.float64) # makes values binary (zero / non-zero)
    # Subdivide volume into subvolumes that only contain a single blob
    Z_blobs = get_blobs_2D(np.max(volume,2)) # YX
    for Z_blob in Z_blobs:
        # crop along Y and X for given Z_blob
        dy       = Z_blob['offset'][0] # implicit offset-subtraction in y direction
        dx       = Z_blob['offset'][1] # implicit offset-subtraction in x direction
        Y_length = Z_blob['boundingbox'][0]
        X_length = Z_blob['boundingbox'][1]
        subvolume = np.zeros([Y_length,X_length,volume.shape[2]]) # reduce in y and x dimension
        pointlist_yx     = np.asarray(Z_blob['points'])
        pointlist_yx_rel = np.asarray(Z_blob['points']) - [dy,dx]
        subvolume[pointlist_yx_rel[:,0],pointlist_yx_rel[:,1],:]               = volume[pointlist_yx[:,0],pointlist_yx[:,1],:]
        Y_blobs = get_blobs_2D(np.max(subvolume,0))
        for Y_blob in Y_blobs:
            # from Y perspective, now also crop Z
            dz       = Y_blob['offset'][1] # implicit offset-subtraction in z direction
            Z_length = Y_blob['boundingbox'][1]
            subsubvolume = np.zeros([subvolume.shape[0],subvolume.shape[1],Z_length]) # only reduce in z dimension
            pointlist_xz     = np.asarray(Y_blob['points']) + [dx,0] # correct x offset from previous cut
            pointlist_xz_rel = np.asarray(Y_blob['points']) - [0,dz]
            subsubvolume[:,pointlist_xz_rel[:,0],pointlist_xz_rel[:,1]]        = volume[dy:dy+Y_length,pointlist_xz[:,0],pointlist_xz[:,1]]
            X_blobs = get_blobs_2D(np.max(subsubvolume,1))
            for X_blob in X_blobs:
                # from X perspective, further crop Y and Z
                dy2       = X_blob['offset'][0] # further implicit offset-subtraction in y direction
                dz2       = X_blob['offset'][1] # further implicit offset-subtraction in z direction
                Y_length2 = X_blob['boundingbox'][0]
                Z_length2 = X_blob['boundingbox'][1]
                subsubsubvolume = np.zeros([Y_length2,subsubvolume.shape[1],Z_length2]) # only keep x dimension
                pointlist_yz     = np.asarray(X_blob['points']) + [dy,dz]   # correct y and z offset from previous cuts
                pointlist_yz_rel = np.asarray(X_blob['points']) - [dy2,dz2]
                subsubsubvolume[pointlist_yz_rel[:,0],:,pointlist_yz_rel[:,1]] = volume[pointlist_yz[:,0],dx:dx+X_length,pointlist_yz[:,1]]
                
                blob = get_single_blob(subsubsubvolume)
                blob['id'] = len(blobs)
                blob['points'] = (np.asarray(blob['points']) + [dy+dy2,dx,dz+dz2]).tolist()
                blob['offset'] = np.min(blob['points'],axis=0,keepdims=True).flatten() # update offset in patch coordinates
                blobs.append(blob)
    return blobs


#%%
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
        * blob['characterization']['compactness'] - Volume of blob divided by volume of enclosing sphere
        * blob['characterization']['sphereness'] - Ratio of max_dist to diameter of a sphere with same volume as blob
        * blob['characterization']['stringness'] - Defined as "1-sphereness"; approaches 1 for string-like shapes
        * blob['characterization']['skewness'] - Approaches 1 if blob is thick/dense on one end and has large tail on other side
    '''
    # Crop to relevant region
    if(len(blob['points'])==0):
        print('WARNING: Blob is empty')
        blob['volume'] = 0
        blob['CoM'] = None
        blob['MOP'] = None
        blob['max_dist'] = 0
        blob['characterization'] = {}
        blob['characterization']['compactness'] = None
        blob['characterization']['sphereness'] = None
        blob['characterization']['stringness'] = None
        blob['characterization']['skewness'] = None
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
        blob['MOP'] = relpointers[0]
        blob['max_dist'] = 0
        blob['characterization'] = {}
        blob['characterization']['compactness'] = None
        blob['characterization']['sphereness'] = None
        blob['characterization']['stringness'] = None
        blob['characterization']['skewness'] = None
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
    # Create subdict, if not existent
    if('characterization' not in blob.keys()):
        blob['characterization'] = {}
    # Compactness
    compactness = volume/(4/3*np.pi*(max_dist/2)**3) # volume of blob divided by volume of enclosing sphere
    compactness = np.clip(compactness,0,1) # clip to 0 and 1 for rounding errors (discrete vs. continuous geometry)
    blob['characterization']['compactness'] = compactness
    # Sphereness
    d_min = 2*(volume/(4/3*np.pi))**(1/3) # diameter of a sphere with same volume
    sphereness = d_min/max_dist # will be 1 if blob is a sphere,
    sphereness = np.clip(sphereness,0,1) # clip to 0 and 1 for rounding errors (discrete vs. continuous geometry)
    blob['characterization']['sphereness'] = sphereness
    # Stringness/elongation
    stringness = 1-d_min/max_dist
    stringness = np.clip(stringness,0,1) # clip to 0 and 1 for rounding errors (discrete vs. continuous geometry)
    blob['characterization']['stringness'] = stringness
    # Skewness
    skewness = 2*(dist_to_MOP/max_dist - 0.5) # approaches 1 if blob is thick/dense on one end and has large tail
    blob['characterization']['skewness'] = skewness
    
    return blob
