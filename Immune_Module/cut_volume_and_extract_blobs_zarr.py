
import os
import sys
import numpy as np
import zarr
from multiprocessing import Pool, current_process
from tqdm import tqdm

import pickle
import blobanalysis

def psave(path, variable):
    '''
    psave(path, variable)
    
    Takes a variable (given as string with its name) and saves it to a file as specified in the path.
    The path must at least contain the filename (no file ending needed), and can also include a 
    relative or an absolute folderpath, if the file is not to be saved to the current working directory.
    
    # ToDo: save several variables (e.g. take X args, store them to special DICT, and save to file)
    '''
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    #path = path.replace('\\','/')
    #cwd = os.getcwd().replace('\\','/')
    #if(path[0:2] != cwd[0:2] and path[0:5] != '/mnt/'):
    #    path = os.path.abspath(cwd + '/' + path) # If relatice path was given, turn into absolute path
    #folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    #if(os.path.isdir(folderpath) == False):
    #    os.makedirs(folderpath) # create folder(s) if missing so far.
    file = open(path, 'wb')
    pickle.dump(variable,file,protocol=4)


def find_blobs_slow_in_volume(volume: np.ndarray, save_dir: str, region: dict, pid: int):
    """
    Extract blobs from a 3D patch and save them with absolute coordinates.
    """
    #os.makedirs(save_dir, exist_ok=True)
    all_blobs = []
    base_offset = region["patches"][int(pid)]["offset"]

    blobs = blobanalysis.get_blobs(volume)
    for blob in blobs:
        blob['patch_id']    = pid
        blob['abs_loc']     = blob['offset'] + base_offset + blob['CoM']
        blob['abs_points']  = [pt + base_offset for pt in blob['points']]
        all_blobs.append(blob)
    print('Worker', current_process().name, 'processed patch', pid, 'with', len(all_blobs), 'blobs.')
    psave(os.path.join(save_dir, f'prediction_{pid}'), all_blobs)


def extract_patch(zarr_path: str, patch: dict) -> np.ndarray:
    """
    Lazily read a 3D patch directly from the zarr volume on disk.
    Assumes zarr is stored as (Z, Y, X).
    """
    # each call opens only the required chunks
    zvol = zarr.open(zarr_path, mode='r')
    off = patch['offset']    # [y, x, z]
    bb  = patch['boundingbox']
    # slice order: z, y, x

    #check if the patch is within bounds
    
    patch_vol = zvol[ :,:,off[2]:off[2]+bb[2],
                      off[0]:off[0]+bb[0],
                      off[1]:off[1]+bb[1] ]
    return np.asarray(patch_vol).reshape(bb[2], bb[0], bb[1])  # ensure correct shape (Z, Y, X)


def process_patch(args):
    """Wrapper for multiprocessing or serial execution."""
    region, patch, zarr_path, out_dir = args
    pid = patch['id']
    patch_vol = extract_patch(zarr_path, patch)
    patch_vol=patch_vol.transpose(2, 1, 0)  # convert to (Y, X, Z) for blob analysis
    save_dir = out_dir#os.path.join(out_dir, f'patch_{pid}')
    print(f"Worker {current_process().name} processing patch {pid}")
    find_blobs_slow_in_volume(patch_vol, save_dir, region, pid)


def cut_volume(parameters: dict):
    """Cut a Zarr volume into patches and extract blobs from each."""
    # paths from parameters
    zarr_path = parameters['dataset']['sourcefolder']
    out_dir   = parameters['dataset']['localfolder']
    os.makedirs(out_dir, exist_ok=True)

    # peek at metadata
    vol = zarr.open(zarr_path, mode='r')
    print(f"Zarr volume shape: {vol.shape}")

    # build region with patch grid
    region = {'patches': []}
    ps   = np.array(parameters['partitioning']['patch_size'], dtype=int)
    ov   = parameters['partitioning']['patch_overlap']
    off0 = np.array(parameters['partitioning']['cropping_offset'], dtype=int)
    bb0  = np.array(parameters['partitioning']['cropping_boundingbox'], dtype=int)
    steps = np.floor((bb0 - off0 - ov) / (ps - ov))#.astype(int)
    print('Steps:',steps)
    steps = steps.astype(int)
    pid = 0
    for y in range(steps[0]):
        for x in range(steps[1]):
            for z in range(steps[2]):
                offset = off0 + (ps - ov) * np.array([y, x, z])
                region['patches'].append({
                    'id': pid,
                    'offset': offset,
                    'boundingbox': ps,
                })
                pid += 1

    # prepare tasks
    empty_ids = set(parameters.get('advanced', {}).get('empty_patches', []))
    tasks = []
    for patch in region['patches']:
        if str(patch['id']) in empty_ids:
            continue
        if not os.path.isfile(os.path.join(out_dir, f'prediction_{patch["id"]}.pickledump')):
            # only process patches that are not already saved
            # and not in the empty patches list
            tasks.append((region, patch, zarr_path, out_dir))

    # execute
    if parameters.get('multiprocessing', True):
        workers = parameters.get('num_workers', os.cpu_count() - 1)
        with Pool(processes=workers) as pool:
            for _ in tqdm(pool.imap_unordered(process_patch, tasks), total=len(tasks)):
                pass
    else:
        for args in tqdm(tasks):
            process_patch(args)

    psave(os.path.join(out_dir, 'region_metadata'), region)


# Module entry-point: users should import and call cut_volume(parameters) directly.
# Example:
# import cut_volume_and_extract_blobs_zarr
# cut_volume_and_extract_blobs_zarr.cut_volume(parameters)
