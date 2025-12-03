import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import datetime
import pathlib
from shutil import copyfile
import h5py
import numpy as np


def extract_vessel_graph(volume_path: str,
                         outdir: str,
                         tempdir: str,
                         cachedir:str,
                         bulge_size: float,
                         workspace_file: str,
                         voreen_tool_path: str,
                         name='',
                         generate_graph_file=False,
                         verbose=False):

    bulge_size_identifier = f'{bulge_size}'
    bulge_size_identifier = bulge_size_identifier.replace('.','_')

    bulge_path = f'<Property mapKey="minBulgeSize" name="minBulgeSize" value="{bulge_size}"/>'

    bulge_size_identifier = f'{bulge_size}'
    bulge_size_identifier = bulge_size_identifier.replace('.','_')
    edge_path = f'{outdir}{name}_edges.csv'
    node_path = f'{outdir}{name}_nodes.csv'
    graph_path = f'{outdir}{name}_graph.vvg'

    volume_name = volume_path.split("/")[-1].split(".")[0]
    # create temp directory
    temp_directory = os.path.join(tempdir,datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+name)
    print(f"Creating temp dir {temp_directory}",flush=False)
    pathlib.Path(temp_directory).mkdir(parents=True, exist_ok=True)
    voreen_workspace = 'vesselgraphextraction.vws'
    print(f"Copying voreen workspace from {workspace_file}",flush=False)
    copyfile(workspace_file,os.path.join(temp_directory,voreen_workspace))

    # Read in the file
    print(f"Reading voreen file {os.path.join(temp_directory, voreen_workspace)}",flush=False)
    with open(os.path.join(temp_directory,voreen_workspace), 'r') as file :
        filedata = file.read()

    out_path = f'{temp_directory}/sample.h5'

    # Replace the target string
    filedata = filedata.replace("volume.nii", volume_path)
    filedata = filedata.replace("nodes.csv", node_path)
    filedata = filedata.replace("edges.csv", edge_path)
    filedata = filedata.replace("graph.vvg", graph_path)
    filedata = filedata.replace('<Property mapKey="continousSave" name="continousSave" value="false" /> <Property mapKey="graphFilePath"',
                                f'<Property mapKey="continousSave" name="continousSave" value="{str(generate_graph_file).lower()}" /> <Property mapKey="graphFilePath"')
    filedata = filedata.replace('<Property mapKey="minBulgeSize" name="minBulgeSize" value="3" />', bulge_path)
    filedata = filedata.replace("input.nii", volume_path)
    filedata = filedata.replace("output.h5", out_path)

    print(f"Writing out the file out again",flush=False)
    # Write the file out again
    with open(os.path.join(temp_directory,voreen_workspace), 'w') as file:
        file.write(filedata)

    workspace_file = os.path.join(temp_directory,voreen_workspace)

    # absolute_temp_path = os.path.join(tempdir)

    workspace_file = workspace_file.replace(" ", "\\ ")
    outdir          = outdir.replace(" ", "\\ ")
    tempdir         = tempdir.replace(" ", "\\ ")
    cachedir        = cachedir.replace(" ", "\\ ")
    voreen_command = f'cd {voreen_tool_path} ; ./voreentool \
        --workspace {workspace_file} \
        -platform minimal --trigger-volumesaves --trigger-geometrysaves  --trigger-imagesaves \
        --workdir {outdir} --tempdir {tempdir} --cachedir {cachedir}' + ("" if verbose else "--logLevel error >/dev/null 2>&1")
    print(f"Voreen command \n{voreen_command}\n",flush=False)
    # extract graph and delete temp directory
    os.system(voreen_command)
    
    # if generate_graph_file:
    #     os.rename(graph_path, graph_path.replace(".vvg", ".vvg"))
    # with h5py.File(out_path, "r") as f:
    #     # Print all root level object names (aka keys) 
    #     # these can be group or dataset names 
    #     a_group_key = list(f.keys())[0]
    #     ds_arr = f[a_group_key][()]  # returns as a numpy array
    #     os.system(f"rm -rf '{absolute_temp_path}' 2> /dev/null")
    # ret = ds_arr[1]
    # ret = np.flip(np.rot90(ret),0)
    # return ret

