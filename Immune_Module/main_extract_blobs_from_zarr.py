import cut_volume_and_extract_blobs_zarr

parameters = {}

parameters["file_format"] = "Nifti"
parameters["region_name"] = "CD68" 


parameters["dataset"] = {}
parameters["dataset"]["channelname"] = ""
parameters['dataset']['sourcefolder'] ="OUT_PATH/binaries.zarr"
parameters['dataset']['localfolder'] = "THIS_IS_WHERE_THE_BLOB_PICKLES_WILL_BE_SAVED"
parameters['dataset']['syncfolder'] =  "NOT_USED_TODO_REMOVE"
parameters['dataset']['cachefolder'] = "NOT_USED_TODO_REMOVE"
parameters['dataset']['downsampling'] = 'None'

parameters['partitioning'] = {}
parameters['partitioning']['patch_size'] = [500, 500, 100]
parameters['partitioning']['patch_overlap'] = 0
parameters['partitioning']['cropping_offset'] = [0,0,0]#if the scan has black space
#first cut the first 3 z slices

parameters['partitioning']['cropping_boundingbox'] = [ 1000,1000,1000]#until where the mouse spans in the scan
parameters['partitioning']['safe_cache'] = False

parameters['multiprocessing'] = True

parameters['advanced'] = {}
parameters['advanced']['empty_patches'] = [] #outside_list

cut_volume_and_extract_blobs_zarr.cut_volume(parameters)

