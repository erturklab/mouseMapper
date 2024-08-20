
import os
import numpy as np
import cv2
import nibabel as nib
from shutil import copy



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

def writeNifti(path,volume,compress=False):
    '''
    writeNifti(path,volume)
    
    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path. 
    '''
    if(path.find('.nii')==-1 and compress==False):
        path = path + '.nii'
    if(path.find('.nii.gz')==-1 and compress==True):
        path = path + '.nii.gz'
    folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    if(os.path.isdir(folderpath) == False):
        os.makedirs(folderpath) # create folder(s) if missing so far.
    # Save volume with adjusted orientation
    # --> Swap X and Y axis to go from (y,x,z) to (x,y,z)
    # --> Show in RAI orientation (x: right-to-left, y: anterior-to-posterior, z: inferior-to-superior)
    affmat = np.eye(4)
    affmat[0,0] = affmat[1,1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume,0,1), affine=affmat)
    nib.save(NiftiObject,path)

def downsample_folder(path_base , mousenames, channelnames , downsampling_factor_xy, downsampling_factor_z):

    """
    Downsamples TIFF files for the selected mice and channels
    Args:
        path_base (str): The base path of the folder containing the mice data.
        mousenames (list): A list of mouse names.
        channelnames (list): A list of channel names.
        downsampling_factor_xy (int): The downsampling factor for the x and y dimensions.
        downsampling_factor_z (int): The downsampling factor for the z dimension.
    Returns:
        None
    Raises:
        AssertionError: If the downsampling factors are not integers.
    Notes:
        - This function iterates over all mice and downsamples TIFF files for the given channel.
        - The downsampling is performed by resizing the images using OpenCV.
        - The downsampled files are saved in a target folder.
        - Any unwanted files in the target folder are deleted.

    """
        

    #%% Iterate over all mice and downsample TIFF files for given channel
    assert(downsampling_factor_xy == int(downsampling_factor_xy))
    assert(downsampling_factor_z == int(downsampling_factor_z))

    for mousename in mousenames:
        print('Downsampling data for mouse:', mousename, 'by factors: ',downsampling_factor_xy,downsampling_factor_z )
        for channelname in channelnames:
            print(mousename+" ("+channelname+") ----------")
            sourcefolder = path_base + mousename +'/' + channelname  + '/'
            targetfolder = path_base + mousename + '/DownsampledScan/xy'+str(downsampling_factor_xy)+'z'+str(downsampling_factor_z)+'/' + channelname  + '/'
        
            # create folder, if not already present
            if(os.path.isdir(targetfolder) is False):
                os.makedirs(targetfolder)
                
            # iterate over files in source folder
            desired_fnames = []
            i = 0
            fnames = sorted(os.listdir(sourcefolder))
            for fname in fnames[::downsampling_factor_z]:
                desired_fnames.append(fname)
                if(os.path.isfile(targetfolder + fname) is False):
                    # downsample & copy file, if not already present
                    i = i+1
                    try:
                        image = cv2.imread(sourcefolder + '/' + fname, 2)# '2' forces cv2 to keep original bitdepth
                        img_dim = (image.shape[1]//downsampling_factor_xy,image.shape[0]//downsampling_factor_xy)
                        image_ds = cv2.resize(image,img_dim, interpolation = cv2.INTER_NEAREST)#.astype(np.uint16
                        cv2.imwrite(targetfolder + fname,image_ds)
                        print('Downsampled ' + fname + '('+str(i)+' of '+str(int(len(fnames)/downsampling_factor_z+1)) + ')')
                    except:
                        print(' [!] Error with ' + fname)
                
            # remove any files that may have been in target folder but are unwanted
            fnames = sorted(os.listdir(targetfolder))
            for fname in fnames:
                if(fname not in desired_fnames):
                    os.remove(targetfolder + fname)
                    print('Deleted unwanted file: ' + fname)
                    
            if(i==0):
                print('All files were already downsampled.')
    print('Done.')

def preprocess_zslices_to_nifti(path_base , mousenames, channelnames , downsampling_factor_xy, downsampling_factor_z,path_ouput_preprocessing):
    
    """
    Preprocesses z-slices of images and saves them as NIfTI files.
    Args:
        path_base (str): The base path where the images are located.
        mousenames (list): A list of mouse names.
        channelnames (list): A list of channel names.
        downsampling_factor_xy (int): The downsampling factor for the x and y dimensions.
        downsampling_factor_z (int): The downsampling factor for the z dimension.
        path_ouput_preprocessing (str): The path to save the preprocessed NIfTI files.
    Returns:
        None
    """

    for mousename in mousenames:
        for channelname in channelnames:

            folder_load = path_base + mousename + '/DownsampledScan/xy'+str(downsampling_factor_xy)+'z'+str(downsampling_factor_z)+'/' + channelname  + '/'
            z_slices_sorted = sorted(os.listdir(folder_load))
            
            image = cv2.imread(folder_load + z_slices_sorted[0], 2) #load one tiff
            bb_y, bb_x = image.shape
            bb_z = len(z_slices_sorted)
            #C01 is autofluorescence, used as _0000, and C02 is PI, _0001
            fileindex = '_0000'
            if 'C02' in channelname:
                fileindex = '_0001'
            outvol_name = path_ouput_preprocessing+mousename + '_xy' + str(downsampling_factor_xy)+ '_z'+str(downsampling_factor_z)+fileindex+'.nii.gz'
            canvas = np.zeros((bb_y,bb_x,bb_z),np.uint16)
            for z,z_slice_name in enumerate(z_slices_sorted):
                image = cv2.imread(folder_load + z_slice_name, 2)
                canvas[:,:,z] = image
            writeNifti(outvol_name, canvas, compress=True)
            print( ' saved Nifti for', outvol_name)

def postprocess_predictions(folder_in_pred, folder_pred_postprocessed=None):
    """
    Postprocesses the predictions by modifying the predicted volume. As the model was initially trained with 22 classes, we need to merge the gut components.
    Args:
        folder_in_pred (str): The path to the folder containing the predicted volumes.
        folder_pred_postprocessed (str, optional): The path to the folder where the postprocessed predictions will be saved. 
            If not provided, the postprocessed predictions will be saved in the same folder as the input predictions.
    Returns:
        None
    """
    if folder_pred_postprocessed is None:
        folder_pred_postprocessed = folder_in_pred
    for sample in os.listdir(folder_in_pred):
        pred_vol = readNifti(folder_in_pred+'/'+sample)
        pred_vol[pred_vol==12]=11
        pred_vol[pred_vol==13]=11

        for i in range(13,23,1):
            pred_vol[pred_vol==i]=i-2
        
        writeNifti(folder_pred_postprocessed+'/'+sample,pred_vol )

def upsample_prediction_mask(folder_pred_postprocessed, folder_out, folder_fullres_in,downsampling_factor_z):
    """
    Upsamples the prediction masks for each mouse in the given folder.
    Args:
        folder_pred_postprocessed (str): Path to the folder containing the post-processed prediction masks.
        folder_out (str): Path to the output folder where the upsampled masks will be saved.
        folder_fullres_in (str): Path to the folder containing the full-resolution slices.
        downsampling_factor_z (int): Downsampling factor for the z-axis.
    Returns:
        None
    """

    for mouse in os.listdir(folder_pred_postprocessed):
        if '.nii.gz' in mouse:
            print('Upsampling masks for mouse: ', mouse)
            path_volume_in=folder_pred_postprocessed+'/'+mouse

            scan_name = mouse.replace('.nii.gz','')
            scan_name=scan_name[:scan_name.find('_xy')]

            slices_fullres=folder_fullres_in+scan_name+'/C01/'

            path_slices_out_small= folder_out+scan_name+'/slicesds/'
            path_slices_out_fullres=folder_out+scan_name+'/slicesus/'
            final_path = folder_out+scan_name+'/slices_fullres/'

            #TODO assert that the necessary folders exist

            if not os.path.exists(path_slices_out_small):
                os.makedirs(path_slices_out_small)
            if not os.path.exists(path_slices_out_fullres):
                os.makedirs(path_slices_out_fullres)
            if not os.path.exists(final_path):
                os.makedirs(final_path)

            #1. Slice up the volume

            vol=readNifti(path_volume_in)

            print('Creating zslices')
            for myslice in range(vol.shape[2]):
                cv2.imwrite(path_slices_out_small+'label_Z'+str(downsampling_factor_z*myslice).zfill(4)+'.tif',(vol[:,:,myslice]).astype(np.uint8))

            #upsample the samples:
            print('Upsampling the zslices')
            one_orig_file = os.listdir(slices_fullres)[0]
            image = cv2.imread(slices_fullres + '/' + one_orig_file, 2)
            img_dim = (image.shape[1],image.shape[0])

            for sample in os.listdir(path_slices_out_small):
                
                gt_slice = cv2.imread(path_slices_out_small + sample, 2)
            
                image_ds = cv2.resize(np.zeros(gt_slice.shape),img_dim, interpolation = cv2.INTER_NEAREST).astype(np.uint8)
        
                if (np.sum(gt_slice)>0):
                    uniques = np.unique(gt_slice)
                    for uniqueval in uniques:
                        if (uniqueval):
                            slice_here = (gt_slice==uniqueval).astype(np.uint8)
                            slice_up = cv2.resize(slice_here,img_dim, interpolation = cv2.INTER_LINEAR)
                            slice_up[image_ds>0]=0
                            image_ds+=uniqueval*(slice_up>0).astype(np.uint8)

                cv2.imwrite(path_slices_out_fullres + sample,image_ds)

                
            print('Filling in missing zslices')
            #copy z-slices forward and back 
            for index in range(len(os.listdir(slices_fullres))):
                slice_number = index
                if(os.path.isfile(final_path+'label_Z'+str(slice_number).zfill(4)+'.tif') is False):
                    if(slice_number%downsampling_factor_z)<=downsampling_factor_z//2:
                        copy(path_slices_out_fullres+'label_Z'+str((slice_number//downsampling_factor_z)*downsampling_factor_z).zfill(4)+'.tif',final_path+'label_Z'+str(slice_number).zfill(4)+'.tif')
                    else:
                        try:
                            copy(path_slices_out_fullres+'label_Z'+str(((slice_number//downsampling_factor_z) + 1)*downsampling_factor_z).zfill(4)+'.tif',final_path+'label_Z'+str(slice_number).zfill(4)+'.tif')
                        except:
                            copy(path_slices_out_fullres+sorted(os.listdir(path_slices_out_fullres))[-1],final_path+'label_Z'+str(slice_number).zfill(4)+'.tif')



def mask_out_organs_from_scan(folder_masks, folder_base_mice, folder_masked_out, channels = ['C01','C02']):
    """
    Masks out organs from a scan by applying a segmentation mask. This results in improved tissue segmentation downstream.
    Args:
        folder_masks (str): Path to the folder containing the segmentation masks.
        folder_base_mice (str): Path to the folder containing the mice scans.
        folder_masked_out (str): Path to the folder where the masked out scans will be saved.
        channels (list, optional): List of channels to be processed. Defaults to ['C01','C02'].
    Returns:
        None
    """

    mice = [mouse for mouse in os.listdir(folder_masks)]

    for mouse in mice:

        print('Masking out organs for mouse: ', mouse)
        folder_samples = folder_base_mice+mouse
        folder_segm = folder_masks +mouse+'/slices_fullres/'
        folder_out = folder_masked_out+mouse+'/'
        all_labels = sorted(os.listdir(folder_segm))

        path_save_nonorgans = folder_out


        if not os.path.exists(path_save_nonorgans):
            os.makedirs(path_save_nonorgans)
        for channel in channels:
            if not os.path.exists(path_save_nonorgans+channel):
                os.makedirs(path_save_nonorgans+channel)

        for index,sample in enumerate((all_labels)):
            label_file = cv2.imread(folder_segm+sample, 2)
            
            
            for channel in channels:
                path_channel_of_interest = folder_samples+'/'+channel
                file_name = sorted(os.listdir(path_channel_of_interest))[index]
                if not os.path.isfile(path_save_nonorgans+channel+'/'+file_name):
                    if (np.sum(label_file)):
                        image_of_interest = cv2.imread(path_channel_of_interest+'/'+file_name,2)
                        label_inverted = np.ones(label_file.shape)-(label_file>0).astype(np.uint8)
                        image_filtered = label_inverted*image_of_interest
                        cv2.imwrite(path_save_nonorgans+channel+'/'+file_name,image_filtered.squeeze().astype(np.uint16))
                    else:
                        copy(path_channel_of_interest+'/'+file_name,path_save_nonorgans+channel+'/'+file_name)
