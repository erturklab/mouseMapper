def organ_segmentation(config):
    pass


##### Preprocess
# 1. Get path for AF and PI full-res channels
# 2. Downsample
# 3. Make niftis with the correct naming
##### 4. Run nnUNET
# 5. Post-processs (unify labels, reassign)
# 6. Upsample organ masks
# 7. Create masked organs, if wanted
# 8. Create non-masked organs, if wanted