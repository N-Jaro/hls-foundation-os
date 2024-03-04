from map_seg.customLoader import LoadImageWithMetadata 

dataset_type = 'SomeDatasetType'  # A simple dataset type
data_root = '/projects/bbym/shared/all_patched_data/training/poly/map_patches/'

# img_norm_cfg = dict(...)   # If you need normalization
train_pipeline = [
    dict(type='LoadImageWithMetadata'),  # Use your custom loader
    # ... other necessary transforms
]
