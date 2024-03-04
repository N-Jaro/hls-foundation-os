from map_seg.customLoader import LoadImageWithMetadata 
# dataset_type = 'SomeDatasetType'  # A simple dataset type
# data_root = 'path/to/your/test/images'

# img_norm_cfg = dict(...)   # If you need normalization
train_pipeline = [
    dict(type='LoadImageWithMetadata'),  # Use your custom loader
    # ... other necessary transforms
]
