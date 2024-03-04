import os
custom_imports = dict(imports=["geospatial_fm"])
custom_imports = dict(imports=["map_seg"])

# base options
dist_params = dict(backend="nccl")
log_level = "INFO"

dataset_type = "MapSegDataset"

# TO BE DEFINED BY USER: data directory
data_root = "/projects/bbym/nathanj/hls-foundation-os/data/map_seg"
train_pipeline = [
    dict(type="LoadMapSegDataPatch"),
    dict(type="LoadMapSegAnnotations"),
]