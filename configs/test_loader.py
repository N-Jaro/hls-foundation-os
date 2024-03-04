import os
custom_imports = dict(imports=["geospatial_fm"])
custom_imports = dict(imports=["map_seg"])

# base options
dist_params = dict(backend="nccl")
log_level = "INFO"

dataset_type = "MapSegDataset"
img_suffix = ".png"
seg_map_suffix = "_mask.png"

# TO BE DEFINED BY USER: data directory
data_root = "/projects/bbym/nathanj/hls-foundation-os/data/map_seg"

train_pipeline = [
    dict(type="LoadMapSegDataPatch"),
    dict(type="LoadMapSegAnnotations"),
]

CLASSES = ("non seg", "seg")

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="training",
        ann_dir="training",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,
        ignore_index=-1,
    ),
)

