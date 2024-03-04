import os
custom_imports = dict(imports=["geospatial_fm", "map_seg"])

palette = [[0,0,0], [128,128,128]]

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
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(128, 128), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # Consider adding more augmentations if needed
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

CLASSES = ('background', 'foreground')

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
    val=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="validation",
        ann_dir="validation",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,
        ignore_index=-1,
    ),
)

model = dict(
    type='EncoderDecoder',
    pretrained=None,  # No pre-trained model for a truly minimal setup
    backbone=dict(
        type='SwinTransformer',  # A modern backbone; you can use others 
        embed_dims=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        ),
    decode_head=dict(
        type='UnetHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=512,
        num_classes=2,  # Adjust to your number of classes
    )
)
