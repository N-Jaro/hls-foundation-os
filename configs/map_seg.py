import os

custom_imports = dict(imports=["geospatial_fm","map_seg"])

# base options
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
cudnn_benchmark = True

dataset_type = "MapSegDataset"

# TO BE DEFINED BY USER: data directory
data_root = "/projects/bbym/nathanj/hls-foundation-os/data/map_seg/"

num_frames = 1
img_size = 256
num_workers = 1
samples_per_gpu = 1

bands = [0, 1, 2, 3, 4, 5]
tile_size = 256
orig_nsize = 1
crop_size = (tile_size, tile_size)
# img_suffix = "_img.png"
# seg_map_suffix = "_mask.png"
ignore_index = -1
image_nodata = -9999
image_nodata_replace = 0
image_to_float32 = True

# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = "/projects/bbym/nathanj/hls-foundation-os/Prithvi-100M-burn-scar/burn_scars_Prithvi_100M.pth"
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1
output_embed_dim = num_frames * embed_dim
max_intervals = 10000
evaluation_interval = 1000

# TO BE DEFINED BY USER: model path
experiment = "/projects/bbym/nathanj/hls-foundation-os/"
project_dir = "map_seg"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir

save_path = work_dir
train_pipeline = [
    dict(type="LoadMapSegDataPatch", to_float32=image_to_float32),
    dict(type="LoadMapSegAnnotations", reduce_zero_label=False),
    # dict(type="BandsExtract", bands=bands),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    # to channels first
    # dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    # dict(type="TorchNormalize", **img_norm_cfg),
    # dict(type="TorchRandomCrop", crop_size=(tile_size, tile_size)),
    # dict(
    #     type="Reshape",
    #     keys=["img"],
    #     new_shape=(len(bands), num_frames, tile_size, tile_size),
    # ),
    # dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),
    # dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadMapSegDataPatch", to_float32=image_to_float32),
    # dict(type="BandsExtract", bands=bands),
    dict(type="ToTensor", keys=["img"]),
    # to channels first
    # dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    # dict(type="TorchNormalize", **img_norm_cfg),
    # dict(
    #     type="Reshape",
    #     keys=["img"],
    #     new_shape=(len(bands), num_frames, -1, -1),
    #     look_up=dict({"2": 1, "3": 2}),
    # ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ],
    ),
]

CLASSES = ("non-seg", "seg")

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=num_workers,
    train=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_suffix='', 
        seg_map_suffix='',
        img_dir= data_root +'images/training',
        ann_dir= data_root +'annotations/training',
        pipeline=train_pipeline,
        ignore_index=-1,
    ),
    val=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_suffix='', 
        seg_map_suffix='',
        img_dir= data_root +'images/validation',
        ann_dir= data_root +'annotations/validation',
        pipeline=test_pipeline,
        ignore_index=-1,
    ),
    test=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_suffix='', 
        seg_map_suffix='',
        img_dir= data_root +'images/validation',
        ann_dir= data_root +'annotations/validation',
        pipeline=test_pipeline,
        ignore_index=-1,
    ),
)

optimizer = dict(type="Adam", lr=1.3e-05, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook", by_epoch=False),
    ],
)
checkpoint_config = dict(by_epoch=True, interval=10, out_dir=save_path)
evaluation = dict(
    interval=evaluation_interval,
    metric="mIoU",
    pre_eval=True,
    save_best="mIoU",
    by_epoch=False,
)

loss_func = dict(type="DiceLoss", use_sigmoid=False, loss_weight=1, ignore_index=-1)

runner = dict(type="IterBasedRunner", max_iters=max_intervals)
workflow = [("train", 1)]
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="TemporalEncoderDecoder",
    frozen_backbone=False,
    backbone=dict(
        type="TemporalViTEncoder",
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=False,
    ),
    neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
        embed_dim=embed_dim * num_frames,
        output_embed_dim=output_embed_dim,
        drop_cls_token=True,
        Hp=14,
        Wp=14,
    ),
    decode_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func,
    ),
    auxiliary_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),
)
auto_resume = False
