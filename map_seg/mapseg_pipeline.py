
import os
from PIL import Image
import numpy as np
import mmcv
import rioxarray
import torchvision as TV
import torchvision.transforms.functional as F
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import PIPELINES
from torchvision import transforms


# def open_tiff(fname):
#     data = rioxarray.open_rasterio(fname)
#     return data.to_numpy()

@PIPELINES.register_module()
class LoadMapSegDataPatch(object):

    def __init__(self, to_float32=True, nodata=None, nodata_replace=0.0, resize_to=(224, 224)):
        self.to_float32 = to_float32
        self.nodata = nodata
        self.nodata_replace = nodata_replace
        self.resize_to = resize_to

    def __call__(self, results):

        # print('filename:', results["img_info"]["filename"])

        if results.get("img_prefix") is not None:
            filename = os.path.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]

        # Patch and legend logic
        patch_name = os.path.basename(filename)
        base_name = patch_name.split('_poly')[0]
        base_folder = 'training' if 'training' in filename else 'validation'
        data_dir = '/projects/bbym/shared/all_patched_data/'
        legend_path = os.path.join(data_dir, base_folder, 'poly', 'legend', base_name + '_poly.png')

        # Load image patch
        img = np.array(Image.open(filename)) 
        # img = img.resize(self.resize_to, Image.ANTIALIAS)
        img = mmcv.imresize(img, self.resize_to, return_scale=False)
        # Load legend
        legend = np.array(Image.open(legend_path))
        # legend = legend.resize(self.resize_to, Image.ANTIALIAS)
        legend = mmcv.imresize(legend, self.resize_to, return_scale=False)

        img = np.concatenate((img,legend), axis=-1) 

        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        results["flip"] = False
        results["flip_direction"] = 'vertical'
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}"
        return repr_str


@PIPELINES.register_module()
class LoadMapSegAnnotations(object):

    def __init__(
        self,
        reduce_zero_label=False,
        nodata=None,
        nodata_replace=-1,
        resize_to=(224, 224)
    ):
        self.reduce_zero_label = reduce_zero_label
        self.nodata = nodata
        self.nodata_replace = nodata_replace
        self.resize_to = resize_to

    def __call__(self, results):

        if results.get("seg_prefix", None) is not None:
            filename = os.path.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        else:
            filename = results["ann_info"]["seg_map"]

        gt_semantic_seg = np.array(Image.open(filename)) 
        # gt_semantic_seg = gt_semantic_seg.resize(self.resize_to, Image.ANTIALIAS)
        gt_semantic_seg = mmcv.imresize(gt_semantic_seg, self.resize_to, return_scale=False)

        if gt_semantic_seg.shape[-1] == 1:  # Check if the last dimension is 1
            gt_semantic_seg = np.squeeze(gt_semantic_seg, axis=-1)  # Squeeze to remove the extra dimension

        # print("gt_semantic_seg.shape:",gt_semantic_seg.shape)

        if results.get("label_map", None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results["label_map"].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id

        results["gt_semantic_seg"] = gt_semantic_seg
        results["seg_fields"].append("gt_semantic_seg")
        return results