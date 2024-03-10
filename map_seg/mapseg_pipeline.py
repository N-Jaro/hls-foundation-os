
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

    def __init__(self, to_float32=False, nodata=None, nodata_replace=0.0):
        self.to_float32 = True
        self.nodata = nodata
        self.nodata_replace = nodata_replace

    def __call__(self, results):

        print('filename:', results["img_info"]["filename"])

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

        # Load legend
        legend = np.array(Image.open(legend_path))

        img = np.concatenate((img,legend), axis=-1) 

        if self.to_float32:
            img = img.astype(np.float32) / 255.0

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
    ):
        self.reduce_zero_label = reduce_zero_label
        self.nodata = nodata
        self.nodata_replace = nodata_replace

    def __call__(self, results):

        if results.get("seg_prefix", None) is not None:
            filename = os.path.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        else:
            filename = results["ann_info"]["seg_map"]

        gt_semantic_seg = np.array(Image.open(filename)) 

        if gt_semantic_seg.shape[-1] == 1:  # Check if the last dimension is 1
            gt_semantic_seg = np.squeeze(gt_semantic_seg, axis=-1)  # Squeeze to remove the extra dimension

        # print("gt_semantic_seg.shape:",gt_semantic_seg.shape)

        results["gt_semantic_seg"] = gt_semantic_seg
        results["seg_fields"].append("gt_semantic_seg")
        return results