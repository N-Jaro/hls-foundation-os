
import os
from PIL import Image
import numpy as np

import numpy as np
import rioxarray
import torchvision.transforms.functional as F
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import PIPELINES
from torchvision import transforms

# def open_tiff(fname):
#     data = rioxarray.open_rasterio(fname)
#     return data.to_numpy()

@PIPELINES.register_module()
class LoadMapSegDataPatch(object):
    """

    It loads a tiff image. Returns in channels last format.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        nodata (float/int): no data value to substitute to nodata_replace
        nodata_replace (float/int): value to use to replace no data
    """

    def __init__(self, to_float32=False, nodata=None, nodata_replace=0.0):
        self.to_float32 = to_float32
        self.nodata = nodata
        self.nodata_replace = nodata_replace

    def __call__(self, results):

        print('filename:', results["img_info"]["filename"])

        if results.get("img_prefix") is not None:
            filename = os.path.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]

        patch_name = os.path.basename(filename)
        base_name = patch_name.split('_poly')[0]
        base_folder = 'training' if 'training' in filename else 'validation'
        data_dir='/projects/bbym/shared/all_patched_data/'
        legend_path = os.path.join(data_dir, base_folder,'poly', 'legend', base_name + '_poly.png')
        
        # lgnd_frame = Image.open(legend_path)
        # lgnd = np.array(lgnd_frame.getdata())

        # # Handle potential channel conversions (RGB, RGBA, Grayscale)
        # if len(lgnd.shape) == 2:  # Grayscale
        #     lgnd = np.expand_dims(lgnd, axis=-1)  # Add a channel dimension
        # elif len(lgnd.shape) == 3 and lgnd.shape[2] == 4: # RGBA 
        #     lgnd = lgnd[:, :, :3]  # Discard the alpha channel

        # lgnd = np.transpose(img, (1, 2, 0))

        im_frame = Image.open(filename)
        img = np.array(im_frame.getdata())

        # Handle potential channel conversions (RGB, RGBA, Grayscale)
        if len(img.shape) == 2:  # Grayscale
            img = np.expand_dims(img, axis=-1)  # Add a channel dimension
        elif len(img.shape) == 3 and img.shape[2] == 4: # RGBA 
            img = img[:, :, :3]  # Discard the alpha channel

        # img = np.transpose(img, (1, 2, 0))

        # # concatenate legend and image patches 
        # img = np.concatenate((img,lgnd), axis=0) 

        print("img.shape:",img.shape)

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
    """Load annotations for semantic segmentation.

    Args:
        to_uint8 (bool): Whether to convert the loaded label to a uint8
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        nodata (float/int): no data value to substitute to nodata_replace
        nodata_replace (float/int): value to use to replace no data


    """

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
        
        print('seg_map:', results["ann_info"]["seg_map"])

        if results.get("seg_prefix", None) is not None:
            filename = os.path.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        else:
            filename = results["ann_info"]["seg_map"]

        print("Current file:", filename)

        im_frame = Image.open(filename)
        gt_semantic_seg = np.array(im_frame.getdata())

        # gt_semantic_seg = np.transpose(gt_semantic_seg, (1, 2, 0))

        # if self.nodata is not None:
        #     gt_semantic_seg = np.where(
        #         gt_semantic_seg == self.nodata, self.nodata_replace, gt_semantic_seg
        #     )
        # reduce zero_label
        # if self.reduce_zero_label:
        #     # avoid using underflow conversion
        #     gt_semantic_seg[gt_semantic_seg == 0] = 255
        #     gt_semantic_seg = gt_semantic_seg - 1
        #     gt_semantic_seg[gt_semantic_seg == 254] = 255

        # if results.get("label_map", None) is not None:
        #     # Add deep copy to solve bug of repeatedly
        #     # replace `gt_semantic_seg`, which is reported in
        #     # https://github.com/open-mmlab/mmsegmentation/pull/1445/
        #     gt_semantic_seg_copy = gt_semantic_seg.copy()
        #     for old_id, new_id in results["label_map"].items():
        #         gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id

        print("gt_semantic_seg.shape:",gt_semantic_seg.shape)

        results["gt_semantic_seg"] = gt_semantic_seg
        results["seg_fields"].append("gt_semantic_seg")
        return results