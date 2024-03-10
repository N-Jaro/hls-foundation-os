from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from .mapseg_pipeline import LoadMapSegAnnotations

        
@DATASETS.register_module()
class MapSegDataset(CustomDataset):

    def __init__(self, CLASSES=(0, 1), PALETTE=None, **kwargs):
        
        self.CLASSES = CLASSES

        self.PALETTE = PALETTE
        
        gt_seg_map_loader_cfg = kwargs.pop('gt_seg_map_loader_cfg') if 'gt_seg_map_loader_cfg' in kwargs else dict()
        # reduce_zero_label = kwargs.pop('reduce_zero_label') if 'reduce_zero_label' in kwargs else False
        
        super(MapSegDataset, self).__init__(**kwargs)

        self.gt_seg_map_loader = LoadMapSegAnnotations(**gt_seg_map_loader_cfg)