# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DRIVEDataset(CustomDataset):
    """DRIVE dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """

    # CLASSES = ('background', 'vessel')
    CLASSES = ('background','headerLogo', 'recieverAddress', 'text', 'senderAddress', 'ortDatum', 'companyInfo', 'postBarcode',
               'signatureImage', 'copyLogo', 'footerLogo', 'footerText', 'twoColTable', 'multiColTable', 'boxTable','fullTableTyp1')
    # PALETTE = [[120, 120, 120], [6, 230, 230]]
    PALETTE = [[250, 250, 250],[120, 120, 120], [180, 20, 120], [6, 230, 230], [80, 150, 50],
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [119, 11, 32] , [110, 11, 255], [222, 150, 50]]
    def __init__(self, **kwargs):
        super(DRIVEDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
