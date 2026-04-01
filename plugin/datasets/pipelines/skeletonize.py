import numpy as np
from mmdet.datasets.builder import PIPELINES
from skimage.morphology import skeletonize, dilation


@PIPELINES.register_module(force=True)
class SkeletonizeMap(object):
    """Generate skeleton mask from semantic_mask for thin-structure classes.

    Applies skeletonize + dilation to specified class channels of the
    semantic segmentation mask and stores the result in 'skeleton_mask'.

    Args:
        skel_classes (list[int]): Class indices to skeletonize.
            Default: [1, 2] (divider, boundary).
        num_dilations (int): Number of dilation iterations applied to
            the skeleton. Default: 2.
    """

    def __init__(self, skel_classes=None, num_dilations=2):
        self.skel_classes = skel_classes if skel_classes is not None else [1, 2]
        self.num_dilations = num_dilations

    def __call__(self, input_dict):
        semantic_mask = input_dict['semantic_mask']  # (C, H, W) numpy uint8
        skeleton_mask = np.zeros_like(semantic_mask)  # (C, H, W) numpy uint8

        for cls_idx in self.skel_classes:
            mask_cls = semantic_mask[cls_idx]  # (H, W)
            if mask_cls.any():
                skel = skeletonize(mask_cls > 0)  # returns bool array
                for _ in range(self.num_dilations):
                    skel = dilation(skel)
                skeleton_mask[cls_idx] = skel.astype(np.uint8)

        skeleton_mask = np.ascontiguousarray(skeleton_mask)
        input_dict['skeleton_mask'] = skeleton_mask
        return input_dict

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(skel_classes={self.skel_classes}, '
                f'num_dilations={self.num_dilations})')
