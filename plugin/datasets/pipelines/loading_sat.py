"""AID4AD satellite image loading pipeline.

Loads pre-cropped, ego-aligned aerial satellite tiles from AID4AD dataset.
Each tile is indexed by NuScenes sample token.
"""
import os
import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadAID4ADSatelliteImage(object):
    """Load AID4AD satellite tile for a sample.

    AID4AD provides ego-aligned, pre-cropped high-res aerial imagery
    at 0.15 m/px with ~0.16m mean alignment error.

    Original tiles: 400x200 RGB.
    By default keeps original resolution (400x200) to preserve fine details
    like lane dividers. The SatelliteEncoder handles downsampling to BEV size
    via its FPN backbone.

    Set canvas_size=(200, 100) to use downsampled version instead.

    Args:
        data_root (str): Root directory of AID4AD frames.
            Expected structure: {data_root}/{city}/{counter}_{token}.png
        canvas_size (tuple): Target (W, H) size. None = keep original 400x200.
        normalize (bool): Apply ImageNet normalization.
    """

    CITIES = [
        'boston-seaport', 'singapore-onenorth',
        'singapore-hollandvillage', 'singapore-queenstown',
    ]

    def __init__(self,
                 data_root,
                 canvas_size=None,
                 normalize=True):
        self.data_root = data_root
        self.canvas_size = canvas_size  # (W, H) or None for original
        self.normalize = normalize

        # Build token -> relative path lookup once
        self.token_lookup = {}
        for city in self.CITIES:
            city_dir = os.path.join(data_root, city)
            if not os.path.isdir(city_dir):
                continue
            for fname in os.listdir(city_dir):
                if fname.endswith('.png'):
                    tok = fname.split('_', 1)[1].replace('.png', '')
                    self.token_lookup[tok] = os.path.join(city, fname)

        print(f'[LoadAID4ADSatelliteImage] Indexed {len(self.token_lookup)} tiles '
              f'from {data_root} (canvas={canvas_size or "original 400x200"})')

    def __call__(self, results):
        token = results['token']
        rel_path = self.token_lookup.get(token)

        if rel_path is not None:
            fpath = os.path.join(self.data_root, rel_path)
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.flipud(img).copy()  # flip for BEV convention
            if self.canvas_size is not None:
                W, H = self.canvas_size
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
        else:
            # Fallback: zero image
            if self.canvas_size is not None:
                W, H = self.canvas_size
            else:
                W, H = 400, 200  # AID4AD original size
            img = np.zeros((H, W, 3), dtype=np.float32)

        if self.normalize:
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            img = (img - mean) / std

        # HWC -> CHW
        results['sat_img'] = img.transpose(2, 0, 1)  # (3, H, W)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'data_root={self.data_root}, '
                f'canvas_size={self.canvas_size})')
