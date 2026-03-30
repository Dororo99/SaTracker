import os
import pickle

from.base_dataset import BaseMapDataset
from .map_utils.nuscmap_extractor import NuscMapExtractor
from mmdet.datasets import DATASETS
import numpy as np
import cv2
from PIL import Image
from .visualize.renderer import Renderer
import mmcv
from time import time
from pyquaternion import Quaternion


@DATASETS.register_module()
class NuscDataset(BaseMapDataset):
    """NuScenes map dataset class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """
    
    def __init__(self, data_root, sd_prior_cache_path=None, osm_tile_dir=None,
                 sd_raster_cache_path=None, sd_raster_thickness=5,
                 sd_cache_v3_path=None, **kwargs):
        super().__init__(**kwargs)
        self.map_extractor = NuscMapExtractor(data_root, self.roi_size)
        self.renderer = Renderer(self.cat2id, self.roi_size, 'nusc')

        # Load SD prior cache
        self.sd_prior_cache = {}
        if sd_prior_cache_path is not None:
            if os.path.exists(sd_prior_cache_path):
                with open(sd_prior_cache_path, 'rb') as f:
                    self.sd_prior_cache = pickle.load(f)
                print(f'Loaded SD prior cache: {len(self.sd_prior_cache)} samples from {sd_prior_cache_path}')

        # OSM tile image directory
        self.osm_tile_dir = osm_tile_dir
        if osm_tile_dir is not None:
            print(f'OSM tile dir: {osm_tile_dir} (exists={os.path.isdir(osm_tile_dir)})')

        # SD cache v3: per-point SD prior for decoder cross-attention
        self.sd_cache_v3 = {}
        if sd_cache_v3_path is not None:
            if os.path.exists(sd_cache_v3_path):
                with open(sd_cache_v3_path, 'rb') as f:
                    self.sd_cache_v3 = pickle.load(f)
                print(f'Loaded SD cache v3: {len(self.sd_cache_v3)} samples from {sd_cache_v3_path}')

        # SD raster: rasterize SD cache v2 vectors on-the-fly
        self.sd_raster_cache = {}
        self.sd_raster_thickness = sd_raster_thickness
        if sd_raster_cache_path is not None:
            if os.path.exists(sd_raster_cache_path):
                with open(sd_raster_cache_path, 'rb') as f:
                    self.sd_raster_cache = pickle.load(f)
                print(f'Loaded SD raster cache (v2): {len(self.sd_raster_cache)} samples, thickness={sd_raster_thickness}')
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        
        start_time = time()
        ann = mmcv.load(ann_file)
        samples = ann[::self.interval]
        
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples
    
    def load_matching(self, matching_file):
        with open(matching_file, 'rb') as pf:
            data = pickle.load(pf)
        total_samples = 0
        for scene_name, info in data.items():
            total_samples += len(info['sample_ids'])
        assert total_samples == len(self.samples), 'Matching info not matched with data samples'
        self.matching_meta = data
        print(f'loaded matching meta for {len(data)} scenes')

    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        location = sample['location']

        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample['lidar2ego_translation']

        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        ego2global[:3, 3] = sample['e2g_translation']

        # NOTE: The original StreamMapNet uses the ego location to query the map,
        # to align with the lidar-centered setting in MapTR, we made some modifiactions 
        # here to switch to the lidar-center setting
        lidar2global = ego2global @ lidar2ego
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_translation = [float(x) for x in lidar2global_translation]
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        map_geoms = self.map_extractor.get_map_geom(location, lidar2global_translation, 
                lidar2global_rotation)
        
        lidar_shifted_e2g_translation = np.array(sample['e2g_translation'])
        lidar_shifted_e2g_translation[0] = lidar2global_translation[0]
        lidar_shifted_e2g_translation[1] = lidar2global_translation[1]
        lidar_shifted_e2g_translation = lidar_shifted_e2g_translation.tolist()
        e2g_rotation = sample['e2g_rotation']

        lidar2global = np.eye(4)
        lidar2global[:3,:3] = Quaternion(e2g_rotation).rotation_matrix
        lidar2global[:3, 3] = lidar_shifted_e2g_translation
        global2lidar = np.linalg.inv(lidar2global)
        
        ego2lidar = global2lidar  @ ego2global

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
        
        ego2img_rts = []
        ego2cam_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])

            # ego coord to cam coord
            #ego2cam_rt = extrinsic

            cam2ego_rt = np.linalg.inv(extrinsic)
            cam2lidar_rt = ego2lidar @ cam2ego_rt
            lidar2cam_rt = np.linalg.inv(cam2lidar_rt)
            ego2cam_rt = lidar2cam_rt

            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

            ego2img_rt = (viewpad @ ego2cam_rt)
            ego2cam_rts.append(ego2cam_rt)
            ego2img_rts.append(ego2img_rt)


        input_dict = {
            'location': location,
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, **ego2cam**
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'ego2cam': ego2cam_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            #'ego2global_translation': sample['e2g_translation'], 
            #'ego2global_rotation': Quaternion(sample['e2g_rotation']).rotation_matrix.tolist(),
            'ego2global_translation': lidar_shifted_e2g_translation, 
            'ego2global_rotation': Quaternion(e2g_rotation).rotation_matrix.tolist(),
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name'],
            'lidar2ego_translation': sample['lidar2ego_translation'],
            'lidar2ego_rotation': sample['lidar2ego_rotation'],
        }

        # SD cache v3 (per-point SD prior for decoder cross-attention)
        token = sample['token']
        if token in self.sd_cache_v3:
            input_dict['sd_cache'] = self.sd_cache_v3[token]
        else:
            input_dict['sd_cache'] = None

        # SD prior (legacy v1, for SDQueryEncoder)
        if token in self.sd_prior_cache:
            input_dict['sd_priors'] = self.sd_prior_cache[token]
        else:
            input_dict['sd_priors'] = {
                'polylines': np.zeros((0, 20, 2), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int64),
                'attrs': np.zeros((0, 4), dtype=np.float32),
                'has_tag_mask': np.zeros((0, 2), dtype=bool),
                'reliability': np.zeros((0,), dtype=np.float32),
            }

        # SD raster (from v2 cache vectors) — takes priority over OSM tile
        if self.sd_raster_cache:
            if token in self.sd_raster_cache:
                sd_v2 = self.sd_raster_cache[token]
                raster = self._rasterize_sd_vectors(
                    sd_v2['way_geometry'], sd_v2['crossing_geometry'])
            else:
                raster = np.zeros((100, 200), dtype=np.float32)
            input_dict['osm_tile'] = raster[:, :, np.newaxis]  # [H, W, 1]

        # OSM tile image (ego BEV frame, 200x100 RGB) — fallback if no sd_raster
        elif self.osm_tile_dir is not None:
            tile_path = os.path.join(self.osm_tile_dir, f'{token}.png')
            if os.path.exists(tile_path):
                tile_img = np.flipud(np.array(Image.open(tile_path))).copy()
                input_dict['osm_tile'] = tile_img.astype(np.float32) / 255.0
            else:
                input_dict['osm_tile'] = np.zeros((100, 200, 3), dtype=np.float32)

        return input_dict

    def _rasterize_sd_vectors(self, way_geometry, crossing_geometry):
        """Rasterize SD cache v2 vectors into single-channel BEV image.

        Args:
            way_geometry: list of [Pi, 2] arrays in ego BEV meters
            crossing_geometry: list of [Qj, 2] arrays in ego BEV meters

        Returns:
            raster: [H, W] float32 in {0.0, 1.0}
        """
        roi_x, roi_y = self.roi_size           # (60, 30)
        # Canvas: 10/3 px per meter (consistent with pipeline canvas_size)
        canvas_w = int(roi_x * 10 / 3)        # 200
        canvas_h = int(roi_y * 10 / 3)        # 100
        img = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for poly in way_geometry:
            if len(poly) < 2:
                continue
            col = (poly[:, 0] / roi_x + 0.5) * canvas_w
            row = (poly[:, 1] / roi_y + 0.5) * canvas_h
            pts = np.stack([col, row], axis=1).astype(np.int32)
            cv2.polylines(img, [pts], False, 1.0, self.sd_raster_thickness)

        for poly in crossing_geometry:
            if len(poly) < 2:
                continue
            col = (poly[:, 0] / roi_x + 0.5) * canvas_w
            row = (poly[:, 1] / roi_y + 0.5) * canvas_h
            pts = np.stack([col, row], axis=1).astype(np.int32)
            cv2.polylines(img, [pts], False, 1.0, self.sd_raster_thickness)

        return img