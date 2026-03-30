#!/usr/bin/env python3
"""
Generate SD map cache v2 for all nuScenes samples.

v2 cache structure per sample:
  way_geometry:      List[float32[Pi, 2]]  — vehicle way centerlines (ego BEV, meters), variable points
  way_tags:
    highway_class:   int64[N]              — 0~10
    lanes:           int64[N]              — actual or -1 (missing / secondary / secondary_link)
    width:           float32[N]            — actual or -1.0 (missing)
    city:            int64[N]              — 0~3
  crossing_geometry: List[float32[Qj, 2]]  — ped crossing ways (ego BEV, meters), variable points
  crossing_city:     int64[M]              — 0~3, same encoding as way_tags.city

Usage:
    python tools/generate_sd_cache_v2.py --split train --newsplit
    python tools/generate_sd_cache_v2.py --split val --newsplit
"""

import argparse
import os
import pickle
import numpy as np
from collections import defaultdict

from pyquaternion import Quaternion
from shapely.geometry import LineString, box
from tqdm import tqdm

# Reuse from v1
from generate_sd_prior_cache import (
    ORIGIN_LATLON, EARTH_RADIUS_METERS,
    parse_osm_file, wgs84_to_city_batch, quaternion_yaw,
    clip_to_patch,
)


# ---------------------------------------------------------------------------
# v2 constants
# ---------------------------------------------------------------------------

VEHICLE_HIGHWAY_V2 = {
    'trunk', 'trunk_link',
    'primary', 'primary_link',
    'secondary', 'secondary_link',
    'tertiary', 'tertiary_link',
    'residential', 'unclassified',
    'service',
}

HIGHWAY_CLASS_MAP_V2 = {
    'trunk': 0, 'trunk_link': 1,
    'primary': 2, 'primary_link': 3,
    'secondary': 4, 'secondary_link': 5,
    'tertiary': 6, 'tertiary_link': 7,
    'residential': 8, 'unclassified': 9,
    'service': 10,
}

CITY_MAP = {
    'boston-seaport': 0,
    'singapore-onenorth': 1,
    'singapore-hollandvillage': 2,
    'singapore-queenstown': 3,
}

# secondary / secondary_link → lanes 마스킹
LANES_MASK_HIGHWAYS = {'secondary', 'secondary_link'}

ROI_SIZE = (60, 30)
MIN_WAY_LENGTH = 1.0       # meters
MIN_CROSSING_LENGTH = 0.5  # meters


# ---------------------------------------------------------------------------
# OSM loading (v2: uses VEHICLE_HIGHWAY_V2)
# ---------------------------------------------------------------------------

def load_city_osm_v2(city_name, osm_root):
    """Load OSM data with VEHICLE_HIGHWAY_V2 filter (includes service)."""
    cache_file = os.path.join(osm_root, f'{city_name}.osm')
    if not os.path.exists(cache_file):
        cache_file = os.path.join(osm_root, f'{city_name}_roads.osm')
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f'OSM file not found: {cache_file}')

    nodes, ways = parse_osm_file(cache_file)

    # Batch coordinate conversion
    node_ids = list(nodes.keys())
    node_lats = np.array([nodes[nid].lat for nid in node_ids])
    node_lons = np.array([nodes[nid].lon for nid in node_ids])
    xs, ys = wgs84_to_city_batch(node_lats, node_lons, city_name)
    node_city_coords = {nid: (x, y) for nid, x, y in zip(node_ids, xs, ys)}

    vehicle_ways = {}
    for wid, way in ways.items():
        if way.tags.get('highway') not in VEHICLE_HIGHWAY_V2:
            continue
        coords = [node_city_coords[nid] for nid in way.node_ids if nid in node_city_coords]
        if len(coords) >= 2:
            vehicle_ways[wid] = {
                'coords': np.array(coords),
                'tags': way.tags,
                'line': LineString(coords),
            }

    crossing_ways = {}
    for wid, way in ways.items():
        is_crossing = (
            way.tags.get('footway') == 'crossing'
            or 'crossing' in way.tags
            or way.tags.get('highway') == 'crossing'
        )
        if not is_crossing:
            continue
        coords = [node_city_coords[nid] for nid in way.node_ids if nid in node_city_coords]
        if len(coords) >= 2:
            crossing_ways[wid] = {
                'coords': np.array(coords),
                'tags': way.tags,
                'line': LineString(coords),
            }

    print(f'  {city_name}: {len(nodes)} nodes, {len(vehicle_ways)} vehicle ways '
          f'(v2 filter), {len(crossing_ways)} crossing ways')
    return {'vehicle_ways': vehicle_ways, 'crossing_ways': crossing_ways}


# ---------------------------------------------------------------------------
# Way extraction & tag encoding
# ---------------------------------------------------------------------------

def extract_and_encode(vehicle_ways, crossing_ways, l2g_translation, l2g_rotation,
                       city_name, roi_size=ROI_SIZE):
    """Extract OSM ways in ROI, transform to ego BEV, encode tags.

    Returns dict with:
        way_geometry:      List[float32[Pi, 2]] — raw OSM points per way (variable length)
        way_tags:          {highway_class: int64[N], lanes: int64[N],
                            width: float32[N], city: int64[N]}
        crossing_geometry: List[float32[Qj, 2]] — raw OSM points per crossing (variable length)
        crossing_city:     int64[M] — city id for each crossing (0~3)
    """
    ego_x, ego_y = l2g_translation[0], l2g_translation[1]
    ego_yaw = quaternion_yaw(l2g_rotation)
    cos_a, sin_a = np.cos(-ego_yaw), np.sin(-ego_yaw)

    def global_to_ego(line):
        coords = np.array(line.coords)
        dx, dy = coords[:, 0] - ego_x, coords[:, 1] - ego_y
        rx, ry = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a
        return LineString(np.stack([ry, -rx], axis=1))

    margin = max(roi_size) * 0.7
    global_search = box(ego_x - margin, ego_y - margin, ego_x + margin, ego_y + margin)
    local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2, roi_size[0] / 2, roi_size[1] / 2)

    city_id = CITY_MAP.get(city_name, 0)

    # --- Vehicle ways ---
    way_geoms = []
    hw_classes = []
    lanes_list = []
    width_list = []
    city_list = []

    for wdata in vehicle_ways.values():
        if not wdata['line'].intersects(global_search):
            continue
        ego_line = global_to_ego(wdata['line'])
        for seg in clip_to_patch(ego_line, local_patch):
            if seg.length < MIN_WAY_LENGTH:
                continue
            way_geoms.append(np.array(seg.coords, dtype=np.float32))

            tags = wdata['tags']
            hw = tags.get('highway', 'unclassified')
            hw_classes.append(HIGHWAY_CLASS_MAP_V2.get(hw, 9))

            # lanes: -1 if missing or secondary/secondary_link
            if hw in LANES_MASK_HIGHWAYS:
                lanes_list.append(-1)
            else:
                try:
                    lanes_list.append(int(tags['lanes']))
                except (KeyError, ValueError):
                    lanes_list.append(-1)

            # width: -1.0 if missing
            try:
                width_list.append(float(tags['width']))
            except (KeyError, ValueError):
                width_list.append(-1.0)

            city_list.append(city_id)

    # --- Crossing ways ---
    crossing_geoms = []
    crossing_city_list = []
    for wdata in crossing_ways.values():
        if not wdata['line'].intersects(global_search):
            continue
        ego_line = global_to_ego(wdata['line'])
        for seg in clip_to_patch(ego_line, local_patch):
            if seg.length < MIN_CROSSING_LENGTH:
                continue
            crossing_geoms.append(np.array(seg.coords, dtype=np.float32))
            crossing_city_list.append(city_id)

    # --- Pack ---
    N = len(way_geoms)
    M = len(crossing_geoms)

    result = {
        'way_geometry': way_geoms,          # List[float32[Pi, 2]], length N
        'way_tags': {
            'highway_class': np.array(hw_classes, dtype=np.int64) if N > 0
                             else np.zeros(0, dtype=np.int64),
            'lanes': np.array(lanes_list, dtype=np.int64) if N > 0
                     else np.zeros(0, dtype=np.int64),
            'width': np.array(width_list, dtype=np.float32) if N > 0
                     else np.zeros(0, dtype=np.float32),
            'city': np.array(city_list, dtype=np.int64) if N > 0
                    else np.zeros(0, dtype=np.int64),
        },
        'crossing_geometry': crossing_geoms,  # List[float32[Qj, 2]], length M
        'crossing_city': np.array(crossing_city_list, dtype=np.int64) if M > 0
                         else np.zeros(0, dtype=np.int64),
    }
    return result, N, M


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Generate SD map cache v2')
    parser.add_argument('--data-root', type=str, default='datasets/nuscenes')
    parser.add_argument('--osm-root', type=str, default='datasets/nuscenes/osm_tile_cache')
    parser.add_argument('--out-dir', type=str, default='datasets/nuscenes/sdmap/kyungmin')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'])
    parser.add_argument('--newsplit', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    suffix = '_newsplit' if args.newsplit else ''
    ann_file = os.path.join(args.data_root, f'nuscenes_map_infos_{args.split}{suffix}.pkl')
    print(f'Loading annotations from {ann_file}')
    with open(ann_file, 'rb') as f:
        samples = pickle.load(f)
    print(f'  {len(samples)} samples')

    # Load OSM data (v2 filter: includes service, excludes motorway)
    cities = list(CITY_MAP.keys())
    print('Loading OSM data (v2 filter)...')
    city_osm = {}
    for city in cities:
        city_osm[city] = load_city_osm_v2(city, args.osm_root)

    # Output
    out_file = os.path.join(args.out_dir, f'sd_cache_v3_{args.split}{suffix}.pkl')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    cache = {}
    stats = defaultdict(int)

    for sample in tqdm(samples, desc=f'Processing {args.split}'):
        token = sample['token']
        location = sample['location']

        if location not in city_osm:
            stats['skipped'] += 1
            continue

        # Compute lidar2global
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        ego2global[:3, 3] = sample['e2g_translation']
        l2g = ego2global @ lidar2ego
        l2g_t = list(l2g[:3, 3].astype(float))
        l2g_r = list(Quaternion(matrix=l2g).q)

        # Extract & encode
        osm_data = city_osm[location]
        entry, n_ways, n_cross = extract_and_encode(
            osm_data['vehicle_ways'], osm_data['crossing_ways'],
            l2g_t, l2g_r, location)

        cache[token] = entry

        stats['processed'] += 1
        stats['total_ways'] += n_ways
        stats['total_crossings'] += n_cross
        stats['total_way_points'] += sum(g.shape[0] for g in entry['way_geometry'])
        stats['total_crossing_points'] += sum(g.shape[0] for g in entry['crossing_geometry'])
        # Tag stats
        if n_ways > 0:
            stats['lanes_valid'] += int((entry['way_tags']['lanes'] != -1).sum())
            stats['lanes_masked'] += int((entry['way_tags']['lanes'] == -1).sum())
            stats['width_valid'] += int((entry['way_tags']['width'] != -1.0).sum())
            stats['width_masked'] += int((entry['way_tags']['width'] == -1.0).sum())

    # Save
    with open(out_file, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(out_file) / (1024 * 1024)
    print(f'\nSaved to {out_file} ({file_size:.1f} MB)')

    # Stats
    n = stats['processed']
    tw = stats['total_ways']
    tc = stats['total_crossings']
    tp_way = stats['total_way_points']
    tp_cross = stats['total_crossing_points']
    print(f'\n{"="*40}')
    print(f'Split            : {args.split}{suffix}')
    print(f'Processed        : {n}')
    print(f'Skipped          : {stats["skipped"]}')
    print(f'Total ways       : {tw} (avg {tw/max(n,1):.1f}/sample)')
    print(f'Total crossings  : {tc} (avg {tc/max(n,1):.1f}/sample)')
    print(f'Way points       : {tp_way} (avg {tp_way/max(tw,1):.1f}/way)')
    print(f'Crossing points  : {tp_cross} (avg {tp_cross/max(tc,1):.1f}/crossing)')
    print(f'Lanes valid      : {stats["lanes_valid"]} ({100*stats["lanes_valid"]/max(tw,1):.1f}%)')
    print(f'Lanes masked(-1) : {stats["lanes_masked"]} ({100*stats["lanes_masked"]/max(tw,1):.1f}%)')
    print(f'Width valid      : {stats["width_valid"]} ({100*stats["width_valid"]/max(tw,1):.1f}%)')
    print(f'Width masked(-1) : {stats["width_masked"]} ({100*stats["width_masked"]/max(tw,1):.1f}%)')
    print(f'{"="*40}')


if __name__ == '__main__':
    main()
