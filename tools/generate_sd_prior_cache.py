#!/usr/bin/env python3
"""
Generate offline SD map prior cache for all nuScenes samples.

Produces a pickle file mapping sample token -> SD prior vectors (polylines,
labels, attributes, tag masks, reliability scores).

Logic ported from phase1_sdmap_vectormap.ipynb.
"""

import argparse
import os
import pickle
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List
from xml.etree import cElementTree as ET

from pyquaternion import Quaternion
from shapely.geometry import LineString, box
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORIGIN_LATLON = {
    'boston-seaport': (42.336849169438615, -71.05785369873047),
    'singapore-onenorth': (1.2882100868743724, 103.78475189208984),
    'singapore-hollandvillage': (1.2993652317780957, 103.78217697143555),
    'singapore-queenstown': (1.2782562240223188, 103.76741409301758),
}
EARTH_RADIUS_METERS = 6.378137e6

VEHICLE_HIGHWAY = {
    'motorway', 'motorway_link', 'trunk', 'trunk_link',
    'primary', 'primary_link', 'secondary', 'secondary_link',
    'tertiary', 'tertiary_link', 'residential', 'unclassified',
}

HIGHWAY_CLASS_MAP = {
    'motorway': 0, 'motorway_link': 1, 'trunk': 2, 'trunk_link': 3,
    'primary': 4, 'primary_link': 5, 'secondary': 6, 'secondary_link': 7,
    'tertiary': 8, 'tertiary_link': 9, 'residential': 10, 'unclassified': 11,
}

DEFAULT_LANES = {
    'motorway': 3, 'motorway_link': 1, 'trunk': 2, 'trunk_link': 1,
    'primary': 2, 'primary_link': 1, 'secondary': 2, 'secondary_link': 1,
    'tertiary': 2, 'tertiary_link': 1, 'residential': 2, 'unclassified': 2,
}

DEFAULT_WIDTH = {
    'motorway': 14.0, 'motorway_link': 4.5, 'trunk': 10.0, 'trunk_link': 4.5,
    'primary': 10.0, 'primary_link': 4.5, 'secondary': 8.0, 'secondary_link': 4.0,
    'tertiary': 7.0, 'tertiary_link': 3.5, 'residential': 6.0, 'unclassified': 6.0,
}

# Label encoding: 0=ped_crossing, 1=divider, 2=boundary
LABEL_PED = 0
LABEL_DIV = 1
LABEL_BND = 2

NUM_POINTS = 20
ROI_SIZE = (60, 30)
MIN_LENGTH = 3.0  # metres – filter fragments shorter than this


# ---------------------------------------------------------------------------
# OSM data structures & parsing
# ---------------------------------------------------------------------------

@dataclass
class OSMWay:
    id: int
    node_ids: List[int]
    tags: Dict[str, str]


@dataclass
class OSMNode:
    id: int
    lat: float
    lon: float
    tags: Dict[str, str]


def parse_osm_file(filepath):
    """Parse an .osm XML file, return (nodes, ways) dicts."""
    nodes, ways = {}, {}
    tree = ET.parse(filepath)
    root = tree.getroot()
    for elem in root:
        if elem.tag == 'node':
            tags = {t.attrib['k']: t.attrib['v'] for t in elem.findall('tag')}
            nodes[int(elem.attrib['id'])] = OSMNode(
                int(elem.attrib['id']),
                float(elem.attrib['lat']),
                float(elem.attrib['lon']),
                tags,
            )
        elif elem.tag == 'way':
            nids = [int(nd.attrib['ref']) for nd in elem.findall('nd')]
            tags = {t.attrib['k']: t.attrib['v'] for t in elem.findall('tag')}
            ways[int(elem.attrib['id'])] = OSMWay(int(elem.attrib['id']), nids, tags)
    return nodes, ways


# ---------------------------------------------------------------------------
# Coordinate conversion (Haversine – same as SDTagNet / phase1 notebook)
# ---------------------------------------------------------------------------

def wgs84_to_city_batch(lats, lons, city_name):
    """WGS84 (lat/lon) -> nuScenes city-frame (x, y) in metres."""
    origin_lat, origin_lon = ORIGIN_LATLON[city_name]
    phi_1 = origin_lat * np.pi / 180
    phi_2 = lats * np.pi / 180
    lambda_1 = origin_lon * np.pi / 180
    lambda_2 = lons * np.pi / 180
    delta_phi = (lats - origin_lat) * np.pi / 180
    delta_lambda = (lons - origin_lon) * np.pi / 180

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = EARTH_RADIUS_METERS * c

    y = np.sin(lambda_2 - lambda_1) * np.cos(phi_2)
    x = np.cos(phi_1) * np.sin(phi_2) - np.sin(phi_1) * np.cos(phi_2) * np.cos(lambda_2 - lambda_1)
    theta = np.arctan2(y, x)
    return d * np.sin(theta), d * np.cos(theta)


# ---------------------------------------------------------------------------
# OSM loading & preprocessing (per city)
# ---------------------------------------------------------------------------

def load_city_osm(city_name, osm_root):
    """Load and preprocess OSM data for a single city.

    Returns dict with 'vehicle_ways' and 'crossing_ways', each value being
    a dict  wid -> {coords, tags, line}.
    """
    cache_file = os.path.join(osm_root, f'{city_name}_roads.osm')
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f'OSM cache not found: {cache_file}')
    nodes, ways = parse_osm_file(cache_file)

    # Batch convert all node coords to city frame
    node_ids = list(nodes.keys())
    node_lats = np.array([nodes[nid].lat for nid in node_ids])
    node_lons = np.array([nodes[nid].lon for nid in node_ids])
    xs, ys = wgs84_to_city_batch(node_lats, node_lons, city_name)
    node_city_coords = {nid: (x, y) for nid, x, y in zip(node_ids, xs, ys)}

    vehicle_ways = {}
    for wid, way in ways.items():
        if way.tags.get('highway') not in VEHICLE_HIGHWAY:
            continue
        coords_city = [node_city_coords[nid] for nid in way.node_ids if nid in node_city_coords]
        if len(coords_city) >= 2:
            vehicle_ways[wid] = {
                'coords': np.array(coords_city),
                'tags': way.tags,
                'line': LineString(coords_city),
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
        coords_city = [node_city_coords[nid] for nid in way.node_ids if nid in node_city_coords]
        if len(coords_city) >= 2:
            crossing_ways[wid] = {
                'coords': np.array(coords_city),
                'tags': way.tags,
                'line': LineString(coords_city),
            }

    print(f'  {city_name}: {len(nodes)} nodes, {len(vehicle_ways)} vehicle ways, '
          f'{len(crossing_ways)} crossing ways')
    return {'vehicle_ways': vehicle_ways, 'crossing_ways': crossing_ways}


# ---------------------------------------------------------------------------
# Geometry helpers (from phase1 notebook)
# ---------------------------------------------------------------------------

def quaternion_yaw(q):
    """Extract yaw from a Quaternion (or list [w,x,y,z])."""
    if not isinstance(q, Quaternion):
        q = Quaternion(q)
    # Use rotation matrix to extract yaw
    rot = q.rotation_matrix
    return np.arctan2(rot[1, 0], rot[0, 0])


def offset_line(line, distance):
    """Parallel offset of a LineString. Returns None on failure."""
    try:
        offset = line.parallel_offset(abs(distance), 'left' if distance > 0 else 'right')
        if offset.is_empty:
            return None
        if offset.geom_type == 'MultiLineString':
            offset = max(offset.geoms, key=lambda l: l.length)
        if distance < 0:
            offset = LineString(list(offset.coords)[::-1])
        return offset
    except Exception:
        return None


def clip_to_patch(geom, patch):
    """Clip geometry to a rectangular patch, return list of LineStrings."""
    results = []
    if geom is None:
        return results
    clipped = geom.intersection(patch)
    if clipped.is_empty:
        return results
    if clipped.geom_type == 'LineString' and len(clipped.coords) >= 2:
        results.append(clipped)
    elif clipped.geom_type == 'MultiLineString':
        for seg in clipped.geoms:
            if len(seg.coords) >= 2:
                results.append(seg)
    return results


# ---------------------------------------------------------------------------
# SD Map vector generation (core logic from phase1 notebook)
# ---------------------------------------------------------------------------

def generate_sdmap_vectors(vehicle_ways, crossing_ways, l2g_translation, l2g_rotation, roi_size):
    """Generate SD map vectors in ego (lidar) BEV frame.

    Returns dict with keys 'divider', 'boundary', 'ped_crossing',
    each a list of (LineString, tags_dict).  *tags_dict* is the original OSM
    tags for vehicle ways (empty dict for crossings).
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

    dividers, boundaries, ped_crossings = [], [], []

    for wdata in vehicle_ways.values():
        way_line, tags = wdata['line'], wdata['tags']
        if not way_line.intersects(global_search):
            continue

        hw_type = tags.get('highway', 'residential')
        try:
            num_lanes = int(tags.get('lanes', DEFAULT_LANES.get(hw_type, 2)))
        except ValueError:
            num_lanes = DEFAULT_LANES.get(hw_type, 2)
        try:
            width = float(tags.get('width', DEFAULT_WIDTH.get(hw_type, 7.0)))
        except ValueError:
            width = DEFAULT_WIDTH.get(hw_type, 7.0)
        is_oneway = tags.get('oneway', 'no').lower() in ('yes', 'true', '1')

        # Boundaries (both sides)
        for sign in [1, -1]:
            bnd = offset_line(way_line, sign * width / 2)
            if bnd:
                for seg in clip_to_patch(global_to_ego(bnd), local_patch):
                    boundaries.append((seg, tags))

        # Dividers
        if num_lanes >= 2:
            lane_w = width / num_lanes
            if is_oneway:
                for i in range(1, num_lanes):
                    div = offset_line(way_line, -width / 2 + i * lane_w)
                    if div:
                        for seg in clip_to_patch(global_to_ego(div), local_patch):
                            dividers.append((seg, tags))
            else:
                for seg in clip_to_patch(global_to_ego(way_line), local_patch):
                    dividers.append((seg, tags))
                lanes_per_dir = num_lanes // 2
                if lanes_per_dir >= 2:
                    dir_lane_w = (width / 2) / lanes_per_dir
                    for side_sign in [1, -1]:
                        for i in range(1, lanes_per_dir):
                            div = offset_line(way_line, side_sign * i * dir_lane_w)
                            if div:
                                for seg in clip_to_patch(global_to_ego(div), local_patch):
                                    dividers.append((seg, tags))

    # Pedestrian crossings
    for wdata in crossing_ways.values():
        if wdata['line'].intersects(global_search):
            for seg in clip_to_patch(global_to_ego(wdata['line']), local_patch):
                ped_crossings.append((seg, wdata['tags']))

    return {
        'divider': dividers,
        'boundary': boundaries,
        'ped_crossing': ped_crossings,
    }


# ---------------------------------------------------------------------------
# Post-processing: sampling, normalisation, attribute extraction
# ---------------------------------------------------------------------------

def uniform_sample_polyline(coords, num_points):
    """Resample a polyline to exactly *num_points* uniformly spaced points."""
    diffs = np.diff(coords, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len < 1e-6:
        return np.tile(coords[0], (num_points, 1))
    targets = np.linspace(0, total_len, num_points)
    sampled = np.zeros((num_points, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        idx = np.searchsorted(cum_len, t, side='right') - 1
        idx = np.clip(idx, 0, len(seg_lens) - 1)
        if seg_lens[idx] < 1e-8:
            sampled[i] = coords[idx]
        else:
            frac = (t - cum_len[idx]) / seg_lens[idx]
            frac = np.clip(frac, 0.0, 1.0)
            sampled[i] = coords[idx] + frac * diffs[idx]
    return sampled


def extract_attrs(tags, hw_type):
    """Extract (lanes, width, oneway, highway_class) and has_tag_mask."""
    has_lanes = 'lanes' in tags
    has_width = 'width' in tags

    try:
        lanes = float(tags['lanes']) if has_lanes else float(DEFAULT_LANES.get(hw_type, 2))
    except (ValueError, TypeError):
        lanes = float(DEFAULT_LANES.get(hw_type, 2))
        has_lanes = False

    try:
        width = float(tags['width']) if has_width else float(DEFAULT_WIDTH.get(hw_type, 7.0))
    except (ValueError, TypeError):
        width = float(DEFAULT_WIDTH.get(hw_type, 7.0))
        has_width = False

    oneway = 1.0 if tags.get('oneway', 'no').lower() in ('yes', 'true', '1') else 0.0
    highway_class = float(HIGHWAY_CLASS_MAP.get(hw_type, 11))

    attrs = np.array([lanes, width, oneway, highway_class], dtype=np.float32)
    has_tag_mask = np.array([has_lanes, has_width], dtype=bool)
    return attrs, has_tag_mask


def compute_reliability(has_lanes, has_width):
    """Reliability score based on available OSM tag info."""
    if has_lanes and has_width:
        return 1.0
    elif has_lanes:
        return 0.7
    else:
        return 0.4


def process_sample_sd_prior(vehicle_ways, crossing_ways, l2g_translation, l2g_rotation,
                            roi_size=ROI_SIZE, num_points=NUM_POINTS, min_length=MIN_LENGTH):
    """Generate the full SD prior dict for a single sample."""
    sd_map = generate_sdmap_vectors(vehicle_ways, crossing_ways,
                                    l2g_translation, l2g_rotation, roi_size)

    all_polylines = []
    all_labels = []
    all_attrs = []
    all_has_tag = []
    all_reliability = []

    label_map = {
        'ped_crossing': LABEL_PED,
        'divider': LABEL_DIV,
        'boundary': LABEL_BND,
    }

    for cls_name, label_id in label_map.items():
        for (geom, tags) in sd_map.get(cls_name, []):
            coords = np.array(geom.coords)
            length = geom.length
            # Filter short fragments
            if length < min_length:
                continue

            # Sample to fixed number of points
            sampled = uniform_sample_polyline(coords, num_points)

            # Normalize to [0, 1] w.r.t. roi_size
            # BEV coords are in [-roi_size[0]/2, roi_size[0]/2] x [-roi_size[1]/2, roi_size[1]/2]
            sampled[:, 0] = (sampled[:, 0] + roi_size[0] / 2) / roi_size[0]
            sampled[:, 1] = (sampled[:, 1] + roi_size[1] / 2) / roi_size[1]
            # Clip to [0, 1]
            sampled = np.clip(sampled, 0.0, 1.0)

            all_polylines.append(sampled.astype(np.float32))
            all_labels.append(label_id)

            # Extract attributes
            hw_type = tags.get('highway', 'residential')
            attrs, has_tag_mask = extract_attrs(tags, hw_type)
            all_attrs.append(attrs)
            all_has_tag.append(has_tag_mask)
            all_reliability.append(compute_reliability(has_tag_mask[0], has_tag_mask[1]))

    # Build output arrays
    n = len(all_polylines)
    if n == 0:
        return {
            'polylines': np.zeros((0, num_points, 2), dtype=np.float32),
            'labels': np.zeros((0,), dtype=np.int64),
            'attrs': np.zeros((0, 4), dtype=np.float32),
            'has_tag_mask': np.zeros((0, 2), dtype=bool),
            'reliability': np.zeros((0,), dtype=np.float32),
        }

    return {
        'polylines': np.stack(all_polylines, axis=0),                    # [N, 20, 2]
        'labels': np.array(all_labels, dtype=np.int64),                  # [N]
        'attrs': np.stack(all_attrs, axis=0),                            # [N, 4]
        'has_tag_mask': np.stack(all_has_tag, axis=0),                   # [N, 2]
        'reliability': np.array(all_reliability, dtype=np.float32),      # [N]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Generate SD map prior cache for nuScenes')
    parser.add_argument('--data-root', type=str, default='datasets/nuscenes',
                        help='Path to datasets/nuscenes')
    parser.add_argument('--osm-root', type=str, default='/home/kyungmin/min_ws/mapping',
                        help='Path to OSM cache directory')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'],
                        help='Dataset split')
    parser.add_argument('--newsplit', action='store_true',
                        help='Use the newsplit (geographical-based split)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine annotation file path
    suffix = '_newsplit' if args.newsplit else ''
    ann_file = os.path.join(args.data_root, f'nuscenes_map_infos_{args.split}{suffix}.pkl')
    print(f'Loading annotations from {ann_file} ...')
    with open(ann_file, 'rb') as f:
        samples = pickle.load(f)
    print(f'  {len(samples)} samples loaded.')

    # Load OSM data for all 4 cities
    cities = ['boston-seaport', 'singapore-hollandvillage',
              'singapore-onenorth', 'singapore-queenstown']
    print('Loading OSM data ...')
    city_osm = {}
    for city in cities:
        city_osm[city] = load_city_osm(city, args.osm_root)

    # Process each sample
    sd_prior_cache = {}
    stats = defaultdict(int)
    total_polylines = 0

    for sample in tqdm(samples, desc=f'Processing {args.split}'):
        token = sample['token']
        location = sample['location']

        if location not in city_osm:
            print(f'  WARNING: unknown location {location} for token {token}, skipping.')
            stats['skipped'] += 1
            continue

        # Compute lidar2global pose (same as nusc_dataset.py)
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample['lidar2ego_translation']

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        ego2global[:3, 3] = sample['e2g_translation']

        lidar2global = ego2global @ lidar2ego
        l2g_translation = list(lidar2global[:3, 3].astype(float))
        l2g_rotation = list(Quaternion(matrix=lidar2global).q)

        osm_data = city_osm[location]
        prior = process_sample_sd_prior(
            osm_data['vehicle_ways'],
            osm_data['crossing_ways'],
            l2g_translation,
            l2g_rotation,
            roi_size=ROI_SIZE,
        )

        sd_prior_cache[token] = prior
        n = prior['polylines'].shape[0]
        total_polylines += n
        stats['processed'] += 1
        if n == 0:
            stats['empty'] += 1

    # Save cache
    out_file = os.path.join(args.data_root, f'sd_prior_cache_{args.split}{suffix}.pkl')
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(sd_prior_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nSaved cache to {out_file}')

    # Print statistics
    print('\n========== Statistics ==========')
    print(f'Split          : {args.split} {"(newsplit)" if args.newsplit else "(oldsplit)"}')
    print(f'Total samples  : {len(samples)}')
    print(f'Processed      : {stats["processed"]}')
    print(f'Skipped        : {stats["skipped"]}')
    print(f'Empty (0 vecs) : {stats["empty"]}')
    print(f'Total polylines: {total_polylines}')
    if stats['processed'] > 0:
        avg = total_polylines / stats['processed']
        print(f'Avg per sample : {avg:.1f}')

    # Per-class breakdown
    n_ped = sum(int((v['labels'] == LABEL_PED).sum()) for v in sd_prior_cache.values())
    n_div = sum(int((v['labels'] == LABEL_DIV).sum()) for v in sd_prior_cache.values())
    n_bnd = sum(int((v['labels'] == LABEL_BND).sum()) for v in sd_prior_cache.values())
    print(f'  ped_crossing : {n_ped}')
    print(f'  divider      : {n_div}')
    print(f'  boundary     : {n_bnd}')

    # Reliability distribution
    all_rel = np.concatenate([v['reliability'] for v in sd_prior_cache.values()
                              if v['reliability'].shape[0] > 0])
    if len(all_rel) > 0:
        print(f'Reliability    : mean={all_rel.mean():.3f}, '
              f'1.0={int((all_rel == 1.0).sum())}, '
              f'0.7={int((all_rel == 0.7).sum())}, '
              f'0.4={int((all_rel == 0.4).sum())}')
    print('================================')


if __name__ == '__main__':
    main()
