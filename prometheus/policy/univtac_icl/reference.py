"""Build and verify a one-shot TRAIN reference from one official UniVTAC HDF5 episode.

Following RoboICL, TRAIN is a sequence of non-overlapping action blocks, each with
its starting observation, an act call, a receipt and the ending observation.
UniVTAC stores measured states at 60 Hz, not logged policy commands, so a TRAIN
row is the measured motion between two retained waypoints, and its finger value
is the measured finger opening at the block's next waypoint.

Optional jam_frames (insert_tube) name source frames where the expert stopped because the held object
stalled (the end of a try_forward atom). Each becomes a retained waypoint and a block boundary, and the
block ending there is rendered as a System 1 jam stop. Every waypoint then also records the mean GelSight
marker shear relative to frame 0 (the guard's features, jam_guard.features).
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .geometry import world_delta

SCHEMA = 'univtac_icl_reference_v2'
IMAGE_KEYS = {'head': 'observation/head/rgb', 'wrist': 'observation/wrist/rgb',
              'left': 'tactile/left_tactile/rgb_marker', 'right': 'tactile/right_tactile/rgb_marker'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_waypoints(ee, finger, atom_ids, thresholds):
    """Keep a frame once motion since the last kept frame is meaningful.

    Large changes (translation_mm / finger_mm / rotation_deg) always start a new
    waypoint; at an expert atom boundary a smaller change (boundary_*) suffices,
    so phase changes such as the slight grip release stay visible.
    """
    kept = [0]
    for frame in range(1, len(ee)):
        last = kept[-1]
        moved = np.linalg.norm(ee[frame, :3] - ee[last, :3]) * 1000
        gripped = abs(finger[frame] - finger[last]) * 1000
        turned = np.degrees(np.linalg.norm(world_delta(ee[last, 3:], ee[frame, 3:])))
        boundary = frame + 1 < len(ee) and atom_ids[frame + 1] != atom_ids[frame]
        if (moved >= thresholds['translation_mm'] or gripped >= thresholds['finger_mm']
                or turned >= thresholds['rotation_deg']
                or boundary and (moved >= thresholds['boundary_translation_mm']
                                 or gripped >= thresholds['boundary_finger_mm'])):
            kept.append(frame)
    if kept[-1] != len(ee) - 1:
        kept.append(len(ee) - 1)
    return kept


def _image_ext(payload):
    return '.png' if payload[:8] == b'\x89PNG\r\n\x1a\n' else '.jpg'


def build(source, output, task, source_id, metadata, thresholds, horizon, finger_max_mm, rotation_floor_deg=1.0,
          jam_frames=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Reference output already exists: {output}')
    if type(horizon) is not int or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    # The official collector writes metadata.json keyed by episode index; only successes are saved.
    record = json.loads(Path(metadata).read_text())[source.stem]
    if record.get('result') != 'success':
        raise ValueError(f'Source episode {source.stem} is not an official success: {record.get("result")}')
    with h5py.File(source, 'r') as f:
        ee = f['embodiment/ee'][:].astype(float)
        finger = f['embodiment/joint'][:, 7:9].mean(axis=1).astype(float)
        steps = f['step'][:].astype(int)
        atom_ids = f['atom/id'][:].tolist()
        press = {side: f[f'tactile/{side}_tactile/press_depth'][:].reshape(len(ee), -1).max(axis=1)
                 for side in ('left', 'right')}
        frames = select_waypoints(ee, finger, atom_ids, thresholds)
        jam_frames = sorted(int(frame) for frame in (jam_frames or []))
        if any(not 0 < frame < len(ee) - 1 for frame in jam_frames):
            raise ValueError('jam_frames must be interior source frames')
        frames = sorted(set(frames) | set(jam_frames))
        shear = None
        if jam_frames:
            from .jam_guard import features
            fields = {pad: f[f'tactile/{pad}/marker'][:, 1] - f[f'tactile/{pad}/marker'][:, 0]
                      for pad in ('left_tactile', 'right_tactile')}
            base = {pad: field[0] for pad, field in fields.items()}
            shear = [features({pad: field[frame] for pad, field in fields.items()}, base) for frame in range(len(ee))]
        (output / 'frames').mkdir(parents=True)
        waypoints = []
        for index, frame in enumerate(frames):
            images = {}
            for name, key in IMAGE_KEYS.items():
                payload = bytes(f[key][frame])
                path = output / 'frames' / f'{index:03d}_{name}{_image_ext(payload)}'
                path.write_bytes(payload)
                images[name] = {'path': str(path.relative_to(output)), 'sha256': sha256(path)}
            waypoints.append({'frame': int(frame), 'step': int(steps[frame]), 'ee': ee[frame].tolist(),
                              'finger_m': float(finger[frame]),
                              'press_depth_mm': {side: float(press[side][frame]) for side in press},
                              'images': images})
            if shear is not None:
                waypoints[-1]['marker_shear_px'] = shear[frame]
    rows = []
    for a, b in zip(waypoints, waypoints[1:]):
        translation = (np.asarray(b['ee'][:3]) - np.asarray(a['ee'][:3])) * 1000
        rotation = np.degrees(world_delta(a['ee'][3:], b['ee'][3:]))
        if np.linalg.norm(rotation) < rotation_floor_deg:
            rotation = np.zeros(3)  # measured orientation jitter, not an expert rotation
        rows.append([round(float(v), 1) for v in translation] + [round(float(v), 1) for v in rotation]
                    + [round(min(b['finger_m'] * 1000, finger_max_mm), 1)])
    jams = [frames.index(frame) for frame in jam_frames]
    blocks = []
    for lo, hi in zip([0] + jams, jams + [len(rows)]):
        blocks += [{'start': start, 'end': min(start + horizon, hi), 'actions': rows[start:min(start + horizon, hi)],
                    'jam': start + horizon >= hi and hi in jams}
                   for start in range(lo, hi, horizon)]
    reference = {'schema': SCHEMA, 'task': task,
                 'source': {'id': source_id, 'sha256': sha256(source), 'frames': len(ee),
                            'official_result': record['result'], 'metadata_sha256': sha256(metadata)},
                 'waypoint_thresholds': thresholds, 'rotation_floor_deg': rotation_floor_deg, 'horizon': horizon,
                 'row_semantics': ('measured motion between retained waypoints (rotations below the floor zeroed); '
                                   'finger = measured opening at the next waypoint, capped at the command maximum'),
                 'jam_frames': jam_frames, 'waypoints': waypoints, 'blocks': blocks}
    (output / 'reference.json').write_text(json.dumps(reference, indent=1, allow_nan=False) + '\n')
    return output / 'reference.json'


def load(path, task, horizon):
    """Load a reference, verifying task, horizon, block continuity and image checksums."""
    path = Path(path).resolve()
    reference = json.loads(path.read_text())
    if reference['schema'] != SCHEMA or reference['task'] != task or reference['source'].get('official_result') != 'success':
        raise ValueError('Reference schema/task mismatch')
    if reference['horizon'] != horizon:
        raise ValueError(f"Reference horizon {reference['horizon']} differs from policy horizon {horizon}")
    previous = 0
    for block in reference['blocks']:
        if block['start'] != previous or block['end'] != block['start'] + len(block['actions']) or not block['actions']:
            raise ValueError('TRAIN blocks must be contiguous, non-overlapping and nonempty')
        previous = block['end']
    if previous != len(reference['waypoints']) - 1:
        raise ValueError('TRAIN blocks must cover every retained waypoint transition')
    for waypoint in reference['waypoints']:
        for spec in waypoint['images'].values():
            image = (path.parent / spec['path']).resolve()
            if not image.is_relative_to(path.parent) or sha256(image) != spec['sha256']:
                raise ValueError(f'Reference image missing or changed: {image}')
            spec['bytes'] = image.read_bytes()
    return reference
