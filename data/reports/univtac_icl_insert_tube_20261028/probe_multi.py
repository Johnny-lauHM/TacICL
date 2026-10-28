"""Diagnostic only (never a policy under evaluation): calibrate tactile direction hints at repeated jams.

The jam guard hint is reliable at the first jam; at later jams (after the hand already moved sideways) it
is much weaker. This probe reproduces later jams on development seeds with known residual offsets:

  loop (up to MAX_JAMS):
    push along the gripper axis in 1 mm steps until the public guard rule fires
      (max |axial shear| >= JAM_PX and risen >= RISE_PX since the push began, episode baseline)
    log public features under two baselines next to privileged labels: tube pose in the hole frame and
    the hand residual (grasp-time tube pose carried by the current hand, in the hole frame):
      episode  marker field at the first observation (what guard v1 uses)
      push     field when this push began (after the correction; retracted_before says whether the
               correction backed off RETRACT_MM along the gripper axis first)
    correction: optional retract, then a hole-frame lateral move of -factor * hand residual per axis with
      factor drawn from FACTORS (under-, over- and wrong-direction corrections), so the next jam starts
      from a varied residual

Privileged values (tube pose relative to the hole) choose corrections and serve as labels only.
"""
import json
from pathlib import Path

import numpy as np

JAM_PX, RISE_PX = 6.0, 1.0
STEP_M = 0.001
MAX_JAMS, MAX_PUSH = 5, 45
RETRACT_MM = 2.0
FACTORS = (-0.6, 0.4, 0.8, 1.0, 1.6)
SLIP_LIMIT_M = 0.025


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


class Policy:
    def __init__(self, args):
        self.out = Path(args['audit_dir'])
        self.out.mkdir(parents=True, exist_ok=True)

    def reset(self):
        self.done = False

    def close(self):
        pass

    def eval(self, task, observation):
        if self.done:
            task.take_action_cnt = task.cfg.step_lim
            return
        self.done = True
        rng = np.random.default_rng(int(abs(sum(task.metadata['random_noise'])) * 1e6))
        jams = (self.out / 'jams.jsonl').open('a')

        def field():
            obs = task._tactile_manager.get_observations(['marker'])
            return {p: _np(obs[p]['marker'])[1] - _np(obs[p]['marker'])[0] for p in ('left_tactile', 'right_tactile')}

        def feats(f, base):
            left = (f['left_tactile'] - base['left_tactile']).mean(0)
            right = (f['right_tactile'] - base['right_tactile']).mean(0)
            return {'axial_left_px': round(float(left[0]), 3), 'axial_right_px': round(float(right[0]), 3),
                    'lateral_left_px': round(float(left[1]), 3), 'lateral_right_px': round(float(right[1]), 3),
                    'lateral_px': round(float((left[1] + right[1]) / 2), 3),
                    'asymmetry_px': round(float(abs(right[0]) - abs(left[0])), 3)}

        def axial(f):
            return max(abs(f['axial_left_px']), abs(f['axial_right_px']))

        def priv():
            rel = task.prism.get_pose().rebase(task.hole_pose)
            inhand = task.prism.get_pose().rebase(task._robot_manager.get_gripper_center_pose())
            axis = rel.to_transformation_matrix()[:3, 2]
            # where the tube would be if it had stayed at its grasp pose in the hand: the hand's own offset
            # from the hole axis (the tube itself is centred by the chamfer once it is in the hole)
            hand = task.origin_inhand_pose.rebase(task.hole_pose, from_coord=task._robot_manager.get_gripper_center_pose())
            return {'res_mm': (np.asarray(rel.p) * 1000).round(2).tolist(),
                    'hand_res_mm': (np.asarray(hand.p) * 1000).round(2).tolist(),
                    'tilt_deg': round(float(np.degrees(np.arccos(np.clip(axis[2], -1, 1)))), 2),
                    'slip_mm': round(float(abs(task.origin_inhand_pose[2] - inhand[2])) * 1000, 2)}

        def move(actions):
            task.move(actions, delay=False, force=False)

        base0 = field()
        retracted = False
        try:
            for k in range(MAX_JAMS):
                base_push = field()
                start = axial(feats(base_push, base0))
                jammed = False
                for _ in range(MAX_PUSH):
                    move(task.atom.move_by_displacement(z=STEP_M, xyz_coord='local'))
                    f = field()
                    now = axial(feats(f, base0))
                    p = priv()
                    if task.check_success() or p['slip_mm'] / 1000 > SLIP_LIMIT_M:
                        break
                    if now >= JAM_PX and now >= start + RISE_PX:
                        jammed = True
                        break
                if not jammed:
                    break
                ee = (_np(task._robot_manager.get_gripper_center_pose()[:3]) * 1000).round(2).tolist()
                retract = bool(rng.integers(2))
                factors = [float(rng.choice(FACTORS)), float(rng.choice(FACTORS))]
                row = {'jam': k, 'priv': p, 'ee_mm': ee, 'retract_next': retract, 'factors_next': factors,
                       'retracted_before': retracted, 'episode': feats(f, base0), 'push': feats(f, base_push)}
                jams.write(json.dumps(row) + '\n')
                jams.flush()
                # correction for the next jam
                if retract:
                    move(task.atom.move_by_displacement(z=-RETRACT_MM / 1000, xyz_coord='local'))
                res = np.asarray(p['hand_res_mm'][:2]) / 1000
                shift = -res * np.asarray(factors)
                move(task.atom.move_by_displacement(x=float(shift[0]), y=float(shift[1]), xyz_coord=task.hole_pose))
                retracted = retract
            success = bool(task.check_success())
            task.eval_success = success
            (self.out / 'result.json').write_text(json.dumps({
                'check_success': success, 'early_stop': bool(task.check_early_stop()), 'final': priv(),
                'metadata': {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v)) for k, v in task.metadata.items()}}, indent=1))
        finally:
            jams.close()
            task.take_action_cnt = task.cfg.step_lim
