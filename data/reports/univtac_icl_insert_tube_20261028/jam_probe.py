"""Diagnostic only (never a policy under evaluation): calibrate a tactile jam signal for insert_tube.

Runs in official eval mode after pre_move with force=False moves, like the eval-mode expert:
  jam:     push along the gripper axis in 2 mm steps from the pre_move pose (hidden 1-4 mm offset)
  realign: privileged re-placement at the hole axis (official expert move, task.hole_pose)
  insert:  push along the gripper axis in 2 mm steps
Every control step logs public tactile features (marker shear, press depth) next to privileged
labels (in-hand slip, tube pose relative to the hole). Privileged values are labels for threshold
calibration only; they decide phase ends here, never in a policy.
"""
import json
from pathlib import Path

import numpy as np

STEP_M = 0.002
JAM_SLIP_M, INSERT_SLIP_M = 0.010, 0.025
JAM_STEPS, INSERT_STEPS = 15, 25


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
        log = (self.out / 'steps.jsonl').open('a')
        state = {'phase': 'start', 'base': None}

        def features():
            obs = task._tactile_manager.get_observations(['marker'])
            pads = {}
            for name in sorted(obs):
                m = _np(obs[name]['marker'])
                d = m[1] - m[0]
                pads[name] = d
            return pads

        def record():
            pads = features()
            if state['base'] is None:
                state['base'] = {k: v.copy() for k, v in pads.items()}
            row = {'step': int(task.step_count), 'phase': state['phase'],
                   'ee_mm': (_np(task._robot_manager.get_gripper_center_pose()[:3]) * 1000).round(2).tolist()}
            press = _np(task._tactile_manager.get_press_depth()).tolist()
            for (name, d), p in zip(sorted(pads.items()), press):
                rel = d - state['base'][name]
                row[name] = {'shear_px': d.mean(0).round(3).tolist(), 'shear_rel_px': rel.mean(0).round(3).tolist(),
                             'rel_mag_px': float(np.linalg.norm(rel, axis=1).mean().round(3)), 'press_mm': round(p, 3)}
            inhand = task.prism.get_pose().rebase(task._robot_manager.get_gripper_center_pose())
            rel_hole = task.prism.get_pose().rebase(task.hole_pose)
            row['priv'] = {'slip_mm': round(float(abs(task.origin_inhand_pose[2] - inhand[2])) * 1000, 2),
                           'tube_rel_hole_mm': (np.asarray(rel_hole.p) * 1000).round(2).tolist()}
            log.write(json.dumps(row) + '\n')
            return row

        original = task._step

        def step(is_save=True):
            original(is_save)
            record()
        task._step = step
        try:
            record()
            for phase, steps, slip_limit in (('jam', JAM_STEPS, JAM_SLIP_M), ('insert', INSERT_STEPS, INSERT_SLIP_M)):
                if phase == 'insert':
                    state['phase'] = 'realign'
                    dis = task.prism.get_pose().rebase(task.hole_pose)[2]
                    task.move(task.atom.place_actor(task.prism, target_pose=task.hole_pose, pre_dis=dis, dis=dis,
                                                    is_open=False), time_dilation_factor=0.5, delay=False, force=False)
                state['phase'] = phase
                for _ in range(steps):
                    task.move(task.atom.move_by_displacement(z=STEP_M, xyz_coord='local'), delay=False, force=False)
                    last = record()
                    if task.check_success() or last['priv']['slip_mm'] / 1000 > slip_limit:
                        break
            success = bool(task.check_success())
            task.eval_success = success
            (self.out / 'result.json').write_text(json.dumps({'check_success': success,
                                                             'early_stop': bool(task.check_early_stop()),
                                                             'metadata': {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                                                                          for k, v in task.metadata.items()}}, indent=1))
        finally:
            task._step = original
            log.close()
            task.take_action_cnt = task.cfg.step_lim
