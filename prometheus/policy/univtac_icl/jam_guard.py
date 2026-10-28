"""Jam guard: a System 1 reflex on the arm under direct control (insert_tube).

System 2 still sends open-loop rows [dx,dy,dz,rx,ry,rz,finger]. Each row is planned as before, but
executed one control step at a time while the guard watches GelSight marker shear (public signal).
When the axial shear on either pad reaches JAM_PX and has risen during the row, the row stops and the
call ends with the event ``jam``, before the tube slides far in the hand.

Features are mean marker displacements relative to the marker field at the first policy observation
(after the official pre_move grasp), in tactile-image pixels:
  axial_left_px / axial_right_px  gel x: load along the tube (mirror signs on the two pads)
  lateral_px                      gel y mean of both pads
  asymmetry_px                    |axial_right| - |axial_left|

Calibration (diagnostic jam_probe on development seeds 2000101-2000106, privileged labels only there):
the straight push enters about 15 mm and stalls; at JAM_PX the sign of lateral_px matched the sign of the
hidden hole-frame x offset 6/6 and the sign of asymmetry_px matched the opposite of the y offset 6/6
(only one seed had y > 0). The privileged expert fix was a 1-2.5 mm hand shift against the offset:
lateral_px > 0 -> shift base -x, asymmetry_px > 0 -> shift base +y. offset_hint reports that mapping.

Repeated-jam calibration (probe_multi on development seeds 2000302-2000315, 20 jams with known hand
residual, the grasp-time tube pose carried by the hand, in the hole frame): asymmetry_px predicts the hand y
residual at first and later jams alike, sign 20/20, hand_y_mm = -0.79 * asymmetry_px - 0.4 (residual std
0.7 mm first jams, 1.1 mm all). lateral_px matched the x residual sign only about half the time there
(pure axial pushes), although it matched the hidden x offset 28/28 at the first jam of the guard policy
runs. offset_estimate therefore gives a y shift in mm and only a low-confidence x direction.
"""
import math

import numpy as np

from .geometry import apply_row
from .system1 import MOTION_EPS_MM, ROTATION_EPS_DEG

JAM_PX = 6.0
RISE_PX = 1.0
PADS = ('left_tactile', 'right_tactile')


def marker_field(task):
    obs = task._tactile_manager.get_observations(['marker'])
    field = {}
    for pad in PADS:
        marker = obs[pad]['marker']
        marker = marker.detach().cpu().numpy() if hasattr(marker, 'detach') else np.asarray(marker)
        field[pad] = marker[1] - marker[0]
    return field


def features(field, baseline):
    left = (field['left_tactile'] - baseline['left_tactile']).mean(axis=0)
    right = (field['right_tactile'] - baseline['right_tactile']).mean(axis=0)
    return {'axial_left_px': round(float(left[0]), 2), 'axial_right_px': round(float(right[0]), 2),
            'lateral_px': round(float((left[1] + right[1]) / 2), 2),
            'asymmetry_px': round(float(abs(right[0]) - abs(left[0])), 2)}


def axial(f):
    return max(abs(f['axial_left_px']), abs(f['axial_right_px']))


def offset_hint(f):
    return {'shift_hand_x': '-' if f['lateral_px'] > 0 else '+',
            'shift_hand_y': '+' if f['asymmetry_px'] > 0 else '-',
            'typical_shift_mm': [1, 3]}


Y_MM_PER_PX = 0.8
Y_SHIFT_MAX_MM = 5.0


def offset_estimate(f):
    shift_y = max(-Y_SHIFT_MAX_MM, min(Y_SHIFT_MAX_MM, Y_MM_PER_PX * f['asymmetry_px']))
    return {'shift_hand_y_mm': round(shift_y * 2) / 2, 'y_confidence': 'high',
            'shift_hand_x': '-' if f['lateral_px'] > 0 else '+', 'x_confidence': 'low'}


def hold_arm(robot):
    """Stopping mid-trajectory leaves the last joint velocity target active and the arm keeps moving
    along the blocked push; hold the measured joint positions with zero velocity instead."""
    import torch
    position = robot.robot.data.joint_pos[0, robot._arm_ids].clone()
    robot.set_arm(position, torch.zeros_like(position), force=False)


def execute(task, controller, current_ee, rows, baseline, check):
    """Run direct-control rows under the guard; returns (executed, stop_reason, report)."""
    from envs.utils.transforms import Pose
    position, quat = np.asarray(current_ee[:3]), np.asarray(current_ee[3:])
    executed, stop, events = 0, None, []
    for row in rows:
        if task.take_action_cnt >= task.cfg.step_lim:
            stop = 'official_step_limit'
            break
        position, quat = apply_row(position, quat, row)
        moves = math.hypot(*row[:3]) > MOTION_EPS_MM or math.hypot(*row[3:6]) > ROTATION_EPS_DEG
        finger_law = controller._position_law(row[6] / 1000)
        start = axial(features(marker_field(task), baseline))

        def law(idx):
            target, done, _ = finger_law(idx)
            now = axial(features(marker_field(task), baseline))
            if now >= JAM_PX and now >= start + RISE_PX:
                return target, True, 'jam'
            return target, done, None

        task.take_action_cnt += 1
        with task._configured_decimation():
            arm = None
            if moves:
                arm = controller.robot.plan_arm(Pose(position.tolist(), quat.tolist()))
                if arm['status'] == 'Fail':
                    task.plan_success = False
                    stop = 'planning_failure'
                    break
            event = controller._dense(arm, law)
            if event is not None and arm is not None:
                hold_arm(controller.robot)
                task._step(True)
        executed += 1
        if event is not None:
            events.append({'row': executed - 1, 'event': event})
        stop = check()
        if stop:
            break
        if event in ('jam', 'grip_timeout'):
            stop = 's1_' + event
            break
    return executed, stop, {'events': events, 'shear': features(marker_field(task), baseline)}
