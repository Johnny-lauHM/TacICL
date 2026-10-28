"""System 1: a deterministic grip controller under the VLM (System 2).

System 2 picks one grip mode per act call plus up to H motion rows. System 1 runs
every simulator control step and owns the finger:

* ``open``          -- open fully, then hold.
* ``position``      -- drive to an absolute opening (mm per finger), then hold.
* ``contact``       -- (tactile control only) close until both gel pads press
  (fast until first touch, then fine), then keep the weaker pad's press inside a
  band around the target while the arm moves, and report contact loss.

It reads only public signals: measured finger joints and per-pad press depth.
Object poses, success checks and task code are never used. Motion planning and
action accounting stay as before: each motion row is one official planned move
and one official action; a grip transition without motion is one action.
"""
import math

import numpy as np
import torch

from .geometry import apply_row

MODES = {'s1_tactile': ('open', 'position', 'contact'), 's1_position': ('open', 'position')}
CONTACT_RANGE_MM = (0.05, 2.0)       # allowed press-depth targets for ``contact``
BAND = (0.5, 2.0)                    # hold band on the weaker pad = target * BAND
SAFE_PRESS_MM = 6.0                  # back off before the official 8 mm early stop
LOST_PRESS_MM = 0.02                 # a pad at or below this has no contact
MAX_SQUEEZE_M = 0.004                # never close more than this inside the first-contact opening
LOST_STEPS = 10                      # consecutive control steps before contact_lost
# The eval finger is a PD joint (stiffness 2e3 N/m): grip force comes from commanding the finger
# inside its blocked position, not from press depth alone. After contact the command is preloaded
# PRELOAD_M inside the measured opening (about 2 N), then moved by GAIN * press error per step.
PRELOAD_M = 0.001
GAIN = 0.5                           # metres of command per metre of press error, per step
MAX_STEP_M = 0.0005                  # official gripper planning step (0.5 mm)
SETTLE_STEPS, SETTLE_M, POSITION_TIMEOUT = 10, 0.00002, 400
FINE_STEP_M, LEAD_M, CLOSE_STEPS = 0.00005, 0.002, 600   # closing to contact
MOTION_EPS_MM, ROTATION_EPS_DEG = 0.05, 0.01
# Rows run at half planner speed, as the official expert moves while holding (gripper_rotate);
# the same for both System 1 controls.
TIME_DILATION = 0.5


class GripError(ValueError):
    pass


def validate_grip(grip, control, finger_max_mm):
    if not isinstance(grip, dict) or set(grip) != {'mode', 'value'}:
        raise GripError('grip must be an object with exactly mode and value')
    mode, value = grip['mode'], grip['value']
    if mode not in MODES[control]:
        raise GripError(f'grip.mode must be one of {list(MODES[control])}')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GripError('grip.value must be a finite number')
    if mode == 'position' and not 0 <= value <= finger_max_mm:
        raise GripError(f'position value must be an opening in [0, {finger_max_mm}] mm')
    if mode == 'contact' and not CONTACT_RANGE_MM[0] <= value <= CONTACT_RANGE_MM[1]:
        raise GripError(f'contact value must be a press depth in [{CONTACT_RANGE_MM[0]}, {CONTACT_RANGE_MM[1]}] mm')
    return {'mode': mode, 'value': float(value)}


class Controller:
    """Executes one act call; returns (executed_rows, stop_reason, report)."""

    def __init__(self, task, control):
        self.task, self.control = task, control
        self.robot = task._robot_manager
        self.max_qpos = float(self.robot.gripper_max_qpos)

    def press(self):
        return [float(v) for v in self.task._tactile_manager.get_press_depth().tolist()]

    def finger(self):
        return self.robot.get_gripper_qpos()

    # -- one dense control loop: arm plan (optional) + per-step finger law ---------------
    def _dense(self, arm_seq, finger_law):
        steps = arm_seq['num_steps'] if arm_seq is not None else 0
        idx = 0
        while True:
            if arm_seq is not None and idx < steps:
                self.robot.set_arm(arm_seq['position'][idx], arm_seq['velocity'][idx], force=False)
            target, done, event = finger_law(idx)
            if target is not None:
                self.robot.set_gripper(torch.tensor(target, dtype=torch.float32, device=self.robot.device),
                                       force=False)
            self.task._step(True)
            idx += 1
            if event is not None:
                return event
            if idx >= steps and done:
                return None
            if idx > steps + 1000:
                return 'grip_timeout'

    def _position_law(self, qpos_target):
        """Ramp at the official 0.5 mm/step, then hold until the finger reaches the target or is
        blocked (no motion for SETTLE_STEPS); the official finger lags its command."""
        start = self.finger()
        n = max(1, int(math.ceil(abs(qpos_target - start) / 0.0005)))
        ramp = np.linspace(start, qpos_target, n)
        state = {'last': start, 'still': 0}

        def law(idx):
            now = self.finger()
            state['still'] = state['still'] + 1 if abs(now - state['last']) < SETTLE_M else 0
            state['last'] = now
            settled = abs(now - qpos_target) < 0.0003 or state['still'] >= SETTLE_STEPS or idx >= POSITION_TIMEOUT
            return float(ramp[min(idx, n - 1)]), idx >= n - 1 and settled, None
        return law

    def _contact_law(self, target_mm, report):
        lo, hi = target_mm * BAND[0], target_mm * BAND[1]
        state = {'cmd': max(0.0, self.finger() - PRELOAD_M), 'had_contact': False, 'lost': 0,
                 'floor': max(0.0, self.finger() - MAX_SQUEEZE_M)}

        def law(idx):
            press = self.press()
            report['max_press_mm'] = max(report['max_press_mm'], max(press))
            if max(press) > SAFE_PRESS_MM:
                state['cmd'] = min(self.max_qpos, state['cmd'] + MAX_STEP_M)
                return state['cmd'], True, 'press_limit'
            if min(press) >= lo:
                state['had_contact'] = True
            # A grasp needs both pads: either pad empty for LOST_STEPS is a lost grasp.
            if state['had_contact'] and min(press) <= LOST_PRESS_MM:
                state['lost'] += 1
                if state['lost'] >= LOST_STEPS:
                    return state['cmd'], True, 'contact_lost'
            else:
                state['lost'] = 0
            # Regulate the weaker pad: contact is two-sided only while both pads press,
            # and the demonstration's pads are asymmetric (about 0.15 vs 0.45 mm).
            if min(press) < lo or min(press) > hi:
                step = max(-MAX_STEP_M, min(MAX_STEP_M, GAIN * (min(press) - target_mm) / 1000))
                state['cmd'] = min(self.max_qpos, max(state['floor'], state['cmd'] + step))
            return state['cmd'], True, None
        return law

    def _close_to_contact(self, target_mm):
        """Close fast (0.5 mm/step) until a pad touches, then finely (0.05 mm/step) until the weaker
        pad reaches half the target. The command never leads the lagging finger by more than
        LEAD_M, so first contact does not arrive with a hidden squeeze."""
        cmd, outcome = self.finger(), None
        still, last = 0, cmd
        for _ in range(CLOSE_STEPS):
            press = self.press()
            if max(press) > SAFE_PRESS_MM:
                outcome = 'press_limit'
                break
            if min(press) >= target_mm * BAND[0]:
                outcome = 'contact_reached'
                break
            now = self.finger()
            still = still + 1 if abs(now - last) < SETTLE_M else 0
            last = now
            if cmd <= 0.0 and still >= SETTLE_STEPS:
                break
            step = FINE_STEP_M if max(press) > LOST_PRESS_MM else MAX_STEP_M
            cmd = max(0.0, cmd - step, now - LEAD_M)
            self.robot.set_gripper(torch.tensor(cmd, dtype=torch.float32, device=self.robot.device), force=False)
            self.task._step(True)
        if outcome is None:
            press = self.press()
            outcome = 'one_sided_contact' if max(press) > LOST_PRESS_MM else 'no_contact'
        return outcome

    def execute(self, current_ee, grip, rows, check):
        """check() -> stop reason from the official success/early-stop checks, or None."""
        from envs.utils.transforms import Pose
        task = self.task
        report = {'grip': grip, 'grip_outcome': None, 'max_press_mm': 0.0, 'events': []}
        mode, value = grip['mode'], grip['value']
        position, quat = np.asarray(current_ee[:3]), np.asarray(current_ee[3:])

        # 1) Grip transition, counted as one official action when it changes anything.
        if mode in ('open', 'position'):
            target = self.max_qpos if mode == 'open' else value / 1000
            if abs(target - self.finger()) * 1000 > 0.3:
                task.take_action_cnt += 1
                with task._configured_decimation():
                    self._dense(None, self._position_law(target))
            report['grip_outcome'] = 'position_reached'
        else:
            press = self.press()
            if min(press) < value * BAND[0]:
                task.take_action_cnt += 1
                with task._configured_decimation():
                    report['grip_outcome'] = self._close_to_contact(value)
            else:
                report['grip_outcome'] = 'contact_already'
        stop = check()
        if stop or not rows:
            report['finger_mm'] = round(self.finger() * 1000, 2)
            return 0, stop, report

        # 2) Motion rows; the finger law runs every control step.
        executed = 0
        # One contact law per call: its squeeze floor and contact state span all rows of the call.
        contact_law = self._contact_law(value, report) if mode == 'contact' else None
        for row in rows:
            if task.take_action_cnt >= task.cfg.step_lim:
                return executed, 'official_step_limit', report
            position, quat = apply_row(position, quat, list(row) + [0.0])
            moves = math.hypot(*row[:3]) > MOTION_EPS_MM or math.hypot(*row[3:6]) > ROTATION_EPS_DEG
            law = contact_law or self._position_law(self.max_qpos if mode == 'open' else value / 1000)
            task.take_action_cnt += 1
            with task._configured_decimation():
                arm = None
                if moves:
                    arm = self.robot.plan_arm(Pose(position.tolist(), quat.tolist()), time_dilation_factor=TIME_DILATION)
                    if arm['status'] == 'Fail':
                        task.plan_success = False
                        report['finger_mm'] = round(self.finger() * 1000, 2)
                        return executed, 'planning_failure', report
                event = self._dense(arm, law)
            executed += 1
            if event is not None:
                report['events'].append({'row': executed - 1, 'event': event})
            stop = check()
            if stop:
                break
            if event in ('contact_lost', 'press_limit', 'grip_timeout'):
                stop = 's1_' + event
                break
        report['finger_mm'] = round(self.finger() * 1000, 2)
        return executed, stop, report
