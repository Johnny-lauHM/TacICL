"""UniVTAC deploy policy: frozen VLM, optional one-shot TRAIN, anchored LIVE memory.

Loaded by the unmodified official scripts/eval_policy.py as policy.<name>.Policy.
Each eval() call is one model turn that predicts 1..H rows; each row is executed
as one official planned move. Episode-ending conditions that the official loop
does not know about (planning failure latch, model-call budget) are logged and
end the episode by exhausting the official action counter, which the official
evaluator records as a failure.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from . import dialogue, intents, jam_guard, reference as reference_io, system2
from .system1 import MODES as S1_MODES, Controller
from .geometry import apply_row
from .memory import LiveTrajectoryMemory
from .observation import live_observation

TRAIN_MODALITIES = ('none', 'vision', 'vision_tactile')
# guard: direct rows, executed under the System 1 jam guard (jam_guard.py).
CONTROLS = ('direct', 'guard') + tuple(S1_MODES)
MOTION_EPS_MM, ROTATION_EPS_DEG, FINGER_EPS_MM = 0.05, 0.01, 0.3


JAM_HINTS = ('none', 'direction', 'estimate')
JAM_HISTORY = 5


def guard_text(live_tactile, hint):
    text = ('A System 1 jam guard runs under your rows: every control step it reads the GelSight marker shear '
            f'and stops the current row when the axial shear on a finger pad reaches {jam_guard.JAM_PX:g} px and is '
            'still rising, which means the held object is blocked; the call then ends with stop_reason s1_jam and '
            'system1.events lists the jam. Rows that relieve the load are not stopped.')
    if live_tactile:
        text += (' system1.marker_shear_px is the mean marker displacement since the start of the episode, in '
                 'tactile image pixels: axial_left_px / axial_right_px along the held object (mirror signs on '
                 'the two pads), lateral_px across it, asymmetry_px = |axial_right| - |axial_left|.')
    if hint == 'direction':
        text += (' After a jam, system1.offset_hint gives the base-frame direction System 1 estimates the hand '
                 'should shift to re-centre on the hole axis (shift_hand_x, shift_hand_y: "+" or "-"); typical '
                 'shifts are 1-3 mm. Use it to re-centre before pushing on.')
    elif hint == 'estimate':
        text += (' After a jam, system1.offset_hint is System 1\'s tactile estimate of how to re-centre the hand on '
                 'the hole axis: shift_hand_y_mm is a calibrated base-frame y shift in mm (reliable, typically '
                 'within about 1 mm); shift_hand_x is only a low-confidence direction ("+"/"-") for base-frame x. '
                 'system1.jam_history lists the hand position (ee_x_mm, ee_y_mm) and the estimate at each jam so '
                 'far. Contact phase recipe: on a jam, back off about 1 mm along the hole axis, shift by '
                 'shift_hand_y_mm (do not add more), then push again along the hole axis in short rows. If the '
                 'y estimate flips sign between jams, the hole lies between those two hand positions: move to a '
                 'point between them instead of shifting the full amount again. Change x only by about 1 mm, '
                 'and only after the y estimate is small (|shift_hand_y_mm| <= 0.5). Use the images for the '
                 'coarse approach to the hole and for the hole axis direction; the tactile estimate is for the '
                 'last few millimetres.')
    return text


def validate_args(args):
    if args['control'] not in CONTROLS:
        raise ValueError(f'control must be one of {CONTROLS}')
    if args['system2'] not in ('qwen', 'replay'):
        raise ValueError('system2 must be qwen or replay')
    if args.get('jam_hint', 'none') not in JAM_HINTS or (args.get('jam_hint', 'none') != 'none' and args['control'] != 'guard'):
        raise ValueError(f'jam_hint must be one of {JAM_HINTS}; direction/estimate need control=guard')
    if args['system2'] == 'replay' and (args['control'] in ('direct', 'guard') or args['train_modality'] == 'none'):
        raise ValueError('replay (diagnostic) re-executes TRAIN intents through System 1; it needs a reference and System 1')
    if args['train_modality'] not in TRAIN_MODALITIES or type(args['live_tactile']) is not bool:
        raise ValueError('train_modality must be none/vision/vision_tactile and live_tactile a boolean')
    if args['train_modality'] == 'vision_tactile' and not args['live_tactile']:
        raise ValueError('Tactile TRAIN requires tactile LIVE observations')
    if (args['train_modality'] == 'none') != (args.get('reference') is None):
        raise ValueError('A reference is required exactly when train_modality is not none')
    for key in ('horizon', 'live_anchor_count', 'max_calls', 'max_tokens', 'timeout_s'):
        if type(args[key]) is not int or args[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if args['task_clarification'] is not None and not (isinstance(args['task_clarification'], str) and args['task_clarification']):
        raise ValueError('task_clarification must be null or a nonempty string')
    if args['reasoning_guidance'] is not None and not (isinstance(args['reasoning_guidance'], str) and args['reasoning_guidance']):
        raise ValueError('reasoning_guidance must be null or a nonempty string')
    if type(args['max_repairs']) is not int or args['max_repairs'] < 0 or type(args['enable_thinking']) is not bool:
        raise ValueError('max_repairs must be a nonnegative integer and enable_thinking a boolean')


class Policy:
    def __init__(self, args):
        validate_args(args)
        self.args = args
        self.horizon, self.limits = args['horizon'], args['limits']
        self.direct = args['control'] in ('direct', 'guard')
        self.audit = Path(args['audit_dir'])
        self.audit.mkdir(parents=True, exist_ok=True)
        self.train = []
        if args['train_modality'] != 'none':
            if reference_io.sha256(args['reference']) != args['reference_sha256']:
                raise ValueError('Reference file differs from the prepared checksum')
            bundle = reference_io.load(args['reference'], args['task_name'], self.horizon)
            for block in bundle['blocks']:
                dialogue.validate_rows(block['actions'], self.horizon, self.limits)
            tactile_train = args['train_modality'] == 'vision_tactile'
            # Guard receipts in TRAIN only for references built with jam_frames (v2 references keep plain receipts).
            guard = self._train_system1(tactile_train) if args['control'] == 'guard' and bundle.get('jam_frames') else None
            self.train = (dialogue.train_messages(bundle, tactile=tactile_train, system1=guard) if self.direct else
                          system2.train_messages(bundle, tactile_train, args['control'], self.horizon))
            if not self.direct:
                self.intents = intents.segment(bundle, args['control'], self.horizon)
        system = (dialogue.system_prompt(self.horizon, self.limits, args['live_tactile']) if self.direct else
                  system2.system_prompt(self.horizon, self.limits, args['live_tactile'], args['control']))
        if args['train_modality'] == 'vision' and args['live_tactile']:
            system += ' TRAIN observations omit the tactile panels and press depth; LIVE observations include them.'
        if args['control'] == 'guard':
            system += '\n\n' + guard_text(args['live_tactile'], args.get('jam_hint', 'none'))
        if args['reasoning_guidance']:
            system += '\n\n' + args['reasoning_guidance']
        self.system = {'role': 'system', 'content': system}
        self.tools = (dialogue.act_tool(self.horizon) if self.direct else
                      system2.act_tool(self.horizon, args['control'], self.limits['finger_max_mm']))
        self.client = dialogue.Client(args['base_url'], args['model'], args['timeout_s'], args['max_tokens'],
                                      args['enable_thinking'], self.audit)
        self.episode = -1

    def _train_system1(self, tactile):
        """Guard receipts for TRAIN blocks: a block ending at a reference jam frame is a jam stop, with the
        demonstration's own marker shear (and the hint System 1 would have given) at that waypoint."""
        hint, history = self.args.get('jam_hint', 'none'), []

        def render(number, block, waypoint):
            jam = bool(block.get('jam'))
            shear = waypoint.get('marker_shear_px')
            if jam and shear is None:
                raise ValueError('A TRAIN jam block needs marker_shear_px; rebuild the reference with jam_frames')
            s1 = {'events': [{'row': len(block['actions']) - 1, 'event': 'jam'}] if jam else []}
            if tactile and shear is not None:
                s1['marker_shear_px'] = shear
            if jam and hint == 'direction':
                s1['offset_hint'] = jam_guard.offset_hint(shear)
            elif jam and hint == 'estimate':
                estimate = jam_guard.offset_estimate(shear)
                s1['offset_hint'] = estimate
                ee = [round(v * 1000, 1) for v in waypoint['ee'][:2]]
                history.append({'call': number, 'ee_x_mm': ee[0], 'ee_y_mm': ee[1],
                                'shift_hand_y_mm': estimate['shift_hand_y_mm'], 'shift_hand_x': estimate['shift_hand_x']})
                s1['jam_history'] = list(history[-JAM_HISTORY:])
            return ('s1_jam' if jam else None), s1
        return render

    def reset(self):
        self.episode += 1
        self.memory = LiveTrajectoryMemory(self.args['live_anchor_count'], self.args['max_calls'] * self.horizon)
        self.observations, self.step, self.calls = {}, 0, 0
        self.marker_baseline = None
        self.jam_history = []

    def close(self):
        pass

    def _log(self, record):
        with (self.audit / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(episode=self.episode, **record), ensure_ascii=False, allow_nan=False) + '\n')

    def _end_episode(self, task, reason):
        """Stop the official loop without success; the reason is in events.jsonl."""
        self._log({'event': 'episode_end', 'reason': reason, 'official_action_count': task.take_action_cnt})
        task.take_action_cnt = task.cfg.step_lim

    def _observe(self, task, raw):
        current = live_observation(raw, self.args['live_tactile'])
        self.observations[self.step] = dialogue.observation_message('LIVE', self.step, current['state'], current['image'])
        return current

    def _execute(self, task, current, rows):
        from envs.utils.atom import Action
        from envs.utils.transforms import Pose
        position, quat = np.asarray(current['ee'][:3]), np.asarray(current['ee'][3:])
        executed, stop_reason, timings = 0, None, []
        for row in rows:
            if task.take_action_cnt >= task.cfg.step_lim:
                stop_reason = 'official_step_limit'
                break
            position, quat = apply_row(position, quat, row)
            finger_m = row[6] / 1000
            moves = math.hypot(*row[:3]) > MOTION_EPS_MM or math.hypot(*row[3:6]) > ROTATION_EPS_DEG
            # The official finger controller lags its target, so a repeated target is re-issued
            # until the measured opening reaches it (or an object blocks it, keeping the squeeze).
            grips = abs(finger_m - task._robot_manager.get_gripper_qpos()) * 1000 > FINGER_EPS_MM
            started = time.monotonic()
            ok = True
            if moves or grips:
                kind = 'all' if moves and grips else 'move' if moves else 'gripper'
                action = Action(kind, target_pose=Pose(position.tolist(), quat.tolist()) if moves else None,
                                target_gripper_pos=finger_m if grips else None)
                task.take_action_cnt += 1
                with task._configured_decimation():
                    ok = task.move([action], delay=False, force=False)
            timings.append(time.monotonic() - started)
            if not ok:
                stop_reason = 'planning_failure'
                break
            executed += 1
            if task.check_success():
                task.eval_success = True
                stop_reason = 'task_success'
                break
            if task.check_early_stop():
                stop_reason = 'official_early_stop'
                break
        return executed, stop_reason, timings

    @staticmethod
    def _official_check(task):
        if task.check_success():
            task.eval_success = True
            return 'task_success'
        if task.check_early_stop():
            return 'official_early_stop'
        return None

    def _request(self, current, instruction, request_id):
        """One model turn with at most max_repairs contract repairs; returns (call, rows, note) or None."""
        text = 'TASK:\n' + instruction
        if self.args['task_clarification']:
            text += '\n\nTASK CLARIFICATION (from the official success check):\n' + self.args['task_clarification']
        task_message = {'role': 'user', 'content': [{'type': 'text', 'text': text}]}
        live = self.memory.render(self.observations[0], self.step, self.observations.__getitem__)
        tail = []
        for attempt in range(self.args['max_repairs'] + 1):
            messages = [self.system, task_message] + self.train + live + tail
            message, meta = self.client.complete(messages, self.tools, f'{request_id}-a{attempt}')
            self._log({'event': 'response', 'request_id': request_id, 'attempt': attempt, **meta,
                       'reasoning_chars': len(message.get('reasoning_content') or '')})
            try:
                if self.direct:
                    call, rows, note = dialogue.parse_call(message, self.horizon, self.limits)
                    return call, None, rows, note
                return system2.parse_call(message, self.horizon, self.limits, self.args['control'])
            except dialogue.ActionError as error:
                self._log({'event': 'rejected', 'request_id': request_id, 'attempt': attempt, 'error': str(error)})
                calls = message.get('tool_calls') or []
                if len(calls) == 1:
                    tail += [{'role': 'assistant', 'content': None, 'tool_calls': calls},
                             {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': json.dumps(
                                 {'rejected': True, 'executed_rows': 0, 'error': str(error)})}]
                else:
                    tail += [{'role': 'assistant', 'content': message.get('content') or ''},
                             {'role': 'user', 'content': [{'type': 'text', 'text': f'Rejected, nothing executed: {error}'}]}]
        return None

    def eval(self, task, observation):
        if task.mode != 'eval' or task.in_pre_move:
            raise RuntimeError('Policy runs only in official eval mode after pre_move')
        if task.plan_success is not True:
            # Before the first call this is an official pre_move failure, not a policy outcome.
            return self._end_episode(task, 'planning_failure_latched' if self.calls else 'pre_move_planning_failure_before_policy')
        if self.calls >= self.args['max_calls']:
            return self._end_episode(task, 'model_call_budget_exhausted')
        current = self._observe(task, observation)
        if self.args['control'] == 'guard' and self.marker_baseline is None:
            # Reference marker field: the first policy observation, after the official pre_move grasp.
            self.marker_baseline = jam_guard.marker_field(task)
        request_id = f"{self.args['run_id']}-e{self.episode}-c{self.calls}"
        self.calls += 1
        if self.args['system2'] == 'replay':
            # Diagnostic only: System 1 executes the demonstration's intents open-loop in motion.
            if self.calls > len(self.intents):
                return self._end_episode(task, 'replay_exhausted')
            intent = self.intents[self.calls - 1]
            if self.args.get('replay_contact_mm') is not None and intent['grip']['mode'] == 'contact':
                intent = dict(intent, grip={'mode': 'contact', 'value': float(self.args['replay_contact_mm'])})
            parsed = ({'id': f'replay_{self.calls - 1}'}, intent['grip'], intent['actions'], 'replay of TRAIN intent')
        else:
            parsed = self._request(current, task.instruction, request_id)
        if parsed is None:
            self._log({'event': 'hold', 'request_id': request_id, 'reason': 'repairs_exhausted'})
            return
        call, grip, rows, note = parsed
        start = self.step
        report = None
        if self.args['control'] == 'guard':
            began = time.monotonic()
            executed, stop_reason, report = jam_guard.execute(
                task, Controller(task, 'guard'), current['ee'], rows, self.marker_baseline,
                lambda: self._official_check(task))
            timings = [time.monotonic() - began]
        elif self.direct:
            executed, stop_reason, timings = self._execute(task, current, rows)
        else:
            began = time.monotonic()
            executed, stop_reason, report = Controller(task, self.args['control']).execute(
                current['ee'], grip, rows, lambda: self._official_check(task))
            timings = [time.monotonic() - began]
        # A grip-only call still advances the step so the next observation gets its own index.
        self.step = start + max(executed, 0 if self.direct else 1)
        after = self._observe(task, task._get_observations())
        receipt = {'source': 'LIVE', 'executed_step_interval': [start, self.step], 'requested_rows': len(rows),
                   'executed_rows': executed, 'stop_reason': stop_reason, 'result_state': after['state']}
        if report is not None and self.args['control'] == 'guard':
            receipt['system1'] = {'events': report['events']}
            if self.args['live_tactile']:
                receipt['system1']['marker_shear_px'] = report['shear']
            hint = self.args.get('jam_hint', 'none')
            if hint != 'none' and any(e['event'] == 'jam' for e in report['events']):
                if hint == 'direction':
                    receipt['system1']['offset_hint'] = jam_guard.offset_hint(report['shear'])
                else:
                    estimate = jam_guard.offset_estimate(report['shear'])
                    receipt['system1']['offset_hint'] = estimate
                    ee = after['state']['ee_pos_mm']
                    self.jam_history.append({'call': self.calls - 1, 'ee_x_mm': ee[0], 'ee_y_mm': ee[1],
                                             'shift_hand_y_mm': estimate['shift_hand_y_mm'],
                                             'shift_hand_x': estimate['shift_hand_x']})
                    receipt['system1']['jam_history'] = self.jam_history[-JAM_HISTORY:]
        elif report is not None:
            receipt['system1'] = {'grip_outcome': report['grip_outcome'], 'events': report['events']}
            if self.args['live_tactile']:
                receipt['system1']['max_press_mm'] = round(report['max_press_mm'], 3)
        summary = {'requested_rows': len(rows), 'executed_rows': executed, 'stop_reason': stop_reason,
                   'execution_note': note}
        messages = (dialogue.call_messages(call['id'], rows, note, receipt) if self.direct else
                    system2.call_messages(call['id'], grip, rows, note, receipt))
        self.memory.capture(start, self.step, messages, summary)
        self._log({'event': 'executed', 'request_id': request_id, 'grip': grip, 'rows': rows, 'note': note,
                   'before': current['state'], 'receipt': receipt, 'row_wall_s': timings,
                   'system1_report': report, 'sim_step': after['step'], 'memory': self.memory.stats()})
        if stop_reason == 'planning_failure':
            self._end_episode(task, 'planning_failure')
