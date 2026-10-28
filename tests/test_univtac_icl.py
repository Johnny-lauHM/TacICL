import json
from pathlib import Path

import numpy as np
import pytest

from prometheus.policy.univtac_icl import dialogue, jam_guard, memory, observation, policy
from prometheus.policy.univtac_icl.geometry import apply_row, world_delta

LIMITS = {'translation_mm': 20, 'rotation_deg': 15, 'finger_max_mm': 39}
REFERENCE = Path(__file__).resolve().parents[1] / 'data/datasets/univtac_icl/lift_can_ep12_h4_v2/reference.json'


def test_validate_rows_accepts_bounded_rows_and_rejects_violations():
    assert dialogue.validate_rows([[1, 2, 3, 0, 0, 5, 20]], 4, LIMITS) == [[1.0, 2.0, 3.0, 0.0, 0.0, 5.0, 20.0]]
    for rows in ([], [[0] * 7] * 5, [[0] * 6], [[30, 0, 0, 0, 0, 0, 10]], [[0, 0, 0, 20, 0, 0, 10]],
                 [[0, 0, 0, 0, 0, 0, 40]], [[0, 0, 0, 0, 0, 0, True]], [[float('nan')] + [0] * 6]):
        with pytest.raises(dialogue.ActionError):
            dialogue.validate_rows(rows, 4, LIMITS)


def test_parse_call_requires_exactly_one_act_with_exact_fields():
    call = {'id': 'c', 'function': {'name': 'act', 'arguments': json.dumps({'actions': [[0] * 7], 'execution_note': 'x'})}}
    assert dialogue.parse_call({'tool_calls': [call]}, 4, LIMITS)[1] == [[0.0] * 7]
    with pytest.raises(dialogue.ActionError):
        dialogue.parse_call({'tool_calls': []}, 4, LIMITS)
    bad = {'id': 'c', 'function': {'name': 'act', 'arguments': json.dumps({'actions': [[0] * 7]})}}
    with pytest.raises(dialogue.ActionError):
        dialogue.parse_call({'tool_calls': [bad]}, 4, LIMITS)


def test_rows_round_trip_world_delta():
    q0 = np.array([0.0, 0.986, 0.0, 0.167]) / np.linalg.norm([0.0, 0.986, 0.0, 0.167])
    p1, q1 = apply_row([0.5, 0.0, 0.2], q0, [10, -5, 2, 3, -4, 5, 0])
    assert np.allclose((p1 - [0.5, 0.0, 0.2]) * 1000, [10, -5, 2])
    assert np.allclose(np.degrees(world_delta(q0, q1)), [3, -4, 5], atol=1e-6)


def _obs(step):
    return {'role': 'user', 'content': [{'type': 'text', 'text': f'obs {step}'}]}


def test_memory_keeps_anchors_and_latest_and_gaps_the_rest():
    live = memory.LiveTrajectoryMemory(count=3, maximum=40)  # anchors 0, 13; slot 3 = latest
    for start in range(0, 40, 4):
        live.capture(start, start + 4, [{'role': 'assistant', 'content': f'chunk {start}'}], {'note': start})
    texts = [m['content'] if isinstance(m['content'], str) else m['content'][0]['text']
             for m in live.render(_obs(0), 40, _obs)]
    assert texts[:3] == ['obs 0', 'chunk 0', 'obs 4']
    assert texts[3].startswith('<TRAJECTORY_GAP>') and '"omitted_action_interval":[4,12]' in texts[3]
    assert texts[4:7] == ['obs 12', 'chunk 12', 'obs 16']
    assert '"omitted_action_interval":[16,36]' in texts[7]
    assert texts[8:] == ['obs 36', 'chunk 36', 'obs 40']
    with pytest.raises(ValueError):
        live.capture(38, 42, [], None)
    assert live.capture(40, 40, [], None) is False


def test_policy_condition_validation():
    base = dict(train_modality='none', live_tactile=False, reference=None, horizon=4, live_anchor_count=6,
                max_calls=25, max_tokens=4096, timeout_s=60, max_repairs=2, enable_thinking=True,
                task_clarification=None, reasoning_guidance=None,
                control='direct', system2='qwen')
    policy.validate_args(base)
    for change in ({'train_modality': 'vision_tactile', 'reference': 'x'}, {'train_modality': 'vision'},
                   {'reference': 'x'}, {'live_tactile': 1}, {'max_repairs': -1}, {'task_clarification': ''}, {'reasoning_guidance': ''},
                   {'control': 'x'}, {'system2': 'replay'}, {'jam_hint': True}, {'jam_hint': 'estimate'}):
        with pytest.raises(ValueError):
            policy.validate_args({**base, **change})
    for hint in policy.JAM_HINTS:
        policy.validate_args({**base, 'control': 'guard', 'jam_hint': hint})


def test_grid_layout_depends_on_tactile():
    image = observation.rgb_image(np.zeros((270, 480, 3), np.uint8))
    assert observation.grid(image, image).size == (960, 270)
    assert observation.grid(image, image, image, image).size == (960, 540)
    with pytest.raises(ValueError):
        observation.grid(image, image, image)


@pytest.mark.skipif(not REFERENCE.exists(), reason='lift_can reference bundle not built')
def test_reference_train_messages_follow_shared_grammar():
    from prometheus.policy.univtac_icl import reference
    bundle = reference.load(REFERENCE, 'lift_can', 4)
    for block in bundle['blocks']:
        dialogue.validate_rows(block['actions'], 4, LIMITS)
    for tactile in (False, True):
        messages = dialogue.train_messages(bundle, tactile)
        roles = [m['role'] for m in messages]
        assert roles[0] == 'user' and roles[-1] == 'user'
        assert roles[1:-1] == ['assistant', 'tool', 'user'] * len(bundle['blocks'])
        states = [json.loads(m['content'][0]['text'])['robot_state'] for m in messages[:-1] if m['role'] == 'user']
        assert all(('press_depth_max_mm' in s) == tactile for s in states)
    with pytest.raises(ValueError):
        reference.load(REFERENCE, 'lift_can', 5)


# ---- System 2 / System 1 -------------------------------------------------------------

from prometheus.policy.univtac_icl import intents, system1, system2


def test_grip_validation_depends_on_control():
    assert system1.validate_grip({'mode': 'contact', 'value': 0.2}, 's1_tactile', 39) == {'mode': 'contact', 'value': 0.2}
    for grip, control in (({'mode': 'contact', 'value': 0.2}, 's1_position'), ({'mode': 'contact', 'value': 5}, 's1_tactile'),
                          ({'mode': 'position', 'value': 40}, 's1_tactile'), ({'mode': 'open'}, 's1_tactile'),
                          ({'mode': 'squeeze', 'value': 1}, 's1_tactile'), ({'mode': 'position', 'value': True}, 's1_position')):
        with pytest.raises(system1.GripError):
            system1.validate_grip(grip, control, 39)


def test_system2_parse_call_accepts_grip_only_and_rejects_seven_column_rows():
    def call(arguments):
        return {'tool_calls': [{'id': 'c', 'function': {'name': 'act', 'arguments': json.dumps(arguments)}}]}
    ok = call({'execution_note': 'x', 'grip': {'mode': 'contact', 'value': 0.2}, 'actions': []})
    assert system2.parse_call(ok, 8, LIMITS, 's1_tactile')[1:3] == ({'mode': 'contact', 'value': 0.2}, [])
    for bad in ({'execution_note': 'x', 'grip': {'mode': 'open', 'value': 0}, 'actions': [[0] * 7]},
                {'execution_note': 'x', 'actions': []},
                {'execution_note': 'x', 'grip': {'mode': 'open', 'value': 0}, 'actions': [[0] * 6] * 9}):
        with pytest.raises(dialogue.ActionError):
            system2.parse_call(call(bad), 8, LIMITS, 's1_tactile')


REFERENCE_H8 = REFERENCE.parents[1] / 'lift_can_ep12_h8_v2/reference.json'


@pytest.mark.skipif(not REFERENCE_H8.exists(), reason='lift_can h8 reference bundle not built')
def test_train_intents_cover_the_demonstration_in_order():
    from prometheus.policy.univtac_icl import reference
    bundle = reference.load(REFERENCE_H8, 'lift_can', 8)
    for control, mode in (('s1_tactile', 'contact'), ('s1_position', 'position')):
        calls = intents.segment(bundle, control, 8)
        assert [c['grip']['mode'] for c in calls] == [mode] * (len(calls) - 1) + ['open']
        assert calls[0]['start'] == 0 and calls[-1]['end'] == len(bundle['waypoints']) - 1
        assert all(a['end'] == b['start'] for a, b in zip(calls, calls[1:]))
        for c in calls:
            system1.validate_grip(c['grip'], control, 39)
            system2.validate_rows(c['actions'], 8, LIMITS)
        messages = system2.train_messages(bundle, control == 's1_tactile', control, 8)
        roles = [m['role'] for m in messages]
        assert roles[1:-1] == ['assistant', 'tool', 'user'] * len(calls)
    assert intents.segment(bundle, 's1_tactile', 8)[0]['grip']['value'] == pytest.approx(0.16, abs=0.05)


class _FakeTactile:
    def __init__(self, gripper):
        self.gripper = gripper

    def get_press_depth(self):
        import torch
        # Contact begins below 25 mm opening; 1 mm of closing adds 1 mm of press on each pad.
        depth = max(0.0, 25.0 - self.gripper.qpos * 1000)
        return torch.tensor([depth, depth])


class _FakeRobot:
    gripper_max_qpos, device = 0.039, 'cpu'

    def __init__(self):
        self.qpos = 0.024

    def get_gripper_qpos(self):
        return self.qpos

    def set_gripper(self, pos, vel=None, force=False):
        self.qpos = float(pos)


class _FakeTask:
    def __init__(self):
        self._robot_manager = _FakeRobot()
        self._tactile_manager = _FakeTactile(self._robot_manager)
        self.steps = 0

    def _step(self, is_save=True):
        self.steps += 1


def test_contact_law_holds_the_weaker_pad_in_band_and_backs_off_at_the_safety_limit():
    task = _FakeTask()
    controller = system1.Controller.__new__(system1.Controller)
    controller.task, controller.control, controller.robot, controller.max_qpos = task, 's1_tactile', task._robot_manager, 0.039
    report = {'max_press_mm': 0.0}
    law = controller._contact_law(0.3, report)
    for idx in range(200):
        target, _, event = law(idx)
        task._robot_manager.set_gripper(target)
        assert event is None
    press = float(task._tactile_manager.get_press_depth()[0])
    assert 0.15 <= press <= 0.6
    task._robot_manager.qpos = 0.018                     # 7 mm press > safety limit
    assert law(0)[2] == 'press_limit'


def test_contact_law_reports_contact_lost_after_consecutive_empty_steps():
    task = _FakeTask()
    controller = system1.Controller.__new__(system1.Controller)
    controller.task, controller.control, controller.robot, controller.max_qpos = task, 's1_tactile', task._robot_manager, 0.039
    task._robot_manager.qpos = 0.0247                    # 0.3 mm press: in contact
    law = controller._contact_law(0.3, {'max_press_mm': 0.0})
    assert law(0)[2] is None
    events = []
    for idx in range(system1.LOST_STEPS):
        task._robot_manager.qpos = 0.030                 # object gone: no press however the law closes
        events.append(law(idx)[2])
    assert events[-1] == 'contact_lost' and events[:-1] == [None] * (system1.LOST_STEPS - 1)


def test_contact_law_never_closes_past_the_squeeze_floor():
    task = _FakeTask()
    controller = system1.Controller.__new__(system1.Controller)
    controller.task, controller.control, controller.robot, controller.max_qpos = task, 's1_tactile', task._robot_manager, 0.039
    task._robot_manager.qpos = 0.030                     # no contact anywhere below
    law = controller._contact_law(0.3, {'max_press_mm': 0.0})
    for idx in range(500):
        target = law(idx)[0]
    assert target >= 0.030 - system1.MAX_SQUEEZE_M - 1e-9


def _controller(task):
    controller = system1.Controller.__new__(system1.Controller)
    controller.task, controller.control, controller.robot, controller.max_qpos = task, 's1_tactile', task._robot_manager, 0.039
    return controller


def test_close_to_contact_stops_at_half_target_or_reports_no_contact():
    task = _FakeTask()
    task._robot_manager.qpos = 0.039
    assert _controller(task)._close_to_contact(0.4) == 'contact_reached'
    assert 0.2 <= float(task._tactile_manager.get_press_depth()[0]) <= 0.5 + 1e-3   # at most one fast step past first touch
    empty = _FakeTask()
    empty._tactile_manager.get_press_depth = lambda: __import__('torch').tensor([0.0, 0.0])
    empty._robot_manager.qpos = 0.039
    assert _controller(empty)._close_to_contact(0.4) == 'no_contact'
    assert empty._robot_manager.qpos == 0.0


def _field(axial_left, axial_right, lateral):
    left = np.zeros((63, 2)) + [axial_left, lateral]
    right = np.zeros((63, 2)) + [axial_right, lateral]
    return {'left_tactile': left, 'right_tactile': right}


def test_jam_guard_features_and_offset_hint_follow_the_calibrated_signs():
    zero = _field(0, 0, 0)
    f = jam_guard.features(_field(-3, 7, 0.8), zero)
    assert (f['axial_left_px'], f['axial_right_px'], f['lateral_px'], f['asymmetry_px']) == (-3, 7, 0.8, 4)
    assert jam_guard.axial(f) == 7
    hint = jam_guard.offset_hint(f)
    assert (hint['shift_hand_x'], hint['shift_hand_y']) == ('-', '+')
    hint = jam_guard.offset_hint(jam_guard.features(_field(-7, 3, -1.0), zero))
    assert (hint['shift_hand_x'], hint['shift_hand_y']) == ('+', '-')


def test_jam_guard_offset_estimate_scales_y_and_clips():
    zero = _field(0, 0, 0)
    estimate = jam_guard.offset_estimate(jam_guard.features(_field(-3, 7, 0.8), zero))  # asymmetry +4 px
    assert estimate == {'shift_hand_y_mm': 3.0, 'y_confidence': 'high', 'shift_hand_x': '-', 'x_confidence': 'low'}
    assert jam_guard.offset_estimate(jam_guard.features(_field(-7, 3, -1.0), zero))['shift_hand_y_mm'] == -3.0
    assert jam_guard.offset_estimate(jam_guard.features(_field(0, 12, 0), zero))['shift_hand_y_mm'] == 5.0


INSERT_REFERENCE = REFERENCE.parents[1] / 'insert_tube_ep37_h8_v3/reference.json'


@pytest.mark.skipif(not INSERT_REFERENCE.exists(), reason='insert_tube v3 reference bundle not built')
def test_v3_reference_renders_the_demonstration_jam_as_a_guard_stop():
    from prometheus.policy.univtac_icl import reference
    bundle = reference.load(INSERT_REFERENCE, 'insert_tube', 8)
    assert [b['jam'] for b in bundle['blocks']] == [True, False, False]
    agent = policy.Policy.__new__(policy.Policy)
    agent.args = {'jam_hint': 'estimate'}
    messages = dialogue.train_messages(bundle, True, agent._train_system1(True))
    receipts = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
    jam = receipts[0]
    assert jam['stop_reason'] == 's1_jam' and jam['system1']['events'] == [{'row': 4, 'event': 'jam'}]
    assert jam['system1']['offset_hint']['shift_hand_y_mm'] == -2.0
    assert jam['system1']['jam_history'][0]['call'] == 0
    assert all(r['stop_reason'] is None and r['system1']['events'] == [] for r in receipts[1:])
    visual = [json.loads(m['content']) for m in dialogue.train_messages(bundle, False, agent._train_system1(False))
              if m['role'] == 'tool']
    assert 'marker_shear_px' not in visual[0]['system1'] and 'offset_hint' in visual[0]['system1']


class _GuardTask(_FakeTask):
    """Axial marker shear rises 0.5 px per control step while the arm pushes."""

    def __init__(self):
        super().__init__()
        import contextlib
        self.take_action_cnt, self.cfg = 0, type('cfg', (), {'step_lim': 300})()
        self._configured_decimation = contextlib.nullcontext
        robot = self._robot_manager
        robot.plan_arm = lambda pose, **kw: {'status': 'Success', 'num_steps': 40,
                                             'position': [None] * 40, 'velocity': [None] * 40}
        robot.set_arm = lambda *a, **kw: robot.arm_commands.append(a)
        robot.arm_commands = []
        import torch
        robot._arm_ids = [0, 1]
        robot.robot = type('art', (), {'data': type('data', (), {'joint_pos': torch.tensor([[0.1, 0.2, 0.0]])})()})()
        robot.device = 'cpu'
        task = self

        class Tactile(_FakeTactile):
            def get_observations(self, data_types):
                field = _field(-0.25 * task.steps, 0.5 * task.steps, 0.1 * task.steps)
                return {pad: {'marker': np.stack([np.zeros((63, 2)), field[pad]])} for pad in field}
        self._tactile_manager = Tactile(robot)


def test_jam_guard_stops_the_row_when_axial_shear_reaches_the_threshold(monkeypatch):
    import sys, types
    transforms = types.ModuleType('envs.utils.transforms')
    transforms.Pose = lambda p, q: (p, q)
    monkeypatch.setitem(sys.modules, 'envs', types.ModuleType('envs'))
    monkeypatch.setitem(sys.modules, 'envs.utils', types.ModuleType('envs.utils'))
    monkeypatch.setitem(sys.modules, 'envs.utils.transforms', transforms)
    task = _GuardTask()
    baseline = jam_guard.marker_field(task)
    controller = _controller(task)
    rows = [[2, 0, -4, 0, 0, 0, 24.0]] * 3
    executed, stop, report = jam_guard.execute(task, controller, [0.5, 0, 0.2, 1, 0, 0, 0], rows, baseline, lambda: None)
    assert stop == 's1_jam' and executed == 1 and report['events'] == [{'row': 0, 'event': 'jam'}]
    assert jam_guard.JAM_PX <= report['shear']['axial_right_px'] <= jam_guard.JAM_PX + 1   # + one hold step
    assert task.take_action_cnt == 1
    held_position, held_velocity = task._robot_manager.arm_commands[-1][:2]
    assert held_position.tolist() == pytest.approx([0.1, 0.2]) and held_velocity.tolist() == [0.0, 0.0]
