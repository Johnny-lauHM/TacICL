"""System 2 dialogue for System 1 control: each act call = note + grip mode + motion rows.

Same RoboICL grammar as dialogue.py (observation -> act -> receipt -> observation);
only the act arguments change: rows carry motion only and the finger is owned by
System 1 through one grip mode per call.
"""
import json
import math

from . import intents
from .dialogue import ActionError, observation_message
from .observation import grid, rgb_image, robot_state
from .system1 import CONTACT_RANGE_MM, MODES, GripError, validate_grip

TRAIN_NOTE = 'Recorded demonstration; no rationale was recorded.'

GRIP_TEXT = {
    'open': 'open: open the fingers fully and keep them open (value ignored, use 0).',
    'position': 'position: drive each finger to the absolute opening value in mm (0 closed, {fmax} fully open) and hold it.',
    'contact': ('contact: close until BOTH gel pads press at least about half of value (mm of gel indentation, '
                '{cmin}..{cmax}), then keep the weaker pad near value while the arm moves; System 1 adjusts the '
                'finger every control step and stops the call if contact is lost. A small value is a light grip '
                'that lets an object pivot between the pads; a large value squeezes.'),
}


def system_prompt(horizon, limits, tactile, control):
    if tactile:
        layout = ('Each observation image is a grid: HEAD camera (top-left), WRIST camera (top-right), '
                  'LEFT GEL and RIGHT GEL (bottom) = GelSight optical tactile images with markers from the two '
                  'finger pads. press_depth_max_mm is the measured maximum gel indentation of each pad '
                  '(0 = no contact); it is not calibrated force. ')
    else:
        layout = 'Each observation image shows the HEAD camera (left) and WRIST camera (right). '
    modes = ' '.join(GRIP_TEXT[m].format(fmax=limits['finger_max_mm'], cmin=CONTACT_RANGE_MM[0],
                                         cmax=CONTACT_RANGE_MM[1]) for m in MODES[control])
    return (
        'Control a Franka Panda arm with a parallel gripper from RGB, the task instruction and proprioception'
        + (' and fingertip tactile sensing' if tactile else '') + ', using exactly one act call per turn. '
        'You are the slow planner (System 2). A fast controller (System 1) owns the fingers and executes your '
        'grip mode every control step. In each act call write execution_note first: one or two sentences on '
        'what the latest observation shows and why the grip and rows follow from it. Simulator object poses, '
        'rewards, success labels and task code are not available.\n\n'
        + layout +
        'ee_pos_mm and ee_quat_wxyz are the panda_hand pose in the robot base frame. finger_mm is the measured '
        'opening of each finger (0 closed, 39 fully open).\n\n'
        'grip = {"mode", "value"} for the whole call. Modes: ' + modes + ' The grip is applied first (it counts '
        'as one action when it changes anything), then the rows run with System 1 holding the grip.\n\n'
        'Each action row has 6 numbers [dx, dy, dz, rx, ry, rz]. dx,dy,dz: translation in millimetres in the '
        'base frame. rx,ry,rz: base-frame rotation vector in degrees, not Euler angles. The first row starts '
        'from the latest observed EE pose; later rows accumulate. Per row, translation norm <= '
        f'{limits["translation_mm"]} mm and rotation norm <= {limits["rotation_deg"]} deg. Give 0..{horizon} rows '
        'per act; 0 rows applies only the grip. Each row runs as one planned motion and one action; execution '
        'stops early on task success, a safety stop, a planning failure (which ends the episode) or a System 1 '
        'event. The receipt reports executed rows, the grip outcome, System 1 events and the measured result '
        'state. Continue until the environment ends the episode.\n\n'
        'TRAIN is a separate recorded trajectory of the same task that the official evaluator judged '
        'successful, re-expressed in this same act format; its final observation shows what task completion '
        'looks like. Learn its progression, grip modes and action scale, then adapt to the current observation; '
        'object size and pose may differ. TRAIN and retained LIVE chunks show a starting observation, an act '
        'call, a receipt and the ending observation. <TRAJECTORY_GAP> marks omitted history, not a continuous '
        'transition. Use the latest LIVE observation for current geometry.')


def act_tool(horizon, control, finger_max_mm):
    value_doc = {'s1_tactile': f'position: opening mm 0..{finger_max_mm}; contact: press depth mm '
                               f'{CONTACT_RANGE_MM[0]}..{CONTACT_RANGE_MM[1]}; open: 0',
                 's1_position': f'position: opening mm 0..{finger_max_mm}; open: 0'}[control]
    return [{'type': 'function', 'function': {
        'name': 'act', 'description': 'Submit one grip mode and 0..H motion rows; receive the executed range and the resulting observation.',
        'parameters': {'type': 'object', 'additionalProperties': False,
                       'required': ['execution_note', 'grip', 'actions'],
                       'properties': {
                           'execution_note': {'type': 'string', 'maxLength': 300,
                                              'description': 'Written first: what the latest observation shows and why this grip and these rows.'},
                           'grip': {'type': 'object', 'additionalProperties': False, 'required': ['mode', 'value'],
                                    'properties': {'mode': {'type': 'string', 'enum': list(MODES[control])},
                                                   'value': {'type': 'number', 'description': value_doc}}},
                           'actions': {'type': 'array', 'minItems': 0, 'maxItems': horizon,
                                       'items': {'type': 'array', 'minItems': 6, 'maxItems': 6, 'items': {'type': 'number'}}}}}}}]


def validate_rows(rows, horizon, limits):
    if not isinstance(rows, list) or len(rows) > horizon:
        raise ActionError(f'actions must be a list of 0..{horizon} rows')
    for index, row in enumerate(rows):
        if (not isinstance(row, list) or len(row) != 6
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row)):
            raise ActionError(f'row {index} must contain 6 finite numbers')
        if math.hypot(*row[:3]) > limits['translation_mm'] + 1e-9:
            raise ActionError(f'row {index} translation norm {math.hypot(*row[:3]):.2f} mm exceeds {limits["translation_mm"]} mm')
        if math.hypot(*row[3:6]) > limits['rotation_deg'] + 1e-9:
            raise ActionError(f'row {index} rotation norm {math.hypot(*row[3:6]):.2f} deg exceeds {limits["rotation_deg"]} deg')
    return [[float(v) for v in row] for row in rows]


def parse_call(message, horizon, limits, control):
    calls = message.get('tool_calls') or []
    if len(calls) != 1 or calls[0]['function']['name'] != 'act':
        raise ActionError('Return exactly one act tool call')
    try:
        arguments = json.loads(calls[0]['function']['arguments'])
    except json.JSONDecodeError as error:
        raise ActionError(f'act arguments are not valid JSON: {error}') from None
    if not isinstance(arguments, dict) or set(arguments) != {'execution_note', 'grip', 'actions'}:
        raise ActionError('act arguments must contain exactly execution_note, grip and actions')
    try:
        grip = validate_grip(arguments['grip'], control, limits['finger_max_mm'])
    except GripError as error:
        raise ActionError(str(error)) from None
    return calls[0], grip, validate_rows(arguments['actions'], horizon, limits), str(arguments['execution_note'])


def call_messages(call_id, grip, rows, note, receipt):
    arguments = json.dumps({'execution_note': note, 'grip': grip, 'actions': rows},
                           ensure_ascii=False, separators=(',', ':'))
    return [{'role': 'assistant', 'content': None,
             'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': 'act', 'arguments': arguments}}]},
            {'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(receipt, ensure_ascii=False)}]


def train_messages(reference, tactile, control, horizon):
    """Fixed TRAIN prefix in the System 2 act format; identical across turns for KV reuse."""
    waypoints = reference['waypoints']

    def observe(index):
        w = waypoints[index]
        images = {name: rgb_image(spec['bytes']) for name, spec in w['images'].items()}
        image = grid(images['head'], images['wrist'], images['left'] if tactile else None, images['right'] if tactile else None)
        state = robot_state(w['ee'], w['finger_m'], w['press_depth_mm'] if tactile else None)
        return state, observation_message('TRAIN', index, state, image, end_of_example=index == len(waypoints) - 1)

    messages = [observe(0)[1]]
    for number, call in enumerate(intents.segment(reference, control, horizon)):
        result_state, result_message = observe(call['end'])
        receipt = {'source': 'TRAIN', 'executed_step_interval': [call['start'], call['end']],
                   'requested_rows': len(call['actions']), 'executed_rows': len(call['actions']),
                   'stop_reason': None, 'system1': {'grip_outcome': None, 'events': []},
                   'result_state': result_state}
        messages += call_messages(f'train_{number}', call['grip'], call['actions'], TRAIN_NOTE, receipt)
        messages.append(result_message)
    messages.append({'role': 'user', 'content': [{'type': 'text', 'text':
        'END OF FIXED TASK REFERENCE. Subsequent observations belong to the active episode.'}]})
    return messages
