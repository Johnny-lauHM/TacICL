"""RoboICL-style tool-call dialogue for UniVTAC: prompt, act tool, validation, transport.

TRAIN and LIVE share one grammar: observation -> act call -> receipt -> observation.
"""
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

from .observation import data_url, grid, rgb_image, robot_state


class ActionError(ValueError):
    """Model output that violates the act contract; reported back verbatim."""


def system_prompt(horizon, limits, tactile):
    if tactile:
        layout = ('Each observation image is a grid: HEAD camera (top-left), WRIST camera (top-right), '
                  'LEFT GEL and RIGHT GEL (bottom) = GelSight optical tactile images with markers from the two '
                  'finger pads. press_depth_max_mm is the measured maximum gel indentation of each pad '
                  '(0 = no contact); it is not calibrated force. ')
    else:
        layout = 'Each observation image shows the HEAD camera (left) and WRIST camera (right). '
    return (
        'Control a Franka Panda arm with a parallel gripper from RGB, the task instruction and proprioception'
        + (' and fingertip tactile sensing' if tactile else '') + ', using exactly one act call per turn. '
        'In each act call write execution_note first: one or two sentences on what the latest observation shows '
        'and why the rows follow from it; then the rows. Simulator object poses, '
        'rewards, success labels and task code are not available.\n\n'
        + layout +
        'ee_pos_mm and ee_quat_wxyz are the panda_hand pose in the robot base frame. finger_mm is the measured '
        'opening of each finger (0 closed, 39 fully open).\n\n'
        'Each action row has 7 numbers [dx, dy, dz, rx, ry, rz, finger]. dx,dy,dz: translation in millimetres '
        'in the base frame. rx,ry,rz: base-frame rotation vector in degrees, not Euler angles. finger: ABSOLUTE '
        'target opening of each finger in millimetres; this is position control, not a force command. '
        'The first row starts from the latest observed EE pose; later rows accumulate from the previous row\'s '
        f'target. Per row, translation norm <= {limits["translation_mm"]} mm, rotation norm <= '
        f'{limits["rotation_deg"]} deg, 0 <= finger <= {limits["finger_max_mm"]}. Predict 1..{horizon} rows per '
        'act. Each row runs as one planned motion; execution stops early on task success, a safety stop or a '
        'planning failure, and a planning failure ends the episode. The tool receipt reports executed rows and '
        'the measured result state. Small tracking residuals do not prove grasping or completion. '
        'Continue until the environment ends the episode.\n\n'
        'TRAIN is a separate recorded trajectory of the same task that the official evaluator judged successful; '
        'its final observation shows what task completion looks like. Learn its progression and action scale, '
        'then adapt to the current observation; do not blindly replay coordinates, object size and pose may '
        'differ. TRAIN and retained LIVE chunks show a starting observation, an act call, a receipt and the '
        'ending observation. <TRAJECTORY_GAP> marks omitted history, not a continuous transition. '
        'Use the latest LIVE observation for current geometry.')


TRAIN_NOTE = 'Recorded demonstration rows; no rationale was recorded.'


def act_tool(horizon):
    return [{'type': 'function', 'function': {
        'name': 'act', 'description': 'Submit 1..H action rows; receive the executed range and the resulting observation.',
        'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['execution_note', 'actions'],
                       'properties': {
                           'execution_note': {'type': 'string', 'maxLength': 300,
                                              'description': 'Written before actions: what the latest observation shows and why these rows.'},
                           'actions': {'type': 'array', 'minItems': 1, 'maxItems': horizon,
                                       'items': {'type': 'array', 'minItems': 7, 'maxItems': 7, 'items': {'type': 'number'}}}}}}}]


def validate_rows(rows, horizon, limits):
    if not isinstance(rows, list) or not 1 <= len(rows) <= horizon:
        raise ActionError(f'actions must be a list of 1..{horizon} rows')
    for index, row in enumerate(rows):
        if (not isinstance(row, list) or len(row) != 7
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row)):
            raise ActionError(f'row {index} must contain 7 finite numbers')
        if math.hypot(*row[:3]) > limits['translation_mm'] + 1e-9:
            raise ActionError(f'row {index} translation norm {math.hypot(*row[:3]):.2f} mm exceeds {limits["translation_mm"]} mm')
        if math.hypot(*row[3:6]) > limits['rotation_deg'] + 1e-9:
            raise ActionError(f'row {index} rotation norm {math.hypot(*row[3:6]):.2f} deg exceeds {limits["rotation_deg"]} deg')
        if not 0 <= row[6] <= limits['finger_max_mm']:
            raise ActionError(f'row {index} finger {row[6]} outside [0, {limits["finger_max_mm"]}] mm')
    return [[float(v) for v in row] for row in rows]


def parse_call(message, horizon, limits):
    calls = message.get('tool_calls') or []
    if len(calls) != 1 or calls[0]['function']['name'] != 'act':
        raise ActionError('Return exactly one act tool call')
    try:
        arguments = json.loads(calls[0]['function']['arguments'])
    except json.JSONDecodeError as error:
        raise ActionError(f'act arguments are not valid JSON: {error}') from None
    if not isinstance(arguments, dict) or set(arguments) != {'actions', 'execution_note'}:
        raise ActionError('act arguments must contain exactly actions and execution_note')
    return calls[0], validate_rows(arguments['actions'], horizon, limits), str(arguments['execution_note'])


def observation_message(source, step, state, image, end_of_example=False):
    header = {'source': source, 'observation_step': step, 'robot_state': state}
    if end_of_example:
        header['end_of_example'] = True
        header['official_outcome'] = 'success'
    return {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(header, ensure_ascii=False)},
                                        {'type': 'image_url', 'image_url': {'url': data_url(image)}}]}


def call_messages(call_id, rows, note, receipt):
    # Note before rows, matching the tool schema: with thinking off the note is the only reasoning (MotorMind's rationale).
    arguments = json.dumps({'execution_note': note, 'actions': rows}, ensure_ascii=False, separators=(',', ':'))
    return [{'role': 'assistant', 'content': None,
             'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': 'act', 'arguments': arguments}}]},
            {'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(receipt, ensure_ascii=False)}]


def train_messages(reference, tactile, system1=None):
    """Fixed TRAIN prefix; identical across turns so the server can reuse its KV cache.

    system1(number, block, end_waypoint) -> (stop_reason, system1 receipt) renders guard receipts."""
    waypoints = reference['waypoints']

    def observe(index):
        w = waypoints[index]
        images = {name: rgb_image(spec['bytes']) for name, spec in w['images'].items()}
        image = grid(images['head'], images['wrist'], images['left'] if tactile else None, images['right'] if tactile else None)
        state = robot_state(w['ee'], w['finger_m'], w['press_depth_mm'] if tactile else None)
        return state, observation_message('TRAIN', index, state, image, end_of_example=index == len(waypoints) - 1)

    messages = [observe(0)[1]]
    for number, block in enumerate(reference['blocks']):
        result_state, result_message = observe(block['end'])
        receipt = {'source': 'TRAIN', 'executed_step_interval': [block['start'], block['end']],
                   'requested_rows': len(block['actions']), 'executed_rows': len(block['actions']),
                   'stop_reason': None, 'result_state': result_state}
        if system1 is not None:
            receipt['stop_reason'], receipt['system1'] = system1(number, block, waypoints[block['end']])
        messages += call_messages(f'train_{number}', block['actions'], TRAIN_NOTE, receipt)
        messages.append(result_message)
    messages.append({'role': 'user', 'content': [{'type': 'text', 'text':
        'END OF FIXED TASK REFERENCE. Subsequent observations belong to the active episode.'}]})
    return messages


class Client:
    """OpenAI-compatible chat completions; every request/response is archived, images by hash."""

    def __init__(self, base_url, model, timeout_s, max_tokens, enable_thinking, audit_dir):
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.model, self.timeout_s, self.max_tokens, self.enable_thinking = model, timeout_s, max_tokens, enable_thinking
        self.audit = Path(audit_dir)
        (self.audit / 'images').mkdir(parents=True, exist_ok=True)

    def _archived(self, body):
        def strip(value):
            if isinstance(value, dict):
                if value.get('type') == 'image_url':
                    url = value['image_url']['url']
                    digest = hashlib.sha256(url.encode()).hexdigest()
                    path = self.audit / 'images' / f'{digest}.txt'
                    if not path.exists():
                        path.write_text(url)
                    return {'type': 'image_url', 'image_sha256': digest}
                return {k: strip(v) for k, v in value.items()}
            if isinstance(value, list):
                return [strip(v) for v in value]
            return value
        return strip(body)

    def complete(self, messages, tools, request_id):
        body = {'model': self.model, 'messages': messages, 'tools': tools,
                'tool_choice': {'type': 'function', 'function': {'name': 'act'}}, 'parallel_tool_calls': False,
                'temperature': 0, 'stream': False, 'max_tokens': self.max_tokens,
                'chat_template_kwargs': {'enable_thinking': self.enable_thinking}}
        started = time.monotonic()
        request = urllib.request.Request(self.url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        record, payload = {'request_id': request_id, 'request': self._archived(body)}, None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            record.update(http_status=error.code, error=error.read().decode(errors='replace'))
            raise RuntimeError(f'Model server HTTP {error.code}: {record["error"][:500]}') from None
        finally:
            record['wall_s'] = time.monotonic() - started
            with (self.audit / 'requests.jsonl').open('a') as f:
                f.write(json.dumps({**record, 'response': payload}, ensure_ascii=False) + '\n')
        choice = payload['choices'][0]
        if choice.get('finish_reason') not in ('tool_calls', 'stop'):
            raise ActionError(f'Generation ended with finish_reason={choice.get("finish_reason")}; answer more briefly')
        return choice['message'], {'usage': payload.get('usage'), 'wall_s': record['wall_s']}
