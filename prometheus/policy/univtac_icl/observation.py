"""Public UniVTAC observations rendered as one image grid plus a small state dict.

The same functions serve recorded TRAIN frames and LIVE simulator frames, so the
two sources share identical image layout, units and rounding.
"""
import base64
import io

import numpy as np
from PIL import Image, ImageDraw

CELL = (480, 270)
TACTILE_SIDES = ('left', 'right')


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def rgb_image(value):
    """Accept HWC uint8 arrays/tensors or encoded image bytes."""
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return Image.open(io.BytesIO(bytes(value))).convert('RGB')
    array = _array(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f'Expected HWC uint8 RGB, got {array.dtype} {array.shape}')
    return Image.fromarray(array)


def _fit(image, size):
    canvas = Image.new('RGB', size, (0, 0, 0))
    copy = image.copy()
    copy.thumbnail(size)
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def grid(head, wrist, left=None, right=None):
    """HEAD | WRIST on top; LEFT GEL | RIGHT GEL below when tactile is shown."""
    if (left is None) != (right is None):
        raise ValueError('Both tactile pads or neither are required')
    rows = 2 if left is not None else 1
    canvas = Image.new('RGB', (2 * CELL[0], rows * CELL[1]))
    panels = [('HEAD', head), ('WRIST', wrist)]
    if left is not None:
        panels += [('LEFT GEL', left), ('RIGHT GEL', right)]
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x, y = (index % 2) * CELL[0], (index // 2) * CELL[1]
        canvas.paste(_fit(image, CELL), (x, y))
        draw.rectangle((x, y, x + 8 * len(label) + 6, y + 14), fill=(0, 0, 0))
        draw.text((x + 3, y + 2), label, fill=(255, 255, 0))
    return canvas


def data_url(image, quality=90):
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()


def robot_state(ee, finger_m, press_depth_mm=None):
    """Rounded public proprioception; tactile depth only when the condition shows tactile."""
    ee = _array(ee).reshape(-1).astype(float)
    if ee.shape != (7,) or not np.isfinite(ee).all():
        raise ValueError('EE pose must be 7 finite numbers [x,y,z,qw,qx,qy,qz]')
    if not np.isclose(np.linalg.norm(ee[3:]), 1, atol=1e-3):
        raise ValueError('EE quaternion is not unit wxyz')
    state = {'ee_pos_mm': [round(float(v) * 1000, 1) for v in ee[:3]],
             'ee_quat_wxyz': [round(float(v), 4) for v in ee[3:]],
             'finger_mm': round(float(finger_m) * 1000, 2)}
    if press_depth_mm is not None:
        state['press_depth_max_mm'] = {side: round(float(press_depth_mm[side]), 3) for side in TACTILE_SIDES}
    return state


def live_observation(raw, tactile):
    """Whitelist public sensors from an official UniVTAC observation dict."""
    joint = _array(raw['embodiment']['joint']).reshape(-1)
    images = {name: rgb_image(raw['observation'][name]['rgb']) for name in ('head', 'wrist')}
    press = None
    if tactile:
        press = {}
        for side in TACTILE_SIDES:
            pad = raw['tactile'][side + '_tactile']
            images[side] = rgb_image(pad['rgb_marker'])
            press[side] = float(_array(pad['press_depth']).max())
    return {'step': int(raw['step']), 'ee': _array(raw['embodiment']['ee']).reshape(-1).astype(float).tolist(),
            'finger_m': float(joint[7:9].mean()),
            'state': robot_state(raw['embodiment']['ee'], joint[7:9].mean(), press),
            'image': grid(images['head'], images['wrist'], images.get('left'), images.get('right'))}
