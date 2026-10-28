"""Re-express a TRAIN reference as System 2 calls: one grip mode + motion rows per call.

The recorded demonstration is segmented by grip phase using only recorded public
signals (EE motion, finger opening, pad press depth):

* closing   -- rows before the first moving row: the hand stays still while the fingers close;
* holding   -- moving rows: the hand moves while the fingers keep (light) contact;
* releasing -- rows after the last moving row: the fingers open.

Under ``s1_tactile`` closing/holding become ``contact`` with the median press depth
of the weaker pad during holding; under ``s1_position`` they become ``position``
with the measured opening at the start of holding. Releasing becomes ``open``.
"""
import math
import statistics

MOVING_MM, MOVING_DEG = 0.5, 0.5


def _moving(row):
    return math.hypot(*row[:3]) > MOVING_MM or math.hypot(*row[3:6]) > MOVING_DEG


def segment(reference, control, horizon):
    waypoints = reference['waypoints']
    rows = [row for block in reference['blocks'] for row in block['actions']]
    if len(rows) != len(waypoints) - 1:
        raise ValueError('Reference rows must connect consecutive waypoints')
    moving = [i for i, row in enumerate(rows) if _moving(row)]
    if not moving:
        raise ValueError('Reference has no moving rows')
    first, last = moving[0], moving[-1]
    hold = waypoints[first:last + 2]
    if control == 's1_tactile':
        target = round(statistics.median(min(w['press_depth_mm'].values()) for w in hold), 2)
        if target <= 0:
            raise ValueError('Reference keeps no contact while moving; no contact target to show')
        grip = {'mode': 'contact', 'value': target}
    elif control == 's1_position':
        grip = {'mode': 'position', 'value': round(waypoints[first]['finger_m'] * 1000, 1)}
    else:
        raise ValueError(f'Unknown control {control}')
    calls = [{'start': 0, 'end': first, 'grip': grip, 'actions': []}]
    for start in range(first, last + 1, horizon):
        end = min(start + horizon, last + 1)
        calls.append({'start': start, 'end': end, 'grip': grip,
                      'actions': [[float(v) for v in row[:6]] for row in rows[start:end]]})
    calls.append({'start': last + 1, 'end': len(waypoints) - 1, 'grip': {'mode': 'open', 'value': 0.0},
                  'actions': []})
    return [call for call in calls if call['end'] > call['start'] or call['actions']]
