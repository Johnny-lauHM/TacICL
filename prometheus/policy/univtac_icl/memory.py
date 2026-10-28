"""Anchored LIVE interaction memory, ported from RoboICL's trajectory_memory.

Source: third_party/RoboICL/roboicl/policy/trajectory_memory.py (MIT, commit
07d6e1c). Logic is unchanged; only messages use the chat-completions content
format. Full chunks are kept at a few uniform step anchors, the latest completed
chunk is kept for immediate correction, and everything else collapses into an
explicit <TRAJECTORY_GAP> with each chunk's interval and stated intent.
"""
from copy import deepcopy
import json


def omitted_interval(source, start, end, *, reason='unshown', summary=None):
    payload = {'source': source, 'reason': reason, 'omitted_action_interval': [start, end],
               'interval_convention': '[start,end)',
               'notice': ('Intermediate actions and observations are not shown. '
                          'Do not infer continuity or task success across this gap.')}
    if summary is not None:
        payload['summary'] = deepcopy(summary)
        payload['summary_scope'] = 'Reported action intent and execution metadata; not verified visual outcome or task success.'
    return {'role': 'user', 'content': [{'type': 'text', 'text': '<TRAJECTORY_GAP>' + json.dumps(
        payload, ensure_ascii=False, separators=(',', ':')) + '</TRAJECTORY_GAP>'}]}


class LiveTrajectoryMemory:
    def __init__(self, count, maximum, retain_latest=True):
        if type(count) is not int or count < 2 or type(maximum) is not int or maximum < 1:
            raise ValueError('Anchored memory needs count >= 2 and a positive step maximum')
        self.count, self.retain_latest = count, retain_latest
        self.anchors = tuple(sorted({i * maximum // count for i in range(count - int(retain_latest))}))
        self.records = []

    def capture(self, start, end, items, summary):
        """Record committed execution [start, end); zero-step chunks never consume an anchor."""
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError('capture bounds must be nonnegative ordered integers')
        if start == end:
            return False
        if self.records and start < self.records[-1]['end']:
            raise ValueError('captured intervals must be chronological and nonoverlapping')
        retained = any(start < anchor <= end or (anchor == 0 and start == 0) for anchor in self.anchors)
        self.records.append({'start': start, 'end': end, 'retained': retained,
                             'items': deepcopy(list(items)), 'summary': deepcopy(summary)})
        return retained

    def stats(self):
        latest = self.records[-1] if self.retain_latest and self.records else None
        retained = [r for r in self.records if r['retained'] or r is latest]
        return {'anchor_steps': list(self.anchors), 'captured_chunks': len(self.records),
                'retained_intervals': [[r['start'], r['end']] for r in retained]}

    def render(self, initial_observation, current_step, observation_fn):
        """Initial observation, retained chunks with boundary observations, gaps, current observation."""
        eligible = [r for r in self.records if r['end'] <= current_step]
        if self.retain_latest and eligible and eligible[-1]['end'] != current_step:
            raise ValueError('Current observation must be the latest committed chunk endpoint')
        latest = eligible[-1] if self.retain_latest and eligible else None
        result, pending, shown, previous_end = [deepcopy(initial_observation)], [], 0, 0

        def flush():
            if pending:
                result.append(omitted_interval('LIVE', pending[0]['action_interval'][0], pending[-1]['action_interval'][1],
                                               reason='compressed_history', summary={'chunks': deepcopy(pending)}))
                pending.clear()

        def show(step):
            nonlocal shown
            if shown != step:
                result.append(deepcopy(observation_fn(step)))
                shown = step

        for record in eligible:
            if previous_end < record['start']:
                pending.append({'action_interval': [previous_end, record['start']], 'reason': 'unrecorded_interval'})
            if record['retained'] or record is latest:
                flush()
                show(record['start'])
                result.extend(deepcopy(record['items']))
                show(record['end'])
            else:
                pending.append({'action_interval': [record['start'], record['end']], 'execution': deepcopy(record['summary'])})
            previous_end = record['end']
        flush()
        show(current_step)
        return result
