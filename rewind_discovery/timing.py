"""호출 경계의 겹치는 wall/process CPU 계측. 순수 network/gzip 분리 아님."""
from contextlib import contextmanager
import time


class Timing:
    def __init__(self, wall=None, cpu=None):
        self.wall = wall or time.monotonic
        self.cpu = cpu or time.process_time
        self.started = self.wall(), self.cpu()
        self.calls = {}

    @contextmanager
    def span(self, name):
        before = self.wall(), self.cpu()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            row = self.calls.setdefault(name, {'calls': 0, 'failed_calls': 0, 'wall_seconds': 0., 'process_cpu_seconds': 0.})
            row['calls'] += 1
            row['failed_calls'] += int(failed)
            row['last_wall_seconds'] = max(0., self.wall()-before[0])
            row['last_process_cpu_seconds'] = max(0., self.cpu()-before[1])
            row['wall_seconds'] += row['last_wall_seconds']
            row['process_cpu_seconds'] += row['last_process_cpu_seconds']

    def call(self, name, function, *args, **kwargs):
        with self.span(name):
            return function(*args, **kwargs)

    def snapshot(self):
        return {'wall_seconds': max(0., self.wall()-self.started[0]),
            'process_cpu_seconds': max(0., self.cpu()-self.started[1]),
            'calls': {k: dict(v) for k, v in self.calls.items()},
            'scope': 'overlapping call wall/process CPU; HTTP headers includes SDK/auth, body_read1 includes stream consumption; no pure network/gzip subtraction'}
