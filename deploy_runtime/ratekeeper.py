import time


class RateKeeper:
    def __init__(self, dt: float):
        self.dt = float(dt)
        self._next_tick = None

    def reset(self):
        now = time.time()
        self._next_tick = now + self.dt

    def sleep_until_next(self):
        if self._next_tick is None:
            self.reset()
            return
        now = time.time()
        sleep_time = self._next_tick - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._next_tick += self.dt
