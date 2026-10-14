import time


class SimpleProgress:
  """tqdm-like progress bar that prints at every 10% milestone."""

  def __init__(self, iterable=None, total=None, desc=None, leave: bool=False, unit="it"):
    self.iterable = iterable
    self.total = total if total is not None else (len(iterable) if iterable is not None else None)
    self.desc = desc
    self.unit = unit
    self.n = 0
    self._last_printed_pct = -1
    self._start_time = None

  def __iter__(self):
    self._start()
    for item in self.iterable:
      yield item
      self.update(1)
    self._finish()

  def __enter__(self):
    self._start()
    return self

  def __exit__(self, *args):
    self._finish()

  def _start(self):
    self.n = 0
    self._last_printed_pct = -1
    self._start_time = time.monotonic()

  def _finish(self):
    if self._last_printed_pct < 100 and self.total and self.n >= self.total:
      self._print_line(100)

  def update(self, n=1):
    if self._start_time is None:
      self._start()
    self.n += n
    if self.total:
      pct = int(self.n / self.total * 100)
      milestone = (pct // 10) * 10
      if 100 >= milestone > self._last_printed_pct:
        self._last_printed_pct = milestone
        self._print_line(milestone)

  @staticmethod
  def _format_time(seconds):
    if seconds is None:
      return "?"
    seconds = int(seconds)
    if seconds < 60:
      return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
      return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"

  def _print_line(self, milestone):
    elapsed = time.monotonic() - self._start_time
    speed = self.n / elapsed if elapsed > 0 else None

    if self.total and self.n > 0 and speed:
      remaining = (self.total - self.n) / speed
    else:
      remaining = None

    prefix = f"{self.desc}: " if self.desc else ""
    bar = "#" * (milestone // 10) + "." * (10 - milestone // 10)
    speed_str = f"{speed:.2f}" if speed is not None else "?"

    print(
      f"{prefix}[{bar}] {milestone:3d}%  "
      f"{self.n}/{self.total or '?'} {self.unit}  "
      f"elapsed {self._format_time(elapsed)}  "
      f"remaining {self._format_time(remaining)}  "
      f"{speed_str} {self.unit}/s"
    )