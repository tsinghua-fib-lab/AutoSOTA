import time
from collections import deque
from tqdm import tqdm

class CustomTqdm(tqdm):
	def __init__(self, *args, **kwargs):
		self._sma_window = deque(maxlen=1000)
		super().__init__(*args, **kwargs)
		self._sma_window.append((time.time(), self.n))

	def update(self, n=1):
		if n > 0:
			super().update(n)
			self._sma_window.append((time.time(), self.n))
		else:
			super().update(n)

	@property
	def format_dict(self):
		d = super().format_dict
		if len(self._sma_window) > 1:
			start_t, start_n = self._sma_window[0]
			end_t, end_n = self._sma_window[-1]
			delta_t = end_t - start_t
			delta_n = end_n - start_n
			rate = delta_n / delta_t if delta_t > 0 else 0
		else:
			rate = d['rate'] if d.get('rate') else 0
		if rate and rate > 0:
			remaining_items = self.total - self.n
			secs = remaining_items / rate
		else:
			secs = 0
		secs = int(secs)
		if secs < 60:
			eta_str = f"{secs}s"
		elif secs < 300:
			eta_str = f"{secs // 60}m{((secs % 60) // 10) * 10}s"
		elif secs < 3600:
			eta_str = f"{secs // 60}m"
		else:
			eta_str = f"{secs // 3600}h{(secs // 60) % 60}m"
		n_val = d.get('n', 0)
		total_val = d.get('total', 0)
		n_len = len(str(n_val))
		total_len = len(str(total_val)) if total_val is not None else 1
		fixed_len = 64 + 1 + 1 + 5
		content_len = fixed_len + n_len + total_len + len(eta_str)
		ncols = d.get('ncols')
		if ncols:
			pad_len = ncols - content_len - 1
			if pad_len > 0:
				d['remaining'] = eta_str + " " * pad_len
			else:
				d['remaining'] = f"{eta_str:<5} "
		else:
			d['remaining'] = f"{eta_str:<5} "
		return d