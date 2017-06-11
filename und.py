import numpy as np
from scipy.io import wavfile
from scipy.signal import argrelmax
import hashlib
import os
from codenamize import codenamize
from scipy import signal
import matplotlib.pyplot as plt
import time

def concat(*args):
  return np.concatenate(args, axis=-1)

def constant(value, duration, rate=44100, channels=1):
  samples = int(np.round(duration * rate))
  shape = (channels, samples) if channels > 1 else samples
  return np.full(shape, value, dtype=float)

def crossfade(a, b, phase):
  assert a.shape == b.shape == phase.shape
  assert 0 <= phase.min()
  assert phase.max() <= 1
  return a * (1 - phase) + b * phase

def line(v1, v2, duration, rate=44100):
  samples = int(np.round(duration * rate))
  return np.linspace(v1, v2, num=samples, endpoint=False)

def logline(v1, v2, duration, rate=44100):
  x = line(np.log(v1), np.log(v2), duration, rate)
  np.exp(x, out = x)
  return x

def lowpass(x, cutoff, order=5, rate=44100):
  nyquist = 0.5 * rate
  normalized_cutoff = cutoff / nyquist
  b, a = signal.butter(order, normalized_cutoff, btype="low")
  return signal.lfilter(b, a, x)

def gaussian(duration, rate=44100):
  samples = int(np.round(duration * rate))
  return np.random.normal(0, 1, samples)

def exponential(scale, cutoff=10):
  # samples are truncated at cutoff * scale
  chosen = np.random.random(scale.shape)
  return -np.log(1 - (1 - np.exp(-cutoff)) * chosen) * scale

def exponential_poisson(spike_rate, rate=44100, cutoff=10):
  p = spike_rate / rate
  neverland = p < 1e-10
  p[neverland] = 0.5
  result = exponential(-1 / np.log(p), cutoff)
  result[neverland] = 0
  result[result < 1] = 0
  return result

def poisson(lam, rate=44100):
  warped_duration = lam.sum()
  chosen = np.empty(0)
  n = 1
  while chosen.sum() < warped_duration:
    chosen = concat(chosen, np.random.exponential(rate, n))
    n *= 2
  times = np.searchsorted(np.cumsum(lam), np.cumsum(chosen))
  result = np.zeros_like(lam)
  result[times[times < len(lam)]] = 1
  return result

def poisson(lam, rate=44100, interval_offset=0):
  t_offset = interval_offset * rate
  result = np.zeros_like(lam)
  for i in np.ndindex(*lam.shape[1:]):
    j = (np.s_[:],) + i
    f_table = concat([0], np.cumsum(lam[j]))
    def f(t):
      k = np.clip(int(t), 0, len(lam) - 1)
      phase = np.clip(t - k, 0, 1)
      return f_table[k] + lam[(k,) + i] * phase
    def af(s):
      k = np.searchsorted(f_table, s) - 1
      if k < 0:
        return 0
      if k >= len(lam):
        return len(lam)
      phase = np.clip((s - f_table[k]) / lam[(k,) + i], 0, 1)
      return k + phase
    times = []
    t = af(np.random.exponential(rate)) + interval_offset
    while t < len(lam):
      times.append(t)
      t = af(f(t + t_offset) + np.random.exponential(rate))
    result[(np.array(times, dtype=int),) + i] = 1
  return result

def local_max(x, min_interval=0, rate=44100):
  result = np.zeros_like(x)
  max_indices = argrelmax(x, order=int(max(min_interval * rate, 1)))
  result[max_indices] = x[max_indices]
  return result

def flip_coins(p):
  return np.random.random(p.shape) < p

def fade(duration, start=0, stop=1, rate=44100):
  samples = int(duration * rate)
  run = start - stop
  return np.clip(np.linspace(start / run, (start - 1) / run, samples), 0, 1)

def sinusoid(f, phase=0.25, rate=44100):
  x = np.empty_like(f, dtype=float)
  x[0] = phase * rate
  x[1:] = f[:-1]
  np.cumsum(x, out = x)
  x *= 2 * np.pi / rate
  np.cos(x, out = x)
  return x

def apply_reverb(x, reverb, origin=0):
  result = np.empty_like(x)
  for i in np.ndindex(*x.shape[1:]):
    j = (np.s_[:],) + i
    result[j] = np.convolve(x[j], reverb)[origin:len(x) + origin]
  return result

def apply_blur(x, blur):
  assert len(blur) % 2
  return apply_reverb(x, blur, origin=len(blur) // 2)

def gaussian_window(sd, rate=44100, threshold=1e-6):
  # sd is in seconds. the window generally sums to 1 but for small sd the
  # single center value can be greater than 1
  y_scale = 1 / (np.sqrt(2 * np.pi) * sd * rate)
  cutoff = int(sd * rate * np.sqrt(2 * np.log(y_scale / threshold)) + 1)
  x_scale = cutoff / (sd * rate)
  arg = np.linspace(-x_scale, x_scale, 2 * cutoff + 1)
  return y_scale * np.exp(-0.5 * arg * arg)

middle_c = 220 * 2 ** 0.25

#import matplotlib.pyplot as plt; x = saw1(0.01, 300, rate=1); plt.plot(x); plt.show()
def saw1(f, duration, phase=0, rate=44100):
  x = np.zeros(int(round(duration * rate)))
  for i in range(1, int(0.5 * rate / f) + 1):
    start = 2 * np.pi * i * phase
    stop = 2 * np.pi * i * (f * len(x) / rate + phase)
    factor = (1 - 2 * (i % 2)) / i
    x += np.sin(np.linspace(start, stop, num=len(x), endpoint=False)) * factor
  x *= -2 / np.pi
  return x

#import matplotlib.pyplot as plt; x = square1(0.01, 300, rate=1); plt.plot(x); plt.show()
def square1(f, duration, phase=0, rate=44100):
  x = np.zeros(int(round(duration * rate)))
  for i in range(1, int(0.5 * rate / f) + 1, 2):
    start = 2 * np.pi * i * phase
    stop = 2 * np.pi * i * (f * len(x) / rate + phase)
    x += np.sin(np.linspace(start, stop, num=len(x), endpoint=False)) / i
  x *= 4 / np.pi
  return x

#import matplotlib.pyplot as plt; x = square2(0.01, 300, pulse=0.25, rate=1); plt.plot(x); plt.show()
def square2(f, duration, pulse=0.5, phase=0, rate=44100):
  x = np.zeros(int(round(duration * rate)))
  for i in range(1, int(0.5 * rate / f) + 1):
    start = 2 * np.pi * i * (phase - 0.5 * pulse)
    stop = 2 * np.pi * i * (f * len(x) / rate + phase - 0.5 * pulse)
    factor = -4 * np.sin(np.pi * i * pulse) / (np.pi * i)
    x += np.cos(np.linspace(start, stop, num=len(x), endpoint=False)) * factor
  return x

def int16_normalize(data):
  if np.iscomplexobj(data):
    raise TypeError()
  normalized = data.T * (32767 / max(np.max(data), -np.min(data)))
  np.round(normalized, out=normalized)
  return normalized.astype(np.int16)

def get_codename(normalized):
  return codenamize(
    hashlib.sha1(normalized.ravel().view(np.uint8)).hexdigest()
  )

last_saved = float("-inf")

def save_helper(normalized, filename, rate):
  global last_saved
  if not os.path.exists("sounds"):
    os.makedirs("sounds")
  if time.time() - last_saved < 1:
    time.sleep(1)
  wavfile.write("sounds/" + filename, rate, normalized)
  last_saved = time.time()

def channel_iter(*args):
  return zip(*[x.reshape((-1, x.shape[-1])) for x in args])

def get_spectrum(x):
  spectrum = np.empty(x.shape[:-1] + (x.shape[-1] // 2 + 1,))
  for channel, spectrum_channel in channel_iter(x, spectrum):
    spectrum_channel[:] = np.abs(np.fft.rfft(channel))
  return spectrum

def save(data, rate=44100):
  normalized = int16_normalize(data)
  filename = get_codename(normalized) + ".wav"
  save_helper(normalized, filename, rate)
  return filename

def save_spectrum(data, rate=44100):
  filename = get_codename(int16_normalize(data)) + "_s.wav"
  save_helper(int16_normalize(get_spectrum(data)), filename, rate)
  return filename

def save_fft(data, channel=None, rate=44100):
  if type(channel) is int:
    channel = [channel]
  if channel is None:
    assert data.ndim == 1
    suffix = ""
  else:
    assert data.ndim == 2
    suffix = ",".join(str(i) for i in channel)
  filename = get_codename(int16_normalize(data)) + "_f" + suffix + ".wav"
  fft = np.fft.rfft(data[channel].mean(axis=0))
  fft = np.array([fft.real, fft.imag])
  save_helper(int16_normalize(fft), filename, rate)
  return filename

def load(filename, rate=44100):
  actual_rate, x = wavfile.read(filename)
  assert actual_rate == rate
  assert x.dtype is np.int16
  x = x.T.astype(float)
  x /= 32768
  return x

def show(*args):
  for x in args:
    plt.plot(x)
  plt.show()

if __name__ == "__main__":
  np.random.seed(0)
  poisson(constant(10, 1))
  
  np.random.seed(0)

  #print(save(
  #  sinusoid(
  #    lowpass(gaussian(2), 200) * 200 + 440
  #  )
  #))

  #input()
