"""DogPilot: highway plan smoothing (param ``DogPilotPlanSmooth``, off by default; used by lateral_angle_ext.py).

On the highway the model's planned turn rate wobbles from one plan to the next, and the wheel follows it: a 0.7-3 s
swing of a few tenths of a m/s^2 in gentle curves. Two things calm it without adding lag:

  plan average     each planned turn-rate point is averaged with its neighbours +-1.0 s along the model's own forecast,
                   for both the request and BluePilot's predicted-curvature channel (the forecast already looks ahead,
                   so averaging along it doesn't delay anything)
  small reversals  a play operator 0.3 m/s^2 wide on the request absorbs back-and-forth smaller than that, with a 1 s
                   pull back toward the input so it can't hold an offset

The total change to the command is capped at +-0.15 m/s^2 of sideways acceleration, so a quick move the model really
means (easing away from an oncoming truck: 0.7 m/s^2) still gets through (0.54 of it in replay).
It only acts above 45 mph (full from 55 mph), never in lane changes, and fades out when the plan holds 1.8-2.5 m/s^2
of turning within the next 3 s, so tight curves, curve entries and everything below 45 mph are exactly as before.
Offline (5 drives): gentle-curve swing -13% at 45-60 mph and -19% at 60-90 mph; the path it would change moves at most
0.28 m.
"""
import numpy as np

from opendbc.car.common.conversions import Conversions as CV

T_IDXS = 10.0 * (np.arange(33) / 32.0) ** 2   # ModelConstants.T_IDXS
AVG_S = 1.0               # plan average half-width (s)
PLAY_MS2 = 0.30           # small-reversal filter width (sideways accel, m/s^2)
PLAY_RECENTRE_S = 1.0
CAP_MS2 = 0.15            # largest change to the command (sideways accel, m/s^2)
SPEED_MPH = (45.0, 55.0)  # off below, full above
DEMAND_MS2 = (1.8, 2.5)   # plan's peak sideways accel within DEMAND_T_S: full below, off above
DEMAND_T_S = 3.0
_AVG_MASKS = [(T_IDXS >= t - AVG_S) & (T_IDXS <= t + AVG_S) for t in T_IDXS]
_DEMAND_MASK = T_IDXS <= DEMAND_T_S


def horizon_average(z):
  z = np.asarray(z, dtype=np.float64)
  return np.array([z[m].mean() for m in _AVG_MASKS])


def curvature_from_plan(z, v, action_t):
  """The request modeld builds from a planned turn-rate row (openpilot's get_curvature_from_plan)."""
  z = np.asarray(z, dtype=np.float64)
  psi = np.concatenate([[0.0], np.cumsum(0.5 * (z[1:] + z[:-1]) * np.diff(T_IDXS))])
  a = max(action_t, 0.3)
  pa = float(np.interp(a, T_IDXS, psi))
  if action_t < 0.3:
    pa *= max(action_t, 0.0) / 0.3
  v = max(v, 1.0)
  return 2.0 * pa / (v * max(action_t, 1e-3)) - float(z[0]) / v


class DPPlanSmooth:
  def __init__(self, dt=0.05):
    self.dt = dt
    self.play_y = None
    self.gate = 0.0

  def gate_for(self, z, v):
    g = float(np.interp(v * CV.MS_TO_MPH, SPEED_MPH, [0.0, 1.0]))
    if g <= 0.0:
      return 0.0
    peak = float(np.max(np.abs(np.asarray(z, dtype=np.float64)[_DEMAND_MASK]))) * max(v, 0.0)
    return g * float(np.interp(peak, DEMAND_MS2, [1.0, 0.0]))

  def update(self, z, desired, v, action_t, active, lane_change):
    """Returns (request, plan turn-rate row, gate). Off (gate 0): the inputs unchanged."""
    z = np.asarray(z, dtype=np.float64)
    z_avg = horizon_average(z)
    k = desired + curvature_from_plan(z_avg, v, action_t) - curvature_from_plan(z, v, action_t)
    if not active or self.play_y is None:
      self.play_y = k
    h = 0.5 * PLAY_MS2 / max(v, 3.0) ** 2
    y = min(max(self.play_y, k - h), k + h)
    self.play_y = y + (k - y) * min(1.0, self.dt / PLAY_RECENTRE_S)
    self.gate = self.gate_for(z, v) if active and not lane_change else 0.0
    if self.gate <= 0.0:
      return desired, z, 0.0
    return desired + self.gate * (self.play_y - desired), z + self.gate * (z_avg - z), self.gate

  @staticmethod
  def cap(base, smoothed, v):
    """Limit the change to the path angle (rad/s-like: path angle x speed = sideways accel) to CAP_MS2."""
    c = CAP_MS2 / max(v, 3.0)
    return base + min(max(smoothed - base, -c), c)
