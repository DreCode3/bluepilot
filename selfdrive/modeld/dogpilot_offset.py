"""DogPilot: lateral camera offset applied in the model frame (used by modeld.py)."""
import numpy as np


def dp_shift_model_warp(warp, model_intrinsics, height, offset):
  """DogPilot: shift the model's view sideways by `offset` m (+ = model centre moves left, like sunnypilot's
  CameraOffset). Applied in the level, calibrated model frame, so on a flat road the shift is the same at
  every distance. (sunnypilot's helper shears in camera pixels around the principal point, which with a
  pitched camera shifts near road more than far road -- partly a yaw change, not a pure offset.)
  Model pixel (u, v) of ground point (x, y): u = fx*y/x + cx, v = fy*h/x + cy  ->  shift u by fx*offset/x.
  """
  fx, fy, cy = model_intrinsics[0, 0], model_intrinsics[1, 1], model_intrinsics[1, 2]
  k = fx * offset / (fy * height)
  shear = np.array([[1.0, k, -k * cy], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
  return (warp @ shear).astype(np.float32)


# DogPilot: curve centering. In curves the driving model plans a line toward the inside of the curve, roughly
# 0.3 m per m/s^2 of lateral acceleration, and the car follows it. Shifting the model's view toward the outside
# (the same shift as CameraOffset) makes the model's own planner aim closer to the lane centre. Curvature comes
# from the previous frame's lane lines; a uniform sideways shift doesn't change it, so this is feed-forward.
DP_CURVE_X_IDX = (8, 10, 12, 14)        # lane-line samples used for the curvature fit: x = 12, 18.75, 27, 36.75 m
DP_CURVE_M_PER_MS2 = 0.20               # outward shift per m/s^2 above the deadband, at strength 1.0
DP_CURVE_DEADBAND_MS2 = 0.25            # no shift on straights and very gentle curves
DP_CURVE_MAX_M = 0.35                   # hard cap on the curve shift, any strength
DP_CURVE_RATE_MS = 0.20                 # max change of the shift, m per second
DP_CURVE_TAU_S = 0.30                   # smoothing of the target
DP_CURVE_V_MIN, DP_CURVE_V_FULL = 9.0, 13.0  # m/s: off below ~20 mph, full above ~29 mph
# Fade the shift out only in extreme curves. It used to fade from 1.8 m/s^2 because the car ran short of
# steering there, but the tight-curve boost now covers that, and removing the outward shift at the entry of a
# 2.4-2.8 m/s^2 highway curve made the car swing 0.5-0.95 m inside (the model also reacts to the shift moving).
DP_CURVE_FADE_MS2 = (3.0, 3.8)
# The fade must not flicker: the measured curvature wobbles inside the 1.8-2.5 band and the model reacts to the
# shift moving (Oct 25 ramp: shift jumped between -0.28 and 0 m every ~3 s). The fade rises quickly, holds while
# the curve is still tight (above DP_CURVE_FADE_HOLD_MS2), and releases quickly once it has eased off.
DP_CURVE_FADE_RISE_PER_S = 2.0
DP_CURVE_FADE_HOLD_MS2 = 1.5
DP_CURVE_FADE_HOLD_FALL_PER_S = 0.08
DP_CURVE_FADE_FALL_PER_S = 1.0
# Early mode (param DogPilotCurveEarly): on the highway the car moves inside within the first 2 s of a curve,
# before the shift has built, and when the shift drops at the exit the model swings back toward its old line.
# Early mode also measures the curve farther ahead (27-61 m, where the model's lane lines are still accurate to
# a few cm) and uses whichever of near/far is curving harder, so the shift builds before the curve and holds
# through it; and it releases the shift more slowly after the curve.
# Wide-lane cap (param DogPilotCurveWideCap): in long moderate curves the shift sits at the 0.35 m cap and the car
# can still drift 0.5-0.8 m in. With this on, the cap follows the lane width the model sees (line centre to line
# centre; 3.26-3.40 m on Oct 25 interstates): half the room beside a 2.0 m-wide car, minus a 0.20 m margin, between
# 0.20 and 0.50 m. 3.4 m allows 0.50 m, 3.26 m 0.43 m, 3.1 m 0.35 m (as today), 3.0 m 0.30 m. It bounds the view
# shift; the car itself moves only 30-75% of it and still sits slightly inside in curves.
DP_CURVE_CAR_WIDTH_M = 2.0
DP_CURVE_LINE_MARGIN_M = 0.20
DP_CURVE_WIDE_CAP_RANGE = (0.20, 0.50)
DP_CURVE_WIDTH_TAU_S = 1.0
DP_CURVE_X_IDX_FAR = (12, 14, 16, 18)   # x = 27, 36.75, 48, 60.75 m
DP_CURVE_RELEASE_MS = 0.08              # early mode: release rate toward 0, m per second


def dp_lane_curvature(left_y, right_y, x_idxs):
  """Curvature (1/m, + = curving right in the device frame) of the lane centre from lane-line y at model x samples."""
  xs = np.asarray(x_idxs, dtype=np.float64)
  centre = (np.asarray(left_y, dtype=np.float64) + np.asarray(right_y, dtype=np.float64)) / 2.0
  a = np.stack([np.ones_like(xs), xs, xs * xs], axis=1)
  return float(2.0 * np.linalg.lstsq(a, centre, rcond=None)[0][2])


class DPCurveCentering:
  """Sideways shift (m, + = car moves left, like CameraOffset) toward the outside of the curve ahead."""

  def __init__(self, x_samples, dt=0.05):
    self._k_row = self._fit_row(x_samples, DP_CURVE_X_IDX)        # curvature = k_row . centre_y
    self._k_row_far = self._fit_row(x_samples, DP_CURVE_X_IDX_FAR)
    self.dt = dt
    self.early = False
    self.wide_cap = False
    self.width = None         # smoothed lane width (m) for the wide-lane cap
    self.tight = 0.0          # held fade level, 0..1 (see DP_CURVE_FADE_*)
    self.target = 0.0
    self.offset = 0.0

  @staticmethod
  def _fit_row(x_samples, x_idx):
    xs = np.asarray(x_samples, dtype=np.float64)[list(x_idx)]
    a = np.stack([np.ones_like(xs), xs, xs * xs], axis=1)
    return 2.0 * np.linalg.pinv(a)[2]

  def _curvature(self, left_y, right_y, left_prob, right_prob):
    """Lane-centre curvature ahead (1/m, + = right), or None when the lane lines aren't usable."""
    left_y, right_y = np.asarray(left_y), np.asarray(right_y)
    width = float(right_y[0] - left_y[0])
    if left_prob < 0.5 or right_prob < 0.5 or not 2.6 < width < 4.6:
      return None
    idx = list(DP_CURVE_X_IDX)
    k = float(self._k_row @ ((left_y[idx] + right_y[idx]) / 2.0))
    if self.early:
      far = list(DP_CURVE_X_IDX_FAR)
      k_far = float(self._k_row_far @ ((left_y[far] + right_y[far]) / 2.0))
      if abs(k_far) > abs(k):
        k = k_far
    return k

  @staticmethod
  def _tight(a_lat):
    return min(max((a_lat - DP_CURVE_FADE_MS2[0]) / (DP_CURVE_FADE_MS2[1] - DP_CURVE_FADE_MS2[0]), 0.0), 1.0)

  def cap(self):
    """Largest shift allowed now: DP_CURVE_MAX_M, or with the wide-lane cap on, from the smoothed lane width."""
    if not self.wide_cap or self.width is None:
      return DP_CURVE_MAX_M
    room = (self.width - DP_CURVE_CAR_WIDTH_M) / 2.0 - DP_CURVE_LINE_MARGIN_M
    return min(max(room, DP_CURVE_WIDE_CAP_RANGE[0]), DP_CURVE_WIDE_CAP_RANGE[1])

  def target_for(self, left_y, right_y, left_prob, right_prob, v_ego, strength, tight=None):
    """Shift for this frame. tight: held fade level (update() passes it); None = this frame's own."""
    k = self._curvature(left_y, right_y, left_prob, right_prob)
    if strength <= 0.0 or k is None:
      return 0.0
    a_lat = abs(k) * v_ego * v_ego
    fade = min(max((v_ego - DP_CURVE_V_MIN) / (DP_CURVE_V_FULL - DP_CURVE_V_MIN), 0.0), 1.0)
    tight = self._tight(a_lat) if tight is None else tight
    shift = min(self.cap(), strength * DP_CURVE_M_PER_MS2 * max(a_lat - DP_CURVE_DEADBAND_MS2, 0.0)) * fade * (1.0 - tight)
    # right curve (k > 0): inside is right, so move the car left (+)
    return shift if k > 0 else -shift

  def update(self, left_y, right_y, left_prob, right_prob, v_ego, strength, active, hold=False):
    """hold: keep the current shift unchanged (during a lane change). The model reacts to the shift moving
    (~1 m/s^2 of steering per m/s of shift change), so releasing it mid lane change nudged the lane change."""
    if hold and active:
      return self.offset
    k = self._curvature(left_y, right_y, left_prob, right_prob) if active else None
    if k is not None:   # usable lane lines: track the lane width for the wide-lane cap
      w = float(np.asarray(right_y)[0] - np.asarray(left_y)[0])
      self.width = w if self.width is None else self.width + (w - self.width) * min(1.0, self.dt / DP_CURVE_WIDTH_TAU_S)
    a_lat = abs(k) * v_ego * v_ego if k is not None else 0.0
    tight_now = self._tight(a_lat)
    if tight_now >= self.tight:
      self.tight = min(tight_now, self.tight + DP_CURVE_FADE_RISE_PER_S * self.dt)
    else:
      fall = DP_CURVE_FADE_HOLD_FALL_PER_S if a_lat > DP_CURVE_FADE_HOLD_MS2 else DP_CURVE_FADE_FALL_PER_S
      self.tight = max(tight_now, self.tight - fall * self.dt)
    raw = self.target_for(left_y, right_y, left_prob, right_prob, v_ego, strength, self.tight) if active else 0.0
    self.target += (raw - self.target) * min(1.0, self.dt / DP_CURVE_TAU_S)
    if raw == 0.0 and abs(self.target) < 1e-4:
      self.target = 0.0
    step = DP_CURVE_RATE_MS * self.dt
    releasing = abs(self.target) < abs(self.offset) and self.target * self.offset >= 0.0
    if self.early and releasing:
      step = DP_CURVE_RELEASE_MS * self.dt
    self.offset += min(max(self.target - self.offset, -step), step)
    return self.offset
