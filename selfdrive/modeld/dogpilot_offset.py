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
# Fade the shift out in very tight curves: there the car runs short of steering (it delivers 0.65-0.85 of
# what it's sent above ~2 m/s^2), and the model's inside line needs less steering than the centre line.
DP_CURVE_FADE_MS2 = (1.8, 2.5)


def dp_lane_curvature(left_y, right_y, x_idxs):
  """Curvature (1/m, + = curving right in the device frame) of the lane centre from lane-line y at model x samples."""
  xs = np.asarray(x_idxs, dtype=np.float64)
  centre = (np.asarray(left_y, dtype=np.float64) + np.asarray(right_y, dtype=np.float64)) / 2.0
  a = np.stack([np.ones_like(xs), xs, xs * xs], axis=1)
  return float(2.0 * np.linalg.lstsq(a, centre, rcond=None)[0][2])


class DPCurveCentering:
  """Sideways shift (m, + = car moves left, like CameraOffset) toward the outside of the curve ahead."""

  def __init__(self, x_samples, dt=0.05):
    xs = np.asarray(x_samples, dtype=np.float64)[list(DP_CURVE_X_IDX)]
    a = np.stack([np.ones_like(xs), xs, xs * xs], axis=1)
    self._k_row = 2.0 * np.linalg.pinv(a)[2]   # curvature = k_row . centre_y
    self.dt = dt
    self.target = 0.0
    self.offset = 0.0

  def target_for(self, left_y, right_y, left_prob, right_prob, v_ego, strength):
    left_y, right_y = np.asarray(left_y), np.asarray(right_y)
    width = float(right_y[0] - left_y[0])
    if strength <= 0.0 or left_prob < 0.5 or right_prob < 0.5 or not 2.6 < width < 4.6:
      return 0.0
    idx = list(DP_CURVE_X_IDX)
    k = float(self._k_row @ ((left_y[idx] + right_y[idx]) / 2.0))
    a_lat = abs(k) * v_ego * v_ego
    fade = min(max((v_ego - DP_CURVE_V_MIN) / (DP_CURVE_V_FULL - DP_CURVE_V_MIN), 0.0), 1.0)
    tight = min(max((a_lat - DP_CURVE_FADE_MS2[0]) / (DP_CURVE_FADE_MS2[1] - DP_CURVE_FADE_MS2[0]), 0.0), 1.0)
    shift = min(DP_CURVE_MAX_M, strength * DP_CURVE_M_PER_MS2 * max(a_lat - DP_CURVE_DEADBAND_MS2, 0.0)) * fade * (1.0 - tight)
    # right curve (k > 0): inside is right, so move the car left (+)
    return shift if k > 0 else -shift

  def update(self, left_y, right_y, left_prob, right_prob, v_ego, strength, active):
    raw = self.target_for(left_y, right_y, left_prob, right_prob, v_ego, strength) if active else 0.0
    self.target += (raw - self.target) * min(1.0, self.dt / DP_CURVE_TAU_S)
    if raw == 0.0 and abs(self.target) < 1e-4:
      self.target = 0.0
    step = DP_CURVE_RATE_MS * self.dt
    self.offset += min(max(self.target - self.offset, -step), step)
    return self.offset
