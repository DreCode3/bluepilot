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
