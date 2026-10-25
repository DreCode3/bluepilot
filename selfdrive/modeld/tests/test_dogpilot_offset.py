# DogPilot: the camera offset must shift what the model sees sideways by the same amount at every distance,
# in the documented direction (+ = model centre moves left, so the car moves left), and be exact identity at 0.
import unittest

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.model import get_warp_matrix, medmodel_intrinsics, sbigmodel_intrinsics
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.dogpilot_offset import (DP_CURVE_MAX_M, DP_CURVE_RATE_MS, DP_CURVE_RELEASE_MS, DPCurveCentering,
                                                        dp_shift_model_warp)

H = 1.40
RPY = np.radians([0.0, 2.5, -2.9])  # a pitched, yawed mount, like the car this was written for


def world_shift_seen(cam_intrinsics, model_intrinsics, big, offset, x, y):
  """Lateral offset (m) between the ground point a model pixel sees with and without the shift."""
  rot = view_frame_from_device_frame @ rot_from_euler(RPY)
  q = cam_intrinsics @ (rot @ np.array([x, y, H]))
  cam = q / q[2]
  warp = get_warp_matrix(RPY, cam_intrinsics, big)
  m = np.linalg.inv(warp) @ cam
  c2 = dp_shift_model_warp(warp, model_intrinsics, H, offset) @ (m / m[2])
  d = rot.T @ (np.linalg.inv(cam_intrinsics) @ (c2 / c2[2]))
  return d[1] * H / d[2] - y


class TestDogPilotOffset(unittest.TestCase):
  def setUp(self):
    self.dc = next(v for k, v in DEVICE_CAMERAS.items() if k[0] == "mici")

  def test_uniform_shift_both_cameras(self):
    for cam, model_k, big in ((self.dc.fcam.intrinsics, medmodel_intrinsics, False),
                              (self.dc.ecam.intrinsics, sbigmodel_intrinsics, True)):
      for offset in (-0.15, 0.2):
        for x, y in ((8.0, 0.0), (20.0, 1.8), (60.0, -1.8)):
          self.assertAlmostEqual(world_shift_seen(cam, model_k, big, offset, x, y), offset, places=3)

  def test_zero_offset_is_identity(self):
    warp = get_warp_matrix(RPY, self.dc.fcam.intrinsics, False).astype(np.float32)
    self.assertTrue(np.array_equal(dp_shift_model_warp(warp, medmodel_intrinsics, H, 0.0), warp))



def lanes(k, shift=0.0, half_width=1.8):
  """Lane lines (y, + = right) of a lane curving with curvature k, car shifted `shift` m from its centre."""
  x = np.array(ModelConstants.X_IDXS)
  centre = 0.5 * k * x * x - shift
  return centre - half_width, centre + half_width


class TestDogPilotCurveCentering(unittest.TestCase):
  def settle(self, cc, k, v, strength=1.0, active=True, s=10.0, shift=0.0):
    left, right = lanes(k, shift)
    for _ in range(int(s / cc.dt)):
      cc.update(left, right, 0.9, 0.9, v, strength, active)
    return cc.offset

  def test_moves_toward_outside(self):
    # right curve (k > 0): car moves left (+); left curve: right (-)
    self.assertGreater(self.settle(DPCurveCentering(ModelConstants.X_IDXS), 0.0015, 30.0), 0.1)
    self.assertLess(self.settle(DPCurveCentering(ModelConstants.X_IDXS), -0.0015, 30.0), -0.1)

  def test_size(self):
    # 0.0015 1/m at 30 m/s = 1.35 m/s^2 -> 0.2 * (1.35 - 0.25) = 0.22 m
    self.assertAlmostEqual(self.settle(DPCurveCentering(ModelConstants.X_IDXS), 0.0015, 30.0), 0.22, places=2)

  def test_unchanged_by_the_shift_itself(self):
    # a uniform sideways shift of the view must not change the curvature it measures (no feedback loop)
    a = DPCurveCentering(ModelConstants.X_IDXS)
    l0, r0 = lanes(0.002, 0.0)
    l1, r1 = lanes(0.002, 0.3)
    self.assertAlmostEqual(a.target_for(l0, r0, 0.9, 0.9, 25.0, 1.0), a.target_for(l1, r1, 0.9, 0.9, 25.0, 1.0), places=6)

  def test_off_cases(self):
    for kw in (dict(k=0.0001, v=30.0), dict(k=0.002, v=30.0, strength=0.0), dict(k=0.002, v=8.0),
               dict(k=0.002, v=30.0, active=False)):
      self.assertEqual(self.settle(DPCurveCentering(ModelConstants.X_IDXS), **kw), 0.0, kw)
    left, right = lanes(0.002)
    self.assertEqual(DPCurveCentering(ModelConstants.X_IDXS).target_for(left, right, 0.3, 0.9, 30.0, 1.0), 0.0)

  def test_fades_out_in_very_tight_curves(self):
    v = 30.0
    full = self.settle(DPCurveCentering(ModelConstants.X_IDXS), 1.7 / v ** 2, v)
    half = self.settle(DPCurveCentering(ModelConstants.X_IDXS), 2.15 / v ** 2, v)
    self.assertGreater(full, 0.25)
    self.assertAlmostEqual(half, 0.5 * min(DP_CURVE_MAX_M, 0.2 * (2.15 - 0.25)), places=3)
    self.assertEqual(self.settle(DPCurveCentering(ModelConstants.X_IDXS), 2.6 / v ** 2, v), 0.0)

  def test_cap_and_rate(self):
    cc = DPCurveCentering(ModelConstants.X_IDXS)
    self.assertAlmostEqual(self.settle(cc, 1.79 / 30.0 ** 2, 30.0, strength=2.0), DP_CURVE_MAX_M, places=6)
    cc = DPCurveCentering(ModelConstants.X_IDXS)
    left, right = lanes(1.79 / 30.0 ** 2)
    prev = 0.0
    for _ in range(100):
      off = cc.update(left, right, 0.9, 0.9, 30.0, 2.0, True)
      self.assertLessEqual(abs(off - prev), DP_CURVE_RATE_MS * cc.dt + 1e-12)
      prev = off
    # disengaging returns to 0 at the same rate
    self.assertEqual(self.settle(cc, 1.79 / 30.0 ** 2, 30.0, active=False, s=5.0), 0.0)


def lanes_curve_from(x0, k, half_width=1.8):
  """Straight lane that starts curving (curvature k) at x0 metres ahead."""
  x = np.array(ModelConstants.X_IDXS)
  centre = np.where(x > x0, 0.5 * k * (x - x0) ** 2, 0.0)
  return centre - half_width, centre + half_width


class TestDogPilotCurveCenteringFadeHold(unittest.TestCase):
  def test_holds_faded_through_the_ramp_pattern(self):
    # Oct 25 ramp: lateral acceleration wobbling between ~2.0 and ~2.6 m/s^2 inside one long tight curve
    v = 25.0
    cc = DPCurveCentering(ModelConstants.X_IDXS)
    for _ in range(40):
      cc.update(*lanes(2.6 / v ** 2), 0.9, 0.9, v, 1.5, True)
    offs = []
    for _ in range(4):
      for a in (2.0, 2.6):
        for _ in range(30):
          offs.append(cc.update(*lanes(a / v ** 2), 0.9, 0.9, v, 1.5, True))
    self.assertLess(max(abs(o) for o in offs), 0.06)   # was jumping between ~0 and ~0.28 m

  def test_returns_after_the_tight_curve(self):
    v = 30.0
    cc = DPCurveCentering(ModelConstants.X_IDXS)
    for _ in range(40):
      cc.update(*lanes(2.6 / v ** 2), 0.9, 0.9, v, 1.0, True)
    for _ in range(200):                               # 10 s of an ordinary 1.2 m/s^2 curve
      cc.update(*lanes(1.2 / v ** 2), 0.9, 0.9, v, 1.0, True)
    self.assertAlmostEqual(cc.offset, 0.2 * (1.2 - 0.25), places=2)


class TestDogPilotCurveCenteringEarly(unittest.TestCase):
  def test_off_by_default_and_identical_when_off(self):
    a, b = DPCurveCentering(ModelConstants.X_IDXS), DPCurveCentering(ModelConstants.X_IDXS)
    self.assertFalse(a.early)
    left, right = lanes(0.0015)
    for _ in range(100):
      self.assertEqual(a.update(left, right, 0.9, 0.9, 30.0, 1.0, True), b.update(left, right, 0.9, 0.9, 30.0, 1.0, True))

  def test_sees_the_curve_farther_ahead(self):
    left, right = lanes_curve_from(40.0, 0.0015)   # curve starts 40 m ahead: invisible to the near fit
    near, early = DPCurveCentering(ModelConstants.X_IDXS), DPCurveCentering(ModelConstants.X_IDXS)
    early.early = True
    self.assertAlmostEqual(near.target_for(left, right, 0.9, 0.9, 30.0, 1.0), 0.0, places=3)
    self.assertGreater(early.target_for(left, right, 0.9, 0.9, 30.0, 1.0), 0.1)

  def test_releases_slower_after_the_curve(self):
    curve, straight = lanes(0.0015), lanes(0.0)
    out = {}
    for mode in (False, True):
      cc = DPCurveCentering(ModelConstants.X_IDXS)
      cc.early = mode
      for _ in range(200):
        cc.update(*curve, 0.9, 0.9, 30.0, 1.0, True)
      peak = cc.offset
      for _ in range(20):                          # 1 s of straight road
        cc.update(*straight, 0.9, 0.9, 30.0, 1.0, True)
      out[mode] = peak - cc.offset
    self.assertLessEqual(out[True], DP_CURVE_RELEASE_MS * 1.0 + 1e-9)
    self.assertGreater(out[False], 2 * out[True])

  def test_reverses_at_full_rate(self):
    cc = DPCurveCentering(ModelConstants.X_IDXS)
    cc.early = True
    for _ in range(200):
      cc.update(*lanes(0.0015), 0.9, 0.9, 30.0, 1.0, True)
    start = cc.offset
    for _ in range(20):
      cc.update(*lanes(-0.0015), 0.9, 0.9, 30.0, 1.0, True)
    self.assertGreater(start - cc.offset, DP_CURVE_RELEASE_MS * 1.5)


if __name__ == "__main__":
  unittest.main()
