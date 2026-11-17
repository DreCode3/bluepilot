# DogPilot: the camera offset must shift what the model sees sideways by the same amount at every distance,
# in the documented direction (+ = model centre moves left, so the car moves left), and be exact identity at 0.
import unittest

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.model import get_warp_matrix, medmodel_intrinsics, sbigmodel_intrinsics
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.dogpilot_offset import (DP_CURVE_FADE_MS2, DP_CURVE_MAX_M, DP_CURVE_RATE_MS, DPCurveCentering,
                                                        DPOffsetCurveFade, dp_faded_offset, dp_offset_for_speed,
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
    f0, f1 = DP_CURVE_FADE_MS2
    mid = (f0 + f1) / 2
    full = self.settle(DPCurveCentering(ModelConstants.X_IDXS), (f0 - 0.1) / v ** 2, v)
    half = self.settle(DPCurveCentering(ModelConstants.X_IDXS), mid / v ** 2, v)
    self.assertGreater(full, 0.25)
    self.assertAlmostEqual(half, 0.5 * min(DP_CURVE_MAX_M, 0.2 * (mid - 0.25)), places=3)
    self.assertEqual(self.settle(DPCurveCentering(ModelConstants.X_IDXS), (f1 + 0.1) / v ** 2, v), 0.0)

  def test_keeps_the_shift_in_tight_highway_curves(self):
    # Oct 25: 2.4-2.8 m/s^2 highway curves; the old 1.8-2.5 fade removed the shift at entry and the car cut 0.9 m
    v = 35.0
    self.assertAlmostEqual(self.settle(DPCurveCentering(ModelConstants.X_IDXS), 2.6 / v ** 2, v, strength=1.5), DP_CURVE_MAX_M, places=6)

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


class TestDogPilotCurveCenteringFadeHold(unittest.TestCase):
  def test_holds_faded_through_the_ramp_pattern(self):
    # a curve wobbling in and out of the fade band (Oct 25 ramp pattern, shifted to the current band)
    v = 25.0
    f0, f1 = DP_CURVE_FADE_MS2
    cc = DPCurveCentering(ModelConstants.X_IDXS)
    for _ in range(40):
      cc.update(*lanes((f1 + 0.1) / v ** 2), 0.9, 0.9, v, 1.5, True)
    offs = []
    for _ in range(4):
      for a in (f0 + 0.2, f1 + 0.1):
        for _ in range(30):
          offs.append(cc.update(*lanes(a / v ** 2), 0.9, 0.9, v, 1.5, True))
    self.assertLess(max(abs(o) for o in offs), 0.06)   # was jumping between ~0 and ~0.28 m

  def test_returns_after_the_tight_curve(self):
    v = 30.0
    cc = DPCurveCentering(ModelConstants.X_IDXS)
    for _ in range(40):
      cc.update(*lanes((DP_CURVE_FADE_MS2[1] + 0.1) / v ** 2), 0.9, 0.9, v, 1.0, True)
    for _ in range(200):                               # 10 s of an ordinary 1.2 m/s^2 curve
      cc.update(*lanes(1.2 / v ** 2), 0.9, 0.9, v, 1.0, True)
    self.assertAlmostEqual(cc.offset, 0.2 * (1.2 - 0.25), places=2)


class TestDogPilotCurveCenteringLaneChangeHold(unittest.TestCase):
  def test_shift_holds_during_a_lane_change_then_resumes(self):
    v = 30.0
    cc = DPCurveCentering(ModelConstants.X_IDXS)
    for _ in range(100):
      cc.update(*lanes(1.3 / v ** 2), 0.9, 0.9, v, 1.0, True)
    before = cc.offset
    self.assertGreater(before, 0.15)
    for _ in range(100):                      # 5 s lane change onto a straight: shift must not move
      self.assertEqual(cc.update(*lanes(0.0), 0.9, 0.9, v, 1.0, True, hold=True), before)
    after = [cc.update(*lanes(0.0), 0.9, 0.9, v, 1.0, True) for _ in range(40)]
    self.assertLess(after[-1], before)          # releases normally once the lane change ends
    self.assertLessEqual(before - after[0], DP_CURVE_RATE_MS * cc.dt + 1e-12)

  def test_disengaging_still_releases_during_a_lane_change(self):
    cc = DPCurveCentering(ModelConstants.X_IDXS)
    for _ in range(100):
      cc.update(*lanes(1.3 / 900.0), 0.9, 0.9, 30.0, 1.0, True)
    before = cc.offset
    cc.update(*lanes(0.0), 0.9, 0.9, 30.0, 1.0, False, hold=True)
    self.assertLess(cc.offset, before)


def lanes_w(k, width):
  return lanes(k, half_width=width / 2)


class TestDogPilotCurveCenteringWideCap(unittest.TestCase):
  def settle(self, width, wide, a=2.5, v=30.0, s=10.0):
    cc = DPCurveCentering(ModelConstants.X_IDXS)
    cc.wide_cap = wide
    for _ in range(int(s / cc.dt)):
      cc.update(*lanes_w(a / v ** 2, width), 0.9, 0.9, v, 1.5, True)
    return cc.offset

  def test_off_keeps_the_fixed_cap(self):
    for width in (3.0, 3.7, 4.2):
      self.assertAlmostEqual(self.settle(width, False), DP_CURVE_MAX_M, places=6)

  def test_interstate_lane_allows_more(self):
    self.assertAlmostEqual(self.settle(3.4, True), 0.50, places=2)   # (3.4 - 2.0) / 2 - 0.20
    self.assertAlmostEqual(self.settle(3.26, True), 0.43, places=2)

  def test_narrower_lane_same_or_less_than_today(self):
    self.assertAlmostEqual(self.settle(3.1, True), 0.35, places=2)
    self.assertAlmostEqual(self.settle(3.0, True), 0.30, places=2)
    self.assertAlmostEqual(self.settle(2.7, True), 0.20, places=2)

  def test_never_above_the_ceiling(self):
    self.assertLessEqual(self.settle(4.5, True), 0.50 + 1e-9)


MPH = 0.44704


class TestDogPilotOffsetBySpeed(unittest.TestCase):
  def test_off_is_static(self):
    for mph in (10, 50, 57.5, 65, 80):
      self.assertEqual(dp_offset_for_speed(-0.20, -0.05, False, mph * MPH), -0.20)

  def test_schedule(self):
    self.assertAlmostEqual(dp_offset_for_speed(-0.20, -0.05, True, 0.0), -0.05)
    self.assertAlmostEqual(dp_offset_for_speed(-0.20, -0.05, True, 50 * MPH), -0.05)
    self.assertAlmostEqual(dp_offset_for_speed(-0.20, -0.05, True, 57.5 * MPH), -0.125)
    self.assertAlmostEqual(dp_offset_for_speed(-0.20, -0.05, True, 65 * MPH), -0.20)
    self.assertAlmostEqual(dp_offset_for_speed(-0.20, -0.05, True, 80 * MPH), -0.20)



class TestDogPilotOffsetCurveFade(unittest.TestCase):
  """Static camera offset faded in curves (DogPilotOffsetCurveFade): full on straights, (1 - F) of it in curves, held
  through short gaps (S-bends), lane changes and unusable lane lines, released after the road straightens."""
  V = 25.0

  def run_k(self, cc, fade, k, s, v=None, active=True, prob=0.9, hold=False):
    v = self.V if v is None else v
    left, right = lanes(k)
    out = []
    for _ in range(int(round(s / cc.dt))):
      cc.update(left, right, prob, prob, v, 1.5, active, hold=hold)
      out.append(fade.update(cc.k, v, active, hold=hold))
    return out

  def setUp(self):
    self.cc = DPCurveCentering(ModelConstants.X_IDXS)
    self.fade = DPOffsetCurveFade(self.cc.dt)

  def test_off_leaves_the_offset_alone(self):
    for level in (0.0, 0.5, 1.0):
      self.assertEqual(dp_faded_offset(-0.12, 0.0, level), -0.12)
    self.assertAlmostEqual(dp_faded_offset(-0.12, 0.6, 1.0), -0.048)
    self.assertAlmostEqual(dp_faded_offset(-0.20, 1.7, 1.0), 0.0)   # F clipped to 1

  def test_straight_keeps_the_full_offset(self):
    self.assertEqual(max(self.run_k(self.cc, self.fade, 0.0, 10.0)), 0.0)

  def test_gentle_curve_below_threshold(self):
    # 0.0005 1/m at 25 m/s = 0.31 m/s^2 < 0.4
    self.assertEqual(max(self.run_k(self.cc, self.fade, 0.0005, 10.0)), 0.0)

  def test_curve_fades_within_a_second(self):
    out = self.run_k(self.cc, self.fade, 0.002, 3.0)   # 1.25 m/s^2
    self.assertAlmostEqual(out[int(1.0 / self.cc.dt)], 1.0, places=6)
    self.assertEqual(self.cc.k is not None, True)

  def test_holds_then_releases_after_the_curve(self):
    self.run_k(self.cc, self.fade, 0.002, 3.0)
    out = self.run_k(self.cc, self.fade, 0.0, 8.0)
    self.assertEqual(out[int(1.4 / self.cc.dt)], 1.0)                        # still held at 1.4 s
    self.assertAlmostEqual(out[int(3.5 / self.cc.dt)], 0.5, delta=0.03)      # 1.5 s hold + 2 s at 0.25/s
    self.assertEqual(out[-1], 0.0)

  def test_holds_through_an_s_bend(self):
    self.run_k(self.cc, self.fade, 0.002, 3.0)
    gap = self.run_k(self.cc, self.fade, 0.0, 2.5)
    self.assertGreaterEqual(min(gap), 0.73)   # 1.5 s hold + 1 s at 0.25/s
    out = self.run_k(self.cc, self.fade, -0.002, 2.0)
    self.assertAlmostEqual(out[-1], 1.0)

  def test_held_during_a_lane_change_and_without_lines(self):
    self.run_k(self.cc, self.fade, 0.002, 3.0)
    self.assertEqual(min(self.run_k(self.cc, self.fade, 0.0, 4.0, hold=True)), 1.0)
    self.assertEqual(min(self.run_k(self.cc, self.fade, 0.0, 4.0, prob=0.2)), 1.0)

  def test_released_with_lateral_off_or_slow(self):
    self.run_k(self.cc, self.fade, 0.002, 3.0)
    out = self.run_k(self.cc, self.fade, 0.002, 2.0, active=False)
    self.assertAlmostEqual(out[-1], 0.5, delta=0.03)                          # falls at 0.25/s straight away
    fade = DPOffsetCurveFade(self.cc.dt)
    cc = DPCurveCentering(ModelConstants.X_IDXS)
    self.assertEqual(max(self.run_k(cc, fade, 0.02, 3.0, v=8.0)), 0.0)      # below 20 mph: never fades


if __name__ == "__main__":
  unittest.main()
