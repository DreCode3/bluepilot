# DogPilot: highway plan smoothing (dogpilot_plan_smooth.py, param DogPilotPlanSmooth, off by default).
#
# Contract: off is BluePilot; on, nothing changes below 45 mph, in lane changes, in tight curves (plan above 2.5 m/s^2)
# or on a steady plan; on a wobbling highway plan the path angle changes, but never by more than 0.15 m/s^2.

import unittest

import numpy as np

from opendbc.sunnypilot.car.ford.dogpilot_plan_smooth import CAP_MS2, T_IDXS, curvature_from_plan
from opendbc.sunnypilot.car.ford.tests.test_lateral_angle_ext import (
  _Actuators, _CC, _CS, _FakeParams, _ForcedDetector, _Harness, _Model, _explorer_cp)

MPH = 0.44704


def _plan(v, a_lat, wobble, i):
  """Turn-rate plan: a steady a_lat curve plus a wobble that moves along the horizon from frame to frame."""
  return [(a_lat + wobble * np.sin(2.0 * t + 0.4 * i)) / v for t in T_IDXS]


def _run(v, a_lat, wobble, smooth, lane_change_state=0, steps=120):
  ext = _Harness(_explorer_cp())
  ext.human_turn_detector = _ForcedDetector(False)
  ext.update_angle_params(_FakeParams({"DogPilotPlanSmooth": smooth} if smooth is not None else {}))
  out = []
  for i in range(steps):
    z = _plan(v, a_lat, wobble, i)
    k = curvature_from_plan(z, v, 0.3)
    m = _Model(lane_change_state=lane_change_state)
    m.orientationRate.z = z
    m.meta.laneChangeDirection = 2
    ext.model = m
    r = ext.update_angle_strategy(_CC(latActive=True), _CS(vEgoRaw=v, vEgo=v, yawRate=-k * v), _Actuators(curvature=k), ext.CP)
    out.append(r.path_angle)
  return np.array(out)


class TestDogPilotPlanSmooth(unittest.TestCase):
  def test_off_is_bluepilot(self):
    np.testing.assert_array_equal(_run(65 * MPH, 0.5, 0.3, False), _run(65 * MPH, 0.5, 0.3, None))

  def test_below_45_mph_unchanged(self):
    for mph in (25, 40, 44):
      np.testing.assert_array_equal(_run(mph * MPH, 0.5, 0.3, True), _run(mph * MPH, 0.5, 0.3, False), mph)

  def test_steady_plan_unchanged(self):
    np.testing.assert_allclose(_run(65 * MPH, 0.8, 0.0, True), _run(65 * MPH, 0.8, 0.0, False), atol=1e-9)

  def test_tight_curve_unchanged(self):
    np.testing.assert_array_equal(_run(60 * MPH, 2.8, 0.3, True), _run(60 * MPH, 2.8, 0.3, False))

  def test_lane_change_unchanged(self):
    np.testing.assert_array_equal(_run(65 * MPH, 0.5, 0.3, True, lane_change_state=1),
                                  _run(65 * MPH, 0.5, 0.3, False, lane_change_state=1))

  def test_wobble_calmed_and_capped(self):
    v = 65 * MPH
    on, off = _run(v, 0.5, 0.4, True), _run(v, 0.5, 0.4, False)
    diff = np.abs(on - off) * v
    self.assertGreater(diff.max(), 0.01)
    self.assertLessEqual(diff.max(), CAP_MS2 + 1e-9)
    self.assertLess(np.std(np.diff(on[20:])), np.std(np.diff(off[20:])))


if __name__ == "__main__":
  unittest.main()
