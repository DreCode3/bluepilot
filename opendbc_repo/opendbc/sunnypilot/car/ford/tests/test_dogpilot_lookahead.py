# DogPilot: curve look-ahead (lateral_angle_ext _DP_LOOKAHEAD_MAX_S, param DogPilotLookAhead, s, 0 = off).
#
# Contract: off (0 or missing) is identical to BluePilot. When set, the predicted-curvature look-up reads at least
# that much further ahead at every speed (stock's extra 0.10 s below 55 mph is kept when larger), so on a curve entry
# the path angle rises earlier; values are clipped to 0.30 s.

import unittest

from selfdrive.modeld.constants import ModelConstants

from opendbc.sunnypilot.car.ford.tests.test_lateral_angle_ext import (
  _Actuators, _CC, _CS, _FakeParams, _ForcedDetector, _Harness, _Model, _explorer_cp)

MPH = 0.44704


def _entry(v, params, slope=0.004, desired=0.0003, steps=40):
  """Curve entry: the model's yaw-rate plan ramps from straight to slope (1/m per s) ahead; the planner still asks little."""
  ext = _Harness(_explorer_cp())
  ext.human_turn_detector = _ForcedDetector(False)
  ext.update_angle_params(_FakeParams(params))
  out = None
  for _ in range(steps):
    m = _Model()
    m.orientationRate.z = [v * slope * t for t in ModelConstants.T_IDXS]
    ext.model = m
    out = ext.update_angle_strategy(_CC(latActive=True), _CS(vEgoRaw=v, vEgo=v, yawRate=-desired * v),
                                    _Actuators(curvature=desired), ext.CP)
  return ext, out.path_angle


def _steady(v, k, params, steps=60):
  ext = _Harness(_explorer_cp())
  ext.human_turn_detector = _ForcedDetector(False)
  ext.update_angle_params(_FakeParams(params))
  out = None
  for _ in range(steps):
    m = _Model()
    m.orientationRate.z = [v * k] * 33
    ext.model = m
    out = ext.update_angle_strategy(_CC(latActive=True), _CS(vEgoRaw=v, vEgo=v, yawRate=-k * v), _Actuators(curvature=k), ext.CP)
  return out.path_angle


class TestDogPilotLookAhead(unittest.TestCase):
  def test_off_is_bluepilot(self):
    for mph in (25, 45, 70):
      base_ext, base = _entry(mph * MPH, {})
      off_ext, off = _entry(mph * MPH, {"DogPilotLookAhead": b"0.0"})
      self.assertEqual(off, base, mph)
      self.assertEqual(off_ext.bp_curvature_lookup_time, base_ext.bp_curvature_lookup_time, mph)

  def test_highway_reads_further_ahead(self):
    base_ext, _ = _entry(70 * MPH, {})
    ext, _ = _entry(70 * MPH, {"DogPilotLookAhead": b"0.25"})
    self.assertAlmostEqual(ext.bp_curvature_lookup_time - base_ext.bp_curvature_lookup_time, 0.25, places=6)

  def test_entry_commands_more_earlier(self):
    for mph in (30, 45, 60, 75):
      _, base = _entry(mph * MPH, {})
      _, la = _entry(mph * MPH, {"DogPilotLookAhead": b"0.25"})
      self.assertGreater(abs(la), abs(base), mph)

  def test_low_speed_keeps_stock_extra_when_larger(self):
    base_ext, _ = _entry(20 * MPH, {})
    ext, _ = _entry(20 * MPH, {"DogPilotLookAhead": b"0.05"})
    self.assertAlmostEqual(ext.bp_curvature_lookup_time, base_ext.bp_curvature_lookup_time, places=9)

  def test_clipped(self):
    a, _ = _entry(70 * MPH, {"DogPilotLookAhead": b"0.30"})
    b, _ = _entry(70 * MPH, {"DogPilotLookAhead": b"0.9"})
    self.assertAlmostEqual(a.bp_curvature_lookup_time, b.bp_curvature_lookup_time, places=9)

  def test_steady_curve_unchanged(self):
    for mph in (35, 70):
      base = _steady(mph * MPH, 0.002, {})
      la = _steady(mph * MPH, 0.002, {"DogPilotLookAhead": b"0.25"})
      self.assertAlmostEqual(la, base, places=6, msg=mph)


if __name__ == "__main__":
  unittest.main()
