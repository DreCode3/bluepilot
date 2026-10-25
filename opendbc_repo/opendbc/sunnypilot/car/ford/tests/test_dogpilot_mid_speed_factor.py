# DogPilot: separate mid-speed factor (lateral_angle_ext _DP_MID_*, param DogPilotMidSpeedFactor, 0 = off).
#
# Contract: off (0 or missing) is identical to BluePilot. When set, below 30 mph nothing changes (the Low Speed
# Adjustment Factor still rules); above 40 mph the gain is exactly what BluePilot gives with the low-speed factor set
# to the mid-speed value; 30-40 mph hands over smoothly; 60 mph and up is unchanged; lane changes cap it at 1.0.

import unittest

from opendbc.sunnypilot.car.ford.tests.test_lateral_angle_ext import (
  _Actuators, _CC, _CS, _FakeParams, _ForcedDetector, _Harness, _Model, _explorer_cp)

MPH = 0.44704


def _pa(v, k, params, lane_change_state=0, steps=60):
  ext = _Harness(_explorer_cp())
  ext.human_turn_detector = _ForcedDetector(False)
  ext.dp_low_speed_lead = False
  ext.update_angle_params(_FakeParams(params))
  out = None
  for _ in range(steps):
    m = _Model(lane_change_state=lane_change_state)
    m.orientationRate.z = [v * k] * 33
    m.meta.laneChangeDirection = 1 if k < 0 else 2
    ext.model = m
    out = ext.update_angle_strategy(_CC(latActive=True), _CS(vEgoRaw=v, vEgo=v, yawRate=-k * v), _Actuators(curvature=k), ext.CP)
  return ext, out.path_angle


class TestDogPilotMidSpeedFactor(unittest.TestCase):
  def test_off_is_bluepilot(self):
    for mph in (25, 45, 55):
      _, base = _pa(mph * MPH, 0.002, {"FordLowSpeedFactor_ang": b"1.25"})
      _, off = _pa(mph * MPH, 0.002, {"FordLowSpeedFactor_ang": b"1.25", "DogPilotMidSpeedFactor": b"0.0"})
      self.assertEqual(off, base, mph)

  def test_city_unchanged(self):
    _, base = _pa(25 * MPH, 0.004, {"FordLowSpeedFactor_ang": b"1.25"})
    _, mid = _pa(25 * MPH, 0.004, {"FordLowSpeedFactor_ang": b"1.25", "DogPilotMidSpeedFactor": b"1.5"})
    self.assertEqual(mid, base)

  def test_above_40_mph_matches_bluepilot_at_that_factor(self):
    for mph in (45, 50, 55):
      _, want = _pa(mph * MPH, 0.002, {"FordLowSpeedFactor_ang": b"1.5"})
      _, got = _pa(mph * MPH, 0.002, {"FordLowSpeedFactor_ang": b"1.25", "DogPilotMidSpeedFactor": b"1.5"})
      self.assertAlmostEqual(got, want, places=9, msg=mph)

  def test_hands_over_between_30_and_40_mph(self):
    _, low = _pa(35 * MPH, 0.002, {"FordLowSpeedFactor_ang": b"1.25"})
    _, high = _pa(35 * MPH, 0.002, {"FordLowSpeedFactor_ang": b"1.5"})
    _, mid = _pa(35 * MPH, 0.002, {"FordLowSpeedFactor_ang": b"1.25", "DogPilotMidSpeedFactor": b"1.5"})
    self.assertGreater(mid, low)
    self.assertLess(mid, high)

  def test_highway_unchanged(self):
    _, base = _pa(65 * MPH, 0.0012, {"FordLowSpeedFactor_ang": b"1.25"})
    _, mid = _pa(65 * MPH, 0.0012, {"FordLowSpeedFactor_ang": b"1.25", "DogPilotMidSpeedFactor": b"1.5"})
    self.assertAlmostEqual(mid, base, places=9)

  def test_lane_changes_capped(self):
    _, stock = _pa(50 * MPH, -0.002, {"FordLowSpeedFactor_ang": b"1.0"}, lane_change_state=2)
    _, mid = _pa(50 * MPH, -0.002, {"FordLowSpeedFactor_ang": b"1.25", "DogPilotMidSpeedFactor": b"1.5"}, lane_change_state=2)
    self.assertAlmostEqual(mid, stock, places=9)

  def test_param_range(self):
    ext, _ = _pa(50 * MPH, 0.002, {"DogPilotMidSpeedFactor": b"9"})
    self.assertEqual(ext.dp_mid_speed_factor, 1.5)
    ext, _ = _pa(50 * MPH, 0.002, {"DogPilotMidSpeedFactor": b"0.05"})
    self.assertEqual(ext.dp_mid_speed_factor, 0.5)
    ext, _ = _pa(50 * MPH, 0.002, {})
    self.assertEqual(ext.dp_mid_speed_factor, 0.0)


if __name__ == "__main__":
  unittest.main()
