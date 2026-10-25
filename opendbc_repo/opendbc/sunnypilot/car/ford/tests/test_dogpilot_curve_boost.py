# DogPilot: high-demand boost (lateral_angle_ext _DP_BOOST_*, param DogPilotCurveBoost).
#
# Contract: identical to stock BluePilot when the request is below 1.8 m/s^2, below 40 mph, or the param is
# 1.0/missing; above that the path angle is scaled up to the param's multiplier at 2.5 m/s^2. The kappa_cmd
# that ford.h's shadow-curvature check reads is never changed, and the path-angle rate limit still applies.

import unittest

from opendbc.sunnypilot.car.ford.lateral_angle_ext import _DP_BOOST_MAX
from opendbc.sunnypilot.car.ford.tests.test_lateral_angle_ext import (
  _Actuators, _CC, _CS, _FakeParams, _ForcedDetector, _Harness, _Model, _explorer_cp)

MPH = 0.44704


def _model_steady(v, k):
  m = _Model()
  m.orientationRate.z = [v * k] * 33
  return m


def _settle(boost, v, k, steps=60):
  """Hold a steady curve k at speed v (measured = requested) until the path-angle rate limit has settled."""
  ext = _Harness(_explorer_cp())
  ext.human_turn_detector = _ForcedDetector(False)
  ext.dp_low_speed_lead = False
  params = {} if boost is None else {"DogPilotCurveBoost": str(boost).encode()}
  ext.update_angle_params(_FakeParams(params))
  ext.model = _model_steady(v, k)
  out = None
  for _ in range(steps):
    out = ext.update_angle_strategy(_CC(latActive=True), _CS(vEgoRaw=v, vEgo=v, yawRate=-k * v), _Actuators(curvature=k), ext.CP)
  return ext, out


class TestDogPilotCurveBoost(unittest.TestCase):
  def test_off_below_threshold(self):
    v = 70 * MPH                       # 0.0010 1/m -> ~1.0 m/s^2
    off, out_off = _settle(None, v, 0.0010)
    on, out_on = _settle(1.3, v, 0.0010)
    self.assertEqual(out_on.path_angle, out_off.path_angle)
    self.assertEqual(on.bp_dp_boost, 1.0)

  def test_full_boost_in_a_tight_highway_curve(self):
    v = 80 * MPH                       # 0.0022 1/m -> ~2.8 m/s^2
    off, out_off = _settle(None, v, 0.0022)
    on, out_on = _settle(1.3, v, 0.0022)
    self.assertAlmostEqual(on.bp_dp_boost, 1.3, places=6)
    self.assertAlmostEqual(out_on.path_angle / out_off.path_angle, 1.3, places=2)
    # the shadow curvature ford.h checks is untouched
    self.assertEqual(on.bp_kappa_cmd, off.bp_kappa_cmd)

  def test_ramps_between_thresholds(self):
    v = 80 * MPH
    k = 2.15 / (v * v)                 # halfway between 1.8 and 2.5 m/s^2
    on, _ = _settle(1.3, v, k)
    self.assertAlmostEqual(on.bp_dp_boost, 1.15, places=3)

  def test_off_below_40_mph(self):
    v = 35 * MPH                       # tight surface curve, ~2.4 m/s^2
    off, out_off = _settle(None, v, 0.0100)
    on, out_on = _settle(1.3, v, 0.0100)
    self.assertEqual(out_on.path_angle, out_off.path_angle)

  def test_param_limits(self):
    on, _ = _settle(9.0, 80 * MPH, 0.0022)
    self.assertEqual(on.dp_curve_boost, _DP_BOOST_MAX)
    off, _ = _settle(0.5, 80 * MPH, 0.0022)
    self.assertEqual(off.dp_curve_boost, 1.0)
    missing, _ = _settle(None, 80 * MPH, 0.0022)
    self.assertEqual(missing.bp_dp_boost, 1.0)

  def test_rate_limit_still_applies(self):
    v = 80 * MPH
    ext = _Harness(_explorer_cp())
    ext.human_turn_detector = _ForcedDetector(False)
    ext.dp_low_speed_lead = False
    ext.update_angle_params(_FakeParams({"DogPilotCurveBoost": b"1.5"}))
    ext.model = _model_steady(v, 0.0022)
    last = 0.0
    for _ in range(30):
      out = ext.update_angle_strategy(_CC(latActive=True), _CS(vEgoRaw=v, vEgo=v, yawRate=-0.0022 * v),
                                      _Actuators(curvature=0.0022), ext.CP)
      self.assertLessEqual(abs(out.path_angle - last), 0.009 + 1e-9)
      last = out.path_angle


if __name__ == "__main__":
  unittest.main()
