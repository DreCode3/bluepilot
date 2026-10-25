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


def _drive(ext, v, ks):
  """Feed a sequence of requested curvatures (one per 20 Hz step); return bp_dp_boost after each."""
  out = []
  for k in ks:
    ext.model = _model_steady(v, k)
    ext.update_angle_strategy(_CC(latActive=True), _CS(vEgoRaw=v, vEgo=v, yawRate=-k * v), _Actuators(curvature=k), ext.CP)
    out.append(ext.bp_dp_boost)
  return out


class TestDogPilotCurveBoostHold(unittest.TestCase):
  def setUp(self):
    self.v = 55 * MPH
    self.ext = _Harness(_explorer_cp())
    self.ext.human_turn_detector = _ForcedDetector(False)
    self.ext.dp_low_speed_lead = False
    self.ext.update_angle_params(_FakeParams({"DogPilotCurveBoost": b"1.3"}))
    self.k = lambda a: -a / self.v ** 2       # left curve with lateral acceleration a

  def test_rises_in_half_a_second(self):
    b = _drive(self.ext, self.v, [self.k(2.8)] * 20)
    self.assertLess(b[4], 1.2)                 # 0.25 s
    self.assertAlmostEqual(b[11], 1.3, places=6)

  def test_holds_through_the_ramp_pattern(self):
    # Oct 25 ramp: request alternating ~2.7 and ~1.3 m/s^2 every ~1.5 s while the curve continued
    b = _drive(self.ext, self.v, ([self.k(2.7)] * 30 + [self.k(1.3)] * 30) * 4)
    self.assertGreater(min(b[30:]), 1.2)        # never drops back to ~1.0 during the dips
    self.assertLess(max(b[30:]) - min(b[30:]), 0.1)

  def test_releases_quickly_after_the_curve(self):
    _drive(self.ext, self.v, [self.k(2.8)] * 40)
    b = _drive(self.ext, self.v, [self.k(0.3)] * 20)
    self.assertLess(b[-1], 1.0 + 1e-9 + 0.0)   # back to 1.0 within 1 s

  def test_restarts_on_engagement(self):
    _drive(self.ext, self.v, [self.k(2.8)] * 40)
    self.ext.update_angle_strategy(_CC(latActive=False), _CS(vEgoRaw=self.v, vEgo=self.v), _Actuators(curvature=0.0), self.ext.CP)
    self.assertEqual(self.ext.dp_boost_level, 0.0)


if __name__ == "__main__":
  unittest.main()
