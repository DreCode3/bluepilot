# DogPilot: low-speed curve anticipation (lateral_angle_ext _DP_LEAD_*, param DogPilotLowSpeedLead).
#
# Contract: identical to stock BluePilot at and above 55 mph and below 12 mph; between them, looks further
# ahead and weights the model's predicted curvature more, so a curve the model sees coming is commanded
# earlier. The param turns it off; a missing param means on.

import unittest

from opendbc.sunnypilot.car.ford.tests.test_lateral_angle_ext import (
  _Actuators, _CC, _CS, _FakeParams, _ForcedDetector, _Harness, _Model, _explorer_cp)

MPH = 0.44704


def _model_curve_ahead(v, k_now, k_later):
  """Model whose yaw-rate plan ramps from k_now (now) to k_later (by 2 s): a curve coming up."""
  m = _Model()
  t_idxs = [10.0 * (i / 32) ** 2 for i in range(33)]
  m.orientationRate.z = [v * (k_now + (k_later - k_now) * min(t / 2.0, 1.0)) for t in t_idxs]
  return m


def _step(lead, v, k_now=0.002, k_later=0.010, params=None):
  ext = _Harness(_explorer_cp())
  ext.human_turn_detector = _ForcedDetector(False)
  ext.dp_low_speed_lead = lead
  if params is not None:
    ext.update_angle_params(_FakeParams(params))
  ext.model = _model_curve_ahead(v, k_now, k_later)
  # measured curvature = current request, so the deviation clip is not what's being tested
  cs = _CS(vEgoRaw=v, vEgo=v, yawRate=-k_now * v)
  act = _Actuators(curvature=k_now)
  out = ext.update_angle_strategy(_CC(latActive=True), cs, act, ext.CP)
  return ext, out


class TestDogPilotLowSpeedLead(unittest.TestCase):
  def test_identical_to_stock_at_and_above_55_mph(self):
    for mph in (55.0, 60.0, 75.0):
      v = mph * MPH
      on, out_on = _step(True, v)
      off, out_off = _step(False, v)
      self.assertEqual(out_on.path_angle, out_off.path_angle, mph)
      self.assertEqual(on.bp_curvature_lookup_time, off.bp_curvature_lookup_time, mph)

  def test_identical_to_stock_below_12_mph(self):
    for mph in (5.0, 10.0, 12.0):
      v = mph * MPH
      on, out_on = _step(True, v)
      off, out_off = _step(False, v)
      self.assertAlmostEqual(out_on.path_angle, out_off.path_angle, places=9, msg=mph)

  def test_looks_further_ahead_and_commands_curve_earlier_at_30_mph(self):
    v = 30.0 * MPH
    on, out_on = _step(True, v)
    off, out_off = _step(False, v)
    self.assertGreater(on.bp_curvature_lookup_time, off.bp_curvature_lookup_time + 0.5)
    self.assertGreater(out_on.path_angle, out_off.path_angle)

  def test_param_off_restores_stock(self):
    v = 30.0 * MPH
    off_by_param, out_param = _step(True, v, params={"DogPilotLowSpeedLead": b"0"})
    self.assertFalse(off_by_param.dp_low_speed_lead)
    _, out_off = _step(False, v)
    self.assertEqual(out_param.path_angle, out_off.path_angle)

  def test_missing_param_means_on(self):
    ext, _ = _step(True, 30.0 * MPH, params={})
    self.assertTrue(ext.dp_low_speed_lead)


if __name__ == "__main__":
  unittest.main()
