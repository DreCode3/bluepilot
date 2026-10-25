# DogPilot: lane changes use stock BluePilot's low-speed gain (lateral_angle_ext _DP_LC_*).
#
# Contract: while laneChangeState is 1-3, the Low Speed Adjustment Factor is treated as at most 1.0 (easing in
# over 0.5 s, out over 1.5 s after the lane change ends); curves outside lane changes keep the full factor;
# at and above 60 mph, where only the high-speed gain applies, nothing changes.

import unittest
from unittest import mock

from opendbc.sunnypilot.car.ford import lateral_angle_ext
from opendbc.sunnypilot.car.ford.tests.test_lateral_angle_ext import (
  _Actuators, _CC, _CS, _FakeParams, _ForcedDetector, _Harness, _Model, _explorer_cp)

MPH = 0.44704


def _run(v, k, lane_change_states, low_factor=b"1.5", dogpilot=True):
  if not dogpilot:  # without the DogPilot change: the lane-change weight never rises
    with mock.patch.object(lateral_angle_ext, "_DP_LC_IN_S", 1e12):
      return _run(v, k, lane_change_states, low_factor)
  ext = _Harness(_explorer_cp())
  ext.human_turn_detector = _ForcedDetector(False)
  ext.dp_low_speed_lead = False
  ext.update_angle_params(_FakeParams({"FordLowSpeedFactor_ang": low_factor}))
  outs = []
  for lcs in lane_change_states:
    m = _Model(lane_change_state=lcs)
    m.orientationRate.z = [v * k] * 33
    m.meta.laneChangeDirection = 1 if k < 0 else 2
    ext.model = m
    outs.append(ext.update_angle_strategy(_CC(latActive=True), _CS(vEgoRaw=v, vEgo=v, yawRate=-k * v),
                                          _Actuators(curvature=k), ext.CP).path_angle)
  return ext, outs


class TestDogPilotLaneChangeGain(unittest.TestCase):
  def test_low_speed_lane_change_uses_stock_gain(self):
    v, k = 32 * MPH, -0.0037            # the Oct 25 lane change: ~0.0037 1/m at 32 mph
    _, before = _run(v, k, [2] * 40, dogpilot=False)
    _, lc = _run(v, k, [2] * 40)
    _, stock = _run(v, k, [2] * 40, low_factor=b"1.0", dogpilot=False)
    self.assertLess(abs(lc[-1]), abs(before[-1]) * 0.8)
    self.assertAlmostEqual(lc[-1], stock[-1], places=6)

  def test_curves_unchanged(self):
    v, k = 32 * MPH, -0.0037
    _, before = _run(v, k, [0] * 40, dogpilot=False)
    _, now = _run(v, k, [0] * 40)
    self.assertEqual(now, before)

  def test_eases_back_after_the_lane_change(self):
    v, k = 32 * MPH, -0.0037
    _, outs = _run(v, k, [2] * 40 + [0] * 60)
    _, normal = _run(v, k, [0] * 40)
    # 0.5 s after the end it is between the two; 1.5 s after, back to the normal curve gain
    self.assertGreater(abs(outs[50]), abs(outs[39]))
    self.assertLess(abs(outs[50]), abs(normal[-1]))
    self.assertAlmostEqual(outs[-1], normal[-1], places=6)

  def test_highway_unchanged(self):
    v, k = 65 * MPH, -0.0008
    _, before = _run(v, k, [2] * 40, dogpilot=False)
    _, lc = _run(v, k, [2] * 40)
    self.assertAlmostEqual(lc[-1], before[-1], places=9)

  def test_factor_at_or_below_one_unchanged(self):
    v, k = 32 * MPH, -0.0037
    _, before = _run(v, k, [2] * 40, low_factor=b"0.9", dogpilot=False)
    _, lc = _run(v, k, [2] * 40, low_factor=b"0.9")
    self.assertAlmostEqual(lc[-1], before[-1], places=9)


if __name__ == "__main__":
  unittest.main()
