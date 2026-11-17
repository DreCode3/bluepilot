# DogPilot: exit gain hold (lateral_angle_ext _DP_EXIT_*, param DogPilotExitGainHold, s, 0 = off).
#
# Contract: off (0 or missing) is identical to BluePilot. When set, after a real curve (request above 0.6 m/s^2 for 1 s)
# the low/high gain blend keeps the in-curve gain while the request unwinds through 0.0007-0.001 1/m, coming down over
# the set time, below 55 mph (fading out by 65); nothing changes inside the curve, on straights with no curve before
# them, on the highway, or during a lane change.

import unittest

from opendbc.sunnypilot.car.ford.tests.test_lateral_angle_ext import (
  _Actuators, _CC, _CS, _FakeParams, _ForcedDetector, _Harness, _Model, _explorer_cp)

MPH = 0.44704
DT = 0.05


def _drive(v, ks, params, lane_change_at=()):
  """Feed a request sequence ks (1/m) with the car following it exactly; return the path angles sent."""
  ext = _Harness(_explorer_cp())
  ext.human_turn_detector = _ForcedDetector(False)
  ext.update_angle_params(_FakeParams(params))
  out = []
  for i, k in enumerate(ks):
    m = _Model()
    m.orientationRate.z = [v * k] * 33
    if i in lane_change_at:
      m.meta.laneChangeState = 2
    ext.model = m
    r = ext.update_angle_strategy(_CC(latActive=True), _CS(vEgoRaw=v, vEgo=v, yawRate=-k * v), _Actuators(curvature=k), ext.CP)
    out.append(r.path_angle)
  return out


def _curve_then_exit(v, k_curve, curve_s=3.0, unwind_s=2.0, tail_s=4.0):
  """Straight 1 s, a curve held curve_s, unwinding linearly to 0.0004 1/m over unwind_s, then 0.0004 for tail_s."""
  def n(s):
    return int(round(s / DT))
  ks = [0.0] * n(1.0) + [k_curve] * n(curve_s)
  ks += [k_curve + (0.0004 - k_curve) * (i + 1) / n(unwind_s) for i in range(n(unwind_s))]
  ks += [0.0004] * n(tail_s)
  return ks


class TestDogPilotExitGainHold(unittest.TestCase):
  def test_off_is_bluepilot(self):
    ks = _curve_then_exit(30 * MPH, 0.004)
    base = _drive(30 * MPH, ks, {})
    for raw in (b"0.0", b"-1"):
      self.assertEqual(_drive(30 * MPH, ks, {"DogPilotExitGainHold": raw}), base)

  def test_exit_tail_gets_more_command_in_town(self):
    v = 30 * MPH
    ks = _curve_then_exit(v, 0.004)
    base = _drive(v, ks, {})
    hold = _drive(v, ks, {"DogPilotExitGainHold": b"3.0"})
    tail = range(int(6.0 / DT), int(6.0 / DT) + 20)   # first second at 0.0004 1/m after the unwind
    for i in tail:
      self.assertGreater(abs(hold[i]), abs(base[i]) * 1.15, i)

  def test_curve_itself_unchanged(self):
    v = 30 * MPH
    ks = _curve_then_exit(v, 0.004)
    base = _drive(v, ks, {})
    hold = _drive(v, ks, {"DogPilotExitGainHold": b"3.0"})
    for i in range(int(1.0 / DT), int(4.0 / DT)):
      self.assertAlmostEqual(hold[i], base[i], places=9)

  def test_released_after_the_hold(self):
    v = 30 * MPH
    ks = _curve_then_exit(v, 0.004, tail_s=8.0)
    base = _drive(v, ks, {})
    hold = _drive(v, ks, {"DogPilotExitGainHold": b"2.0"})
    for i in range(len(ks) - 20, len(ks)):
      self.assertAlmostEqual(hold[i], base[i], places=9)

  def test_straight_road_wiggles_unchanged(self):
    v = 30 * MPH
    ks = [0.0012 * ((i // 10) % 2) for i in range(200)]   # small corrections across the gain band, no curve before
    self.assertEqual(_drive(v, ks, {"DogPilotExitGainHold": b"3.0"}), _drive(v, ks, {}))

  def test_short_curve_does_not_arm(self):
    v = 30 * MPH
    ks = _curve_then_exit(v, 0.004, curve_s=0.6, unwind_s=0.2)   # 0.6 s above 0.6 m/s^2 (plus the unwind's first frames)
    self.assertEqual(_drive(v, ks, {"DogPilotExitGainHold": b"3.0"}), _drive(v, ks, {}))

  def test_not_during_a_lane_change(self):
    v = 30 * MPH
    ks = _curve_then_exit(v, 0.004)
    lc = set(range(int(5.0 / DT), len(ks)))
    base = _drive(v, ks, {}, lane_change_at=lc)
    hold = _drive(v, ks, {"DogPilotExitGainHold": b"3.0"}, lane_change_at=lc)
    for i in range(int(6.0 / DT), len(ks)):
      self.assertAlmostEqual(hold[i], base[i], places=9)

  def test_highway_unchanged(self):
    for mph in (65, 75):
      ks = _curve_then_exit(mph * MPH, 0.0015)
      car = {"FordHighSpeedDampening_ang": b"0.85"}
      self.assertEqual(_drive(mph * MPH, ks, dict(car, DogPilotExitGainHold=b"3.0")), _drive(mph * MPH, ks, car), mph)

  def test_fades_out_between_55_and_65_mph(self):
    tail = range(int(4.5 / DT), int(4.5 / DT) + 20)   # first second after a 0.5 s unwind
    gains = []
    for mph in (50, 60):
      ks = _curve_then_exit(mph * MPH, 0.004 * (30 / mph) ** 2, unwind_s=0.5)
      car = {"FordHighSpeedDampening_ang": b"0.85"}   # as on the car: gives a low/high step at speed
      base = _drive(mph * MPH, ks, car)
      hold = _drive(mph * MPH, ks, dict(car, DogPilotExitGainHold=b"3.0"))
      gains.append(max(abs(hold[i]) / max(abs(base[i]), 1e-9) for i in tail))
    self.assertGreater(gains[0], gains[1])
    self.assertGreater(gains[1], 1.0)

  def test_clipped(self):
    v = 30 * MPH
    ks = _curve_then_exit(v, 0.004)
    self.assertEqual(_drive(v, ks, {"DogPilotExitGainHold": b"9.0"}), _drive(v, ks, {"DogPilotExitGainHold": b"5.0"}))


if __name__ == "__main__":
  unittest.main()
