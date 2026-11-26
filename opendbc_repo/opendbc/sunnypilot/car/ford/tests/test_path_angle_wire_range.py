# The path angle the car receives must stay inside the DBC signal range in both directions.
#
# carcontroller.py sends -path_angle (host sign: + = right) as LatCtlPath_An_Actl, whose DBC range is
# [-0.5, 0.5235] rad (11 bits, 0.0005/bit, offset -0.5). A host value above +0.5 goes out below -0.5, which the
# packer can't represent: the raw value underflows and wraps to near +0.5235 (full lock the other way). That
# happened once on a car (a 7.7 mph right turn, Nov 2026). So the host clamp must be the DBC range mirrored.

import unittest

from opendbc.sunnypilot.car.ford.lateral_angle_ext import FORD_DBC_PATH_ANGLE_MAX, FORD_DBC_PATH_ANGLE_MIN
from opendbc.sunnypilot.car.ford.tests.test_lateral_angle_ext import (
  _Actuators, _CC, _CS, _FakeParams, _ForcedDetector, _Harness, _Model, _explorer_cp)

SCALE = 0.0005
RAW_MAX = 2047   # 11-bit field


def _wire_raw(path_angle):
  """Raw LatCtlPath_An_Actl as the packer would compute it from carcontroller's -path_angle (unmasked)."""
  return round((-path_angle - FORD_DBC_PATH_ANGLE_MIN) / SCALE)


def _turn(k, start, v=3.4, steps=40):
  """A very tight turn (curvature k, + = right) at low speed, starting from a path angle `start` just below the
  saturation threshold: one rate-limited step (0.055 rad below 9 m/s) can then jump past +-0.5, as on the car
  (0.475 -> 0.5165). Returns every path angle sent."""
  ext = _Harness(_explorer_cp())
  ext.human_turn_detector = _ForcedDetector(False)
  ext.update_angle_params(_FakeParams({}))
  m = _Model()
  m.orientationRate.z = [v * k] * 33
  ext.model = m
  ext.path_angle_last = start
  out = []
  for _ in range(steps):
    r = ext.update_angle_strategy(_CC(latActive=True), _CS(vEgoRaw=v, vEgo=v, yawRate=-k * v), _Actuators(curvature=k), ext.CP)
    out.append(r.path_angle)
  return out


class TestPathAngleWireRange(unittest.TestCase):
  def _check(self, angles):
    for pa in angles:
      raw = _wire_raw(pa)
      self.assertGreaterEqual(raw, 0, f"path_angle {pa:+.4f} -> wire {-pa:+.4f} underflows the DBC field")
      self.assertLessEqual(raw, RAW_MAX, f"path_angle {pa:+.4f} -> wire {-pa:+.4f} overflows the DBC field")

  def test_tight_right_turn_stays_in_range(self):
    # the car's case: 0.46 then a full rate-limited step; before the fix this sent +0.515 (wire -0.515, wraps)
    angles = _turn(+0.20, start=+0.46)
    self._check(angles)
    self.assertLessEqual(max(angles), -FORD_DBC_PATH_ANGLE_MIN + 1e-9)   # host max = +0.5 (wire -0.5)

  def test_tight_right_turn_still_reaches_the_limit(self):
    angles = _turn(+0.20, start=+0.44)
    self.assertGreater(max(angles), 0.49)
    self._check(angles)

  def test_tight_left_turn_keeps_full_range(self):
    angles = _turn(-0.20, start=-0.46)
    self.assertLess(min(angles), -0.5)             # left may use the full range: host -0.5235 = wire +0.5235
    self._check(angles)
    self.assertGreaterEqual(min(angles), -FORD_DBC_PATH_ANGLE_MAX - 1e-9)

  def test_steady_approach_both_ways(self):
    for k in (+0.20, -0.20):
      self._check(_turn(k, start=0.0, steps=200))


if __name__ == "__main__":
  unittest.main()
