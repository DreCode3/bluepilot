# DogPilot: the lateral delay the model and controls use (sunnypilot/livedelay/helpers.get_lat_delay).
#
# Contract: live learning on (LagdToggle) uses the learned delay from liveDelay; off uses the fixed value LagdToggle
# caches (steerActuatorDelay + LagdToggleDelay). Before the fix, off still used the learned value.

import sys
import types

try:
  import openpilot.common.params  # noqa: F401
except ImportError:  # off the device params_pyx isn't built; helpers only needs the name
  sys.modules["openpilot.common.params"] = types.SimpleNamespace(Params=object)

from openpilot.sunnypilot.livedelay.helpers import get_lat_delay


class _Params:
  def __init__(self, toggle, cache):
    self.toggle, self.cache = toggle, cache

  def get_bool(self, key):
    assert key == "LagdToggle"
    return self.toggle

  def get(self, key, return_default=False):
    assert key == "LagdValueCache"
    return self.cache


def test_live_learning_uses_learned_delay():
  assert get_lat_delay(_Params(True, 0.30), 0.38) == 0.38


def test_fixed_uses_cached_fixed_delay():
  assert get_lat_delay(_Params(False, 0.30), 0.38) == 0.30
