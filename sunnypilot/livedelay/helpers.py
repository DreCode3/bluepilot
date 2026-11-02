"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.params import Params


def get_lat_delay(params: Params, stock_lat_delay: float) -> float:
  # DogPilot: sunnypilot's fix (sunnypilot/sunnypilot#1906). Live learning on: use what lagd publishes. Off: use the
  # fixed steerActuatorDelay + LagdToggleDelay that LagdToggle caches. The branches were inverted, so turning live
  # learning off still used the learned value.
  if params.get_bool("LagdToggle"):
    return stock_lat_delay

  return float(params.get("LagdValueCache", return_default=True))
