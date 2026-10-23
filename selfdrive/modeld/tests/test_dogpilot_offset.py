# DogPilot: the camera offset must shift what the model sees sideways by the same amount at every distance,
# in the documented direction (+ = model centre moves left, so the car moves left), and be exact identity at 0.
import unittest

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.model import get_warp_matrix, medmodel_intrinsics, sbigmodel_intrinsics
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.modeld.dogpilot_offset import dp_shift_model_warp

H = 1.40
RPY = np.radians([0.0, 2.5, -2.9])  # a pitched, yawed mount, like the car this was written for


def world_shift_seen(cam_intrinsics, model_intrinsics, big, offset, x, y):
  """Lateral offset (m) between the ground point a model pixel sees with and without the shift."""
  rot = view_frame_from_device_frame @ rot_from_euler(RPY)
  q = cam_intrinsics @ (rot @ np.array([x, y, H]))
  cam = q / q[2]
  warp = get_warp_matrix(RPY, cam_intrinsics, big)
  m = np.linalg.inv(warp) @ cam
  c2 = dp_shift_model_warp(warp, model_intrinsics, H, offset) @ (m / m[2])
  d = rot.T @ (np.linalg.inv(cam_intrinsics) @ (c2 / c2[2]))
  return d[1] * H / d[2] - y


class TestDogPilotOffset(unittest.TestCase):
  def setUp(self):
    self.dc = next(v for k, v in DEVICE_CAMERAS.items() if k[0] == "mici")

  def test_uniform_shift_both_cameras(self):
    for cam, model_k, big in ((self.dc.fcam.intrinsics, medmodel_intrinsics, False),
                              (self.dc.ecam.intrinsics, sbigmodel_intrinsics, True)):
      for offset in (-0.15, 0.2):
        for x, y in ((8.0, 0.0), (20.0, 1.8), (60.0, -1.8)):
          self.assertAlmostEqual(world_shift_seen(cam, model_k, big, offset, x, y), offset, places=3)

  def test_zero_offset_is_identity(self):
    warp = get_warp_matrix(RPY, self.dc.fcam.intrinsics, False).astype(np.float32)
    self.assertTrue(np.array_equal(dp_shift_model_warp(warp, medmodel_intrinsics, H, 0.0), warp))


if __name__ == "__main__":
  unittest.main()
