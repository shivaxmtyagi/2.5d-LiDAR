"""
PLACEHOLDER / TODO — ROS / ROS2 live sensor bridge.

Not implemented in this build. This environment has no ROS installation and
no live sensor to validate against, so shipping a bridge here would be
untested code presented as working — which the project spec explicitly
forbids (see README > Limitations).

To implement for real:
  1. Subscribe to a `sensor_msgs/PointCloud2` topic (ROS1: rospy, ROS2: rclpy).
  2. Use `sensor_msgs_py.point_cloud2.read_points` (ROS2) or
     `ros_numpy.point_cloud2.pointcloud2_to_array` (ROS1) to get a structured
     numpy array.
  3. Map its fields (x, y, z, intensity, ring, t) onto
     `src.common.pointcloud.PointCloud` — the same target every other loader
     in `src/input/` produces, so nothing downstream changes.
  4. Respect `frame_id` from the message header for
     `src.preprocessing.transforms`.
"""

from src.common.pointcloud import PointCloud


def load_from_ros_topic(topic: str, timeout_s: float = 5.0) -> PointCloud:
    raise NotImplementedError(
        "ROS/ROS2 bridge is a documented placeholder in this build — "
        "see module docstring for the implementation plan."
    )
