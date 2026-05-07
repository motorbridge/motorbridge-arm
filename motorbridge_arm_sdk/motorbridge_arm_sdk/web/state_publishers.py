from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class StatePublisher(Protocol):
    def publish(self, event: dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...


class NullStatePublisher:
    def publish(self, event: dict[str, Any]) -> None:
        return

    def close(self) -> None:
        return


class MultiStatePublisher:
    def __init__(self, publishers: list[StatePublisher] | None = None) -> None:
        self._publishers = list(publishers or [])

    def add(self, publisher: StatePublisher) -> None:
        self._publishers.append(publisher)

    def publish(self, event: dict[str, Any]) -> None:
        for publisher in list(self._publishers):
            try:
                publisher.publish(event)
            except Exception:
                logger.exception("state publisher %r failed", publisher)

    def close(self) -> None:
        for publisher in list(self._publishers):
            try:
                publisher.close()
            except Exception:
                logger.exception("state publisher %r failed to close", publisher)


class JsonlStatePublisher:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def publish(self, event: dict[str, Any]) -> None:
        self._fh.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class Ros2JsonStatePublisher:
    """Publish gateway state/events as JSON on a ROS2 std_msgs/String topic.

    This intentionally keeps ROS2 optional. If rclpy/std_msgs are not installed
    in the active environment, construction raises RuntimeError with a clear
    message and the rest of the simulation gateway can still run without ROS2.
    """

    def __init__(self, topic: str = "/motorbridge/simu/events", node_name: str = "motorbridge_simu_gateway") -> None:
        try:
            import rclpy
            from std_msgs.msg import String
        except Exception as exc:  # pragma: no cover - depends on ROS2 environment
            raise RuntimeError("ROS2 publishing requires rclpy and std_msgs in the active environment") from exc

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_context = True
        else:
            self._owns_context = False
        self._node = rclpy.create_node(node_name)
        self._msg_type = String
        self._pub = self._node.create_publisher(String, topic, 10)
        self.topic = topic

    def publish(self, event: dict[str, Any]) -> None:
        msg = self._msg_type()
        msg.data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self._pub.publish(msg)

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_context:
            self._rclpy.shutdown()
