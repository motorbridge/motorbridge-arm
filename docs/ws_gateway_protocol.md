# motorbridge-arm Simu WS Gateway Protocol

Gateway entry:
- `scripts/run_simu_ws_gateway.py`
- default endpoint: `ws://127.0.0.1:9011/ws`

Common startup:
```bash
uv run python scripts/run_simu_ws_gateway.py --host 127.0.0.1 --port 9011
```

Publish the same simulation state/events to a local JSONL stream:
```bash
uv run python scripts/run_simu_ws_gateway.py \
  --host 127.0.0.1 --port 9011 \
  --publish-jsonl /tmp/motorbridge_simu_events.jsonl
```

Publish the same state/events to ROS2 as `std_msgs/String` JSON:
```bash
uv run python scripts/run_simu_ws_gateway.py \
  --host 127.0.0.1 --port 9011 \
  --publish-ros2 \
  --ros2-topic /motorbridge/simu/events
```

ROS2 publishing requires `rclpy` and `std_msgs` to be available in the active
ROS2 environment. The JSON schema is intentionally the same as the WebSocket
event schema so simulation consumers and future real-arm consumers can share
one parser.

## Built-in operations
- `ping`
- `state`
- `sim_set_joint_targets`
- `sim_move_l`
- `waypoint_add`
- `waypoint_update`
- `waypoint_remove`
- `waypoint_clear`
- `waypoint_list`
- `waypoint_validate`
- `sim_run_waypoints`
- `sim_run_sequence`
- `sim_stop`
- `bus_config`
- `bus_snapshot`

## Periodic push
- `{"type":"state","data":...}` at 20 Hz
- `{"type":"waypoint","data":...}` on waypoint add/update/remove/clear
- `{"type":"task","data":...}` on task accepted/done/stopped/error
- state includes:
  - `joint_targets`
  - `pose`
  - `waypoints` (`id -> {label, x, y, z, roll, pitch, yaw}`)
  - `motion` (`running/name/from_id/to_id`)

## External publish schema
When `--publish-jsonl` or `--publish-ros2` is enabled, every state/event is
published as:

```json
{
  "schema": "motorbridge.simu.v1",
  "type": "state",
  "source": "simu_gateway",
  "ts": 0.0,
  "data": {}
}
```

Event types currently include:
- `state`
- `waypoint`
- `task`

This gives downstream systems one synchronized stream that can be consumed by
local replay tools, ROS2 nodes, or a future real-arm bridge.

## Dual-buffer bus design
The gateway includes `ProtocolBus` with two ring buffers:
- TX buffer: outbound messages/events
- RX buffer: inbound messages/events

And configurable channels:
- `websocket`
- `sim`
- `motorbridge_py`
- `ros`

Example:
```json
{
  "op": "bus_config",
  "req_id": 10,
  "tx_buffer_size": 2048,
  "rx_buffer_size": 2048,
  "channels": {
    "websocket": true,
    "sim": true,
    "motorbridge_py": true,
    "ros": false
  }
}
```

This allows future protocol expansion without rewriting the /simu page contract.

## CLI testing
You can test waypoint workflow from terminal:

```bash
python scripts/simu_ws_cli.py state
python scripts/simu_ws_cli.py waypoint-add --id P1 --label "Pick point" --x 0.30 --y 0.00 --z 0.22
python scripts/simu_ws_cli.py waypoint-add --id P2 --label "Place point" --x 0.28 --y 0.12 --z 0.24
python scripts/simu_ws_cli.py waypoint-list
python scripts/simu_ws_cli.py run --from-id P1 --to-id P2 --duration-s 2.0 --profile geodesic
python scripts/simu_ws_cli.py run-seq --ids P1,P3,P2 --duration-s 1.8 --profile min_jerk
python scripts/simu_ws_cli.py stop
```
