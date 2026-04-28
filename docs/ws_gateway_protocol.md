# motorbridge-arm Simu WS Gateway Protocol

Gateway entry:
- `scripts/run_simu_ws_gateway.py`
- default endpoint: `ws://127.0.0.1:9011/ws`

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
- `sim_run_waypoints`
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
  - `waypoints`
  - `motion` (`running/name/from_id/to_id`)

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
python scripts/simu_ws_cli.py waypoint-add --id P1 --x 0.30 --y 0.00 --z 0.22
python scripts/simu_ws_cli.py waypoint-add --id P2 --x 0.28 --y 0.12 --z 0.24
python scripts/simu_ws_cli.py waypoint-list
python scripts/simu_ws_cli.py run --from-id P1 --to-id P2 --duration-s 2.0
python scripts/simu_ws_cli.py stop
```
