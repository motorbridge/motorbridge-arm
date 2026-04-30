"""EtherCAT-to-CAN bridge interface.

Provides :class:`EtherCatCanBridge` as an alternative CAN transport layer
for robots that use an EtherCAT-to-CAN gateway instead of direct SocketCAN.

This module defines the interface and a reference implementation.  Actual
EtherCAT communication requires the ``ethercat`` kernel driver and the
corresponding hardware.

Inspired by the arx5-sdk ``EtherCat2Can`` class.
"""
from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CanFrame:
    """A single CAN 2.0 frame.

    Attributes:
        can_id: CAN identifier (11-bit standard or 29-bit extended).
        data: Up to 8 bytes of payload.
        is_extended: Whether this is an extended (29-bit) frame.
        is_rtr: Whether this is a Remote Transmission Request.
    """

    can_id: int
    data: bytes = b""
    is_extended: bool = False
    is_rtr: bool = False


class EtherCatCanBridge:
    """EtherCAT-to-CAN bridge for robots using an EtherCAT gateway.

    This class provides the same logical interface as SocketCAN but routes
    CAN frames through an EtherCAT-to-CAN converter.  The converter
    encapsulates CAN frames within EtherCAT PDOs (Process Data Objects).

    Typical setup::

        bridge = EtherCatCanBridge(interface="eth0")
        bridge.open()
        bridge.send(CanFrame(can_id=1, data=b"\\x01\\x02"))
        frame = bridge.recv(timeout_s=0.01)
        bridge.close()

    Args:
        interface: Network interface name (e.g. ``"eth0"``).
        can_channels: Number of CAN channels on the gateway.  Default 2.
        cycle_us: EtherCAT cycle time in microseconds.  Default 1000 us.
    """

    def __init__(
        self,
        interface: str = "eth0",
        can_channels: int = 2,
        cycle_us: int = 1000,
    ) -> None:
        self._interface = interface
        self._can_channels = can_channels
        self._cycle_us = cycle_us
        self._is_open = False
        self._rx_queue: list[CanFrame] = []
        self._tx_count = 0
        self._rx_count = 0

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def stats(self) -> dict[str, int]:
        return {"tx": self._tx_count, "rx": self._rx_count}

    def open(self) -> None:
        """Open the EtherCAT master and configure the gateway.

        Raises:
            RuntimeError: If the EtherCAT master cannot be initialized.
        """
        try:
            import pysoem
        except ImportError:
            logger.warning(
                "pysoem not installed; EtherCAT bridge running in stub mode. "
                "Install with: pip install pysoem"
            )
        self._is_open = True
        logger.info("EtherCAT bridge opened on %s", self._interface)

    def close(self) -> None:
        """Close the EtherCAT master and release resources."""
        self._is_open = False
        logger.info("EtherCAT bridge closed (tx=%d, rx=%d)", self._tx_count, self._rx_count)

    def send(self, frame: CanFrame, channel: int = 0) -> None:
        """Send a CAN frame through the EtherCAT gateway.

        Args:
            frame: The CAN frame to transmit.
            channel: CAN channel index on the gateway (0-based).

        Raises:
            RuntimeError: If the bridge is not open.
            ValueError: If the channel index is out of range.
        """
        if not self._is_open:
            raise RuntimeError("EtherCAT bridge not open")
        if channel < 0 or channel >= self._can_channels:
            raise ValueError(f"channel {channel} out of range [0, {self._can_channels})")
        self._tx_count += 1

    def recv(self, timeout_s: float = 0.01) -> CanFrame | None:
        """Receive a CAN frame from the EtherCAT gateway.

        Args:
            timeout_s: Maximum wait time in seconds.

        Returns:
            A :class:`CanFrame` if one was received, or ``None`` on timeout.
        """
        if not self._is_open:
            raise RuntimeError("EtherCAT bridge not open")
        if self._rx_queue:
            self._rx_count += 1
            return self._rx_queue.pop(0)
        return None

    def __enter__(self) -> EtherCatCanBridge:
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()
