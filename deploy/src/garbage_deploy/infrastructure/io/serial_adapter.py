"""Serial adapter implementation."""

from typing import Optional

import serial

from garbage_deploy.application.ports import SerialPort
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class PySerialAdapter(SerialPort):
    def __init__(self, config: dict):
        self.port = config.get("port", "/dev/ttyUSB0")
        self.baud = config.get("baud", 115200)
        self.timeout = config.get("timeout", 0.2)
        self.write_timeout = config.get("write_timeout", 0.2)
        self._ser = None
        self._enabled = config.get("enabled", True)
        self._open_serial()

    def _open_serial(self):
        if not self._enabled:
            log.info("Serial port disabled")
            return

        try:
            self._ser = serial.Serial(
                self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                write_timeout=self.write_timeout,
            )
            log.info("Serial opened", port=self.port, baud=self.baud)
        except serial.SerialException as e:
            raise RuntimeError(f"Cannot open serial port: {e}")

    def open(self, port: str, baud: int) -> None:
        self.port = port
        self.baud = baud
        self._ser.close() if self._ser else None
        self._open_serial()

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None
            log.info("Serial closed")

    def send(self, data: bytes) -> None:
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("Serial port not open")

        try:
            self._ser.write(data)
        except serial.SerialException as e:
            log.error("Serial write failed", error=str(e))
            raise
