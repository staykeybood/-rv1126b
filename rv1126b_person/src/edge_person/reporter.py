"""
报警上报模块。
同一份事件 JSON 可以同时写入本地日志，也可以通过 UDP/TCP/串口发出去。
板端接管理平台时，通常只需要在配置文件里填对应地址或串口。
"""

from __future__ import annotations

import json
import socket
from pathlib import Path


class EventReporter:
    """负责把报警事件发到文件、网络或串口。"""

    def __init__(
        self,
        jsonl_path: Path | None = None,
        udp: str | None = None,
        tcp: str | None = None,
        serial_port: str | None = None,
        serial_baud: int = 115200,
    ):
        self.file = None
        self.udp_addr = parse_addr(udp) if udp else None
        self.tcp_addr = parse_addr(tcp) if tcp else None
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.udp_addr else None
        self.tcp_sock = None
        self.serial = None

        if jsonl_path:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = jsonl_path.open("a", encoding="utf-8")
        if self.tcp_addr:
            self.tcp_sock = socket.create_connection(self.tcp_addr, timeout=3.0)
        if serial_port:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("Install pyserial before using --serial-port") from exc
            self.serial = serial.Serial(serial_port, serial_baud, timeout=0.2)

    def send(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        data = line.encode("utf-8")
        if self.file:
            self.file.write(line)
            self.file.flush()
        if self.udp_sock and self.udp_addr:
            self.udp_sock.sendto(data, self.udp_addr)
        if self.tcp_sock:
            self.tcp_sock.sendall(data)
        if self.serial:
            self.serial.write(data)

    def close(self) -> None:
        if self.file:
            self.file.close()
        if self.tcp_sock:
            self.tcp_sock.close()
        if self.udp_sock:
            self.udp_sock.close()
        if self.serial:
            self.serial.close()


def parse_addr(text: str) -> tuple[str, int]:
    host, port = text.rsplit(":", 1)
    return host, int(port)
