#!/usr/bin/env python3
"""
Extract the current live keymap from a ZMK Studio-enabled board over the Studio
serial RPC transport and render it back into source-friendly bindings.

The script can:
1. Save a raw JSON dump of the keymap and behavior metadata.
2. Print a DTS-style binding snippet for each layer.
3. Replace existing `bindings = < ... >;` blocks in a source keymap file when
   the layer count and key counts match and every runtime behavior can be mapped
   back to a source token.

This is intended for user-config repos where Studio edits are persisted on the
device but not automatically written back into the checked-in `.keymap` file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import sys
import termios
import tty
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


SOF = 0xAB
ESC = 0xAC
EOF = 0xAD

WIRE_VARINT = 0
WIRE_LEN = 2


class StudioRpcError(RuntimeError):
    pass


class UnknownBehaviorError(StudioRpcError):
    pass


class PatchError(RuntimeError):
    pass


@dataclass
class BehaviorDef:
    label: str
    node_name: str
    binding_cells: int
    compatible: str | None = None
    display_name: str | None = None
    source: str | None = None


@dataclass
class BehaviorInfo:
    local_id: int
    display_name: str
    token: str | None
    binding_cells: int | None


@dataclass
class RuntimeBinding:
    behavior_id: int
    param1: int
    param2: int


@dataclass
class RuntimeLayer:
    id: int
    name: str
    bindings: list[RuntimeBinding]


@dataclass
class BindingBlock:
    start: int
    end: int
    line_counts: list[int]
    indent: str
    bindings: list[str]


@dataclass
class SymbolMaps:
    key_symbols: dict[int, str] = field(default_factory=dict)
    mouse_button_symbols: dict[int, str] = field(default_factory=dict)
    output_symbols: dict[int, str] = field(default_factory=dict)
    bt_commands: dict[tuple[int, int | None], str] = field(default_factory=dict)
    rgb_commands: dict[tuple[int, int | None], str] = field(default_factory=dict)
    move_symbols: dict[int, str] = field(default_factory=dict)
    scroll_symbols: dict[int, str] = field(default_factory=dict)


def encode_varint(value: int) -> bytes:
    if value < 0:
        value &= (1 << 64) - 1
    out = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            return bytes(out)


def decode_varint(data: bytes, pos: int = 0) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise StudioRpcError("Unexpected end of protobuf varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7
        if shift > 64:
            raise StudioRpcError("Varint is too large")


def zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def encode_field_key(field_number: int, wire_type: int) -> bytes:
    return encode_varint((field_number << 3) | wire_type)


def encode_bool_field(field_number: int, value: bool) -> bytes:
    return encode_field_key(field_number, WIRE_VARINT) + encode_varint(1 if value else 0)


def encode_varint_field(field_number: int, value: int) -> bytes:
    return encode_field_key(field_number, WIRE_VARINT) + encode_varint(value)


def encode_message_field(field_number: int, payload: bytes) -> bytes:
    return encode_field_key(field_number, WIRE_LEN) + encode_varint(len(payload)) + payload


def parse_message(data: bytes) -> dict[int, list[tuple[int, int | bytes]]]:
    fields: dict[int, list[tuple[int, int | bytes]]] = defaultdict(list)
    pos = 0
    while pos < len(data):
        key, pos = decode_varint(data, pos)
        field_number = key >> 3
        wire_type = key & 0x07
        if wire_type == WIRE_VARINT:
            value, pos = decode_varint(data, pos)
        elif wire_type == WIRE_LEN:
            length, pos = decode_varint(data, pos)
            value = data[pos : pos + length]
            pos += length
        else:
            raise StudioRpcError(f"Unsupported protobuf wire type: {wire_type}")
        fields[field_number].append((wire_type, value))
    return fields


def get_varint(fields: dict[int, list[tuple[int, int | bytes]]], field_number: int, default: int = 0) -> int:
    values = fields.get(field_number)
    if not values:
        return default
    wire_type, value = values[0]
    if wire_type != WIRE_VARINT:
        raise StudioRpcError(f"Field {field_number} is not a varint")
    return int(value)


def get_bytes(
    fields: dict[int, list[tuple[int, int | bytes]]], field_number: int, default: bytes | None = None
) -> bytes | None:
    values = fields.get(field_number)
    if not values:
        return default
    wire_type, value = values[0]
    if wire_type != WIRE_LEN:
        raise StudioRpcError(f"Field {field_number} is not length-delimited")
    return bytes(value)


def parse_packed_varints(payload: bytes) -> list[int]:
    values = []
    pos = 0
    while pos < len(payload):
        value, pos = decode_varint(payload, pos)
        values.append(value)
    return values


def frame_payload(payload: bytes) -> bytes:
    framed = bytearray([SOF])
    for byte in payload:
        if byte in (SOF, ESC, EOF):
            framed.append(ESC)
        framed.append(byte)
    framed.append(EOF)
    return bytes(framed)


class RawSerial:
    def __init__(self, path: str, baud: int = 115200):
        self.path = path
        self.baud = baud
        self.fd: int | None = None
        self._old_attrs = None

    def __enter__(self) -> "RawSerial":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self.fd is not None:
            return
        self.fd = os.open(self.path, os.O_RDWR | os.O_NOCTTY)
        self._old_attrs = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        attrs = termios.tcgetattr(self.fd)
        attrs[2] |= termios.CLOCAL | termios.CREAD
        baud_const = getattr(termios, f"B{self.baud}", termios.B115200)
        attrs[4] = baud_const
        attrs[5] = baud_const
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)

    def close(self) -> None:
        if self.fd is None:
            return
        if self._old_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self._old_attrs)
        os.close(self.fd)
        self.fd = None
        self._old_attrs = None

    def write_all(self, data: bytes) -> None:
        if self.fd is None:
            raise StudioRpcError("Serial port is not open")
        remaining = memoryview(data)
        while remaining:
            written = os.write(self.fd, remaining)
            remaining = remaining[written:]

    def read_frame(self, timeout: float) -> bytes:
        if self.fd is None:
            raise StudioRpcError("Serial port is not open")
        payload = bytearray()
        in_frame = False
        escaped = False
        deadline = select_time() + timeout
        while True:
            remaining = deadline - select_time()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for Studio RPC frame on {self.path}")
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(self.fd, 4096)
            if not chunk:
                continue
            for byte in chunk:
                if not in_frame:
                    if byte == SOF:
                        in_frame = True
                        escaped = False
                        payload.clear()
                    continue
                if escaped:
                    payload.append(byte)
                    escaped = False
                    continue
                if byte == ESC:
                    escaped = True
                    continue
                if byte == EOF:
                    return bytes(payload)
                if byte == SOF:
                    payload.clear()
                    escaped = False
                    continue
                payload.append(byte)


def select_time() -> float:
    # Isolated for testing and to keep the read loop readable.
    import time

    return time.monotonic()


class StudioRpcClient:
    def __init__(self, port: RawSerial, timeout: float = 5.0):
        self.port = port
        self.timeout = timeout
        self.request_id = 0

    def send_request(self, subsystem_field: int, subsystem_payload: bytes) -> dict[str, object]:
        self.request_id += 1
        request_payload = (
            encode_varint_field(1, self.request_id)
            + encode_message_field(subsystem_field, subsystem_payload)
        )
        self.port.write_all(frame_payload(request_payload))
        return self._wait_for_response(self.request_id)

    def _wait_for_response(self, request_id: int) -> dict[str, object]:
        while True:
            payload = self.port.read_frame(self.timeout)
            top = parse_message(payload)
            request_response = get_bytes(top, 1)
            notification = get_bytes(top, 2)
            if notification is not None:
                continue
            if request_response is None:
                continue
            rr = parse_message(request_response)
            if get_varint(rr, 1) != request_id:
                continue
            if 2 in rr:
                meta = parse_message(get_bytes(rr, 2) or b"")
                simple_error = get_varint(meta, 2, default=-1)
                raise StudioRpcError(f"Studio RPC meta error: {simple_error}")
            if 3 in rr:
                return {"subsystem": "core", "payload": get_bytes(rr, 3) or b""}
            if 4 in rr:
                return {"subsystem": "behaviors", "payload": get_bytes(rr, 4) or b""}
            if 5 in rr:
                return {"subsystem": "keymap", "payload": get_bytes(rr, 5) or b""}
            raise StudioRpcError("Unknown Studio RPC subsystem in response")

    def get_device_info(self) -> dict[str, object]:
        response = self.send_request(3, encode_bool_field(1, True))
        if response["subsystem"] != "core":
            raise StudioRpcError("Unexpected subsystem for get_device_info")
        fields = parse_message(response["payload"])
        dev_info = parse_message(get_bytes(fields, 1) or b"")
        name = (get_bytes(dev_info, 1) or b"").decode("utf-8", "replace")
        serial_number = get_bytes(dev_info, 2) or b""
        return {"name": name, "serial_number_hex": serial_number.hex()}

    def get_keymap(self) -> list[RuntimeLayer]:
        response = self.send_request(5, encode_bool_field(1, True))
        if response["subsystem"] != "keymap":
            raise StudioRpcError("Unexpected subsystem for get_keymap")
        keymap_response = parse_message(response["payload"])
        keymap = parse_message(get_bytes(keymap_response, 1) or b"")
        layers: list[RuntimeLayer] = []
        for _, layer_bytes in keymap.get(1, []):
            layer_msg = parse_message(bytes(layer_bytes))
            layer_id = get_varint(layer_msg, 1)
            name_bytes = get_bytes(layer_msg, 2)
            name = name_bytes.decode("utf-8", "replace") if name_bytes is not None else ""
            bindings: list[RuntimeBinding] = []
            for _, binding_bytes in layer_msg.get(3, []):
                binding_msg = parse_message(bytes(binding_bytes))
                raw_behavior_id = get_varint(binding_msg, 1)
                bindings.append(
                    RuntimeBinding(
                        behavior_id=zigzag_decode(raw_behavior_id),
                        param1=get_varint(binding_msg, 2),
                        param2=get_varint(binding_msg, 3),
                    )
                )
            layers.append(RuntimeLayer(id=layer_id, name=name, bindings=bindings))
        return layers

    def get_behavior_details(self, behavior_id: int) -> dict[str, object]:
        behaviors_req = encode_message_field(2, encode_varint_field(1, behavior_id))
        response = self.send_request(4, behaviors_req)
        if response["subsystem"] != "behaviors":
            raise StudioRpcError("Unexpected subsystem for get_behavior_details")
        behaviors_response = parse_message(response["payload"])
        details = parse_message(get_bytes(behaviors_response, 2) or b"")
        display_name_bytes = get_bytes(details, 2) or b""
        return {
            "id": get_varint(details, 1),
            "display_name": display_name_bytes.decode("utf-8", "replace"),
        }


def normalize_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def unfold_c_preprocessor_lines(text: str) -> list[str]:
    lines: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if pending:
            line = pending + line.lstrip()
        if line.endswith("\\"):
            pending = line[:-1] + " "
            continue
        pending = ""
        lines.append(line)
    if pending:
        lines.append(pending)
    return lines


def discover_workspace_root(repo_root: Path) -> Path | None:
    env_root = os.environ.get("ZMK_WORKSPACE")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([Path.cwd(), repo_root, repo_root.parent])
    for base in [repo_root.parent, Path.cwd()]:
        if base.exists():
            candidates.extend(sorted(base.glob(".zmk-*")))
            candidates.extend(sorted(base.glob("*zmk*build*")))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "zmk/app/dts/behaviors").is_dir() and (
            candidate / "zmk/app/include/dt-bindings/zmk"
        ).is_dir():
            return candidate
    return None


def parse_behavior_definitions(paths: list[Path]) -> list[BehaviorDef]:
    definitions: list[BehaviorDef] = []
    start_re = re.compile(r"^\s*(?:/omit-if-no-ref/\s+)?([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)\s*\{")
    compatible_re = re.compile(r'compatible\s*=\s*"([^"]+)"')
    display_re = re.compile(r'display-name\s*=\s*"([^"]+)"')
    cells_re = re.compile(r"#binding-cells\s*=\s*<(\d+)>")
    preprocessor_re = re.compile(r"^\s*#(?:if|ifdef|ifndef|endif|else|elif|include|define|undef|error|warning|pragma)\b")
    for path in paths:
        if not path.exists():
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        current: BehaviorDef | None = None
        depth = 0
        for line in unfold_c_preprocessor_lines(text):
            if preprocessor_re.match(line):
                continue
            if current is None:
                start_match = start_re.match(line)
                if not start_match or "\\" in line:
                    continue
                current = BehaviorDef(
                    label=start_match.group(1),
                    node_name=start_match.group(2),
                    binding_cells=0,
                    source=str(path),
                )
                depth = line.count("{") - line.count("}")
                comp_match = compatible_re.search(line)
                if comp_match:
                    current.compatible = comp_match.group(1)
                display_match = display_re.search(line)
                if display_match:
                    current.display_name = display_match.group(1)
                cells_match = cells_re.search(line)
                if cells_match:
                    current.binding_cells = int(cells_match.group(1))
            else:
                comp_match = compatible_re.search(line)
                if comp_match:
                    current.compatible = comp_match.group(1)
                display_match = display_re.search(line)
                if display_match:
                    current.display_name = display_match.group(1)
                cells_match = cells_re.search(line)
                if cells_match:
                    current.binding_cells = int(cells_match.group(1))
                depth += line.count("{") - line.count("}")
                if depth <= 0:
                    if current.compatible and current.compatible.startswith("zmk,behavior"):
                        definitions.append(current)
                    current = None
    return definitions


def build_behavior_lookup(definitions: list[BehaviorDef]) -> tuple[dict[str, BehaviorDef], dict[str, int]]:
    candidate_map: dict[str, list[tuple[int, BehaviorDef]]] = defaultdict(list)
    binding_cells_by_label: dict[str, int] = {}
    for definition in definitions:
        binding_cells_by_label[definition.label] = definition.binding_cells
        candidates = [
            (0, definition.display_name),
            (1, definition.node_name),
            (2, definition.label),
            (3, definition.node_name.upper()),
            (4, definition.label.upper()),
        ]
        for priority, candidate in candidates:
            normalized = normalize_name(candidate)
            if normalized:
                candidate_map[normalized].append((priority, definition))
    resolved: dict[str, BehaviorDef] = {}
    for normalized, entries in candidate_map.items():
        entries.sort(key=lambda item: item[0])
        best_priority = entries[0][0]
        best = {entry.label: entry for prio, entry in entries if prio == best_priority}
        if len(best) == 1:
            resolved[normalized] = next(iter(best.values()))
    return resolved, binding_cells_by_label


def read_object_macros(paths: list[Path], extra_macros: dict[str, str] | None = None) -> dict[str, str]:
    macros: dict[str, str] = {}
    if extra_macros:
        macros.update(extra_macros)
    define_re = re.compile(r"^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.*)$")
    func_define_re = re.compile(r"^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\((.*?)\)\s+(.*)$")
    for path in paths:
        if not path.exists():
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        for line in unfold_c_preprocessor_lines(text):
            if func_define_re.match(line):
                continue
            match = define_re.match(line)
            if not match:
                continue
            name, expr = match.groups()
            macros[name] = expr.strip()
    return macros


def evaluate_macros(macros: dict[str, str]) -> dict[str, int]:
    resolved: dict[str, int] = {}
    resolving: set[str] = set()
    python_funcs: dict[str, Callable[..., int]] = {
        "BIT": lambda n: 1 << n,
        "ZMK_HID_USAGE": lambda page, usage_id: ((page << 16) | usage_id),
        "APPLY_MODS": lambda mods, keycode: ((mods << 24) | keycode),
        "LC": lambda keycode: ((0x01 << 24) | keycode),
        "LS": lambda keycode: ((0x02 << 24) | keycode),
        "LA": lambda keycode: ((0x04 << 24) | keycode),
        "LG": lambda keycode: ((0x08 << 24) | keycode),
        "RC": lambda keycode: ((0x10 << 24) | keycode),
        "RS": lambda keycode: ((0x20 << 24) | keycode),
        "RA": lambda keycode: ((0x40 << 24) | keycode),
        "RG": lambda keycode: ((0x80 << 24) | keycode),
        "MOVE_Y": lambda vert: (vert & 0xFFFF),
        "MOVE_X": lambda hor: ((hor & 0xFFFF) << 16),
        "MOVE": lambda hor, vert: (((hor & 0xFFFF) << 16) + (vert & 0xFFFF)),
        "UINT32_C": lambda value: value,
    }
    token_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")

    def resolve(name: str) -> int:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise ValueError(f"Recursive macro reference while resolving {name}")
        if name not in macros:
            raise KeyError(name)
        resolving.add(name)
        try:
            expr = macros[name]
            namespace: dict[str, object] = dict(python_funcs)
            for token in token_re.findall(expr):
                if token in namespace or token == name:
                    continue
                if token in macros:
                    try:
                        namespace[token] = resolve(token)
                    except Exception:
                        pass
            value = eval(expr, {"__builtins__": {}}, namespace)
        finally:
            resolving.remove(name)
        if not isinstance(value, int):
            raise ValueError(f"Macro {name} did not evaluate to an integer")
        resolved[name] = value
        return value

    for name in list(macros):
        try:
            resolve(name)
        except Exception:
            continue
    return resolved


def choose_best_symbol(names: list[str]) -> str:
    def score(name: str) -> tuple[int, int, str]:
        penalty = 0
        if name.startswith("NUMBER_") or name.startswith("NUM_"):
            penalty += 3
        if name.startswith("LEFT_") and "BRACKET" in name:
            penalty += 1
        if name.startswith("RIGHT_") and "BRACKET" in name:
            penalty += 1
        if name.endswith("_CMD"):
            penalty += 10
        return (penalty, len(name), name)

    return sorted(names, key=score)[0]


def extract_pointing_defaults(keymap_path: Path) -> tuple[int, int]:
    move_default = 600
    scroll_default = 10
    if not keymap_path.exists():
        return move_default, scroll_default
    text = keymap_path.read_text(encoding="utf-8")
    move_match = re.search(r"#define\s+ZMK_POINTING_DEFAULT_MOVE_VAL\s+(-?\d+)", text)
    scroll_match = re.search(r"#define\s+ZMK_POINTING_DEFAULT_SCRL_VAL\s+(-?\d+)", text)
    if move_match:
        move_default = int(move_match.group(1))
    if scroll_match:
        scroll_default = int(scroll_match.group(1))
    return move_default, scroll_default


def build_symbol_maps(workspace_root: Path | None, keymap_path: Path) -> SymbolMaps:
    maps = SymbolMaps(
        output_symbols={0: "OUT_TOG", 1: "OUT_USB", 2: "OUT_BLE"},
        bt_commands={
            (0, 0): "BT_CLR",
            (1, 0): "BT_NXT",
            (2, 0): "BT_PRV",
            (3, None): "BT_SEL",
            (4, 0): "BT_CLR_ALL",
            (5, None): "BT_DISC",
        },
        rgb_commands={
            (0, 0): "RGB_TOG",
            (1, 0): "RGB_ON",
            (2, 0): "RGB_OFF",
            (3, 0): "RGB_HUI",
            (4, 0): "RGB_HUD",
            (5, 0): "RGB_SAI",
            (6, 0): "RGB_SAD",
            (7, 0): "RGB_BRI",
            (8, 0): "RGB_BRD",
            (9, 0): "RGB_SPI",
            (10, 0): "RGB_SPD",
            (11, 0): "RGB_EFF",
            (12, 0): "RGB_EFR",
        },
    )

    move_default, scroll_default = extract_pointing_defaults(keymap_path)
    maps.mouse_button_symbols = {
        1 << 0: "LCLK",
        1 << 1: "RCLK",
        1 << 2: "MCLK",
        1 << 3: "MB4",
        1 << 4: "MB5",
    }
    maps.move_symbols = {
        move_y(-move_default): "MOVE_UP",
        move_y(move_default): "MOVE_DOWN",
        move_x(-move_default): "MOVE_LEFT",
        move_x(move_default): "MOVE_RIGHT",
    }
    maps.scroll_symbols = {
        move_y(scroll_default): "SCRL_UP",
        move_y(-scroll_default): "SCRL_DOWN",
        move_x(-scroll_default): "SCRL_LEFT",
        move_x(scroll_default): "SCRL_RIGHT",
    }

    if workspace_root is None:
        return maps

    extra_macros = {
        "ZMK_POINTING_DEFAULT_MOVE_VAL": str(move_default),
        "ZMK_POINTING_DEFAULT_SCRL_VAL": str(scroll_default),
    }
    headers = [
        workspace_root / "zmk/app/include/dt-bindings/zmk/hid_usage_pages.h",
        workspace_root / "zmk/app/include/dt-bindings/zmk/hid_usage.h",
        workspace_root / "zmk/app/include/dt-bindings/zmk/modifiers.h",
        workspace_root / "zmk/app/include/dt-bindings/zmk/keys.h",
        workspace_root / "zmk/app/include/dt-bindings/zmk/pointing.h",
    ]
    macros = read_object_macros(headers, extra_macros=extra_macros)
    values = evaluate_macros(macros)
    by_value: dict[int, list[str]] = defaultdict(list)
    for name, value in values.items():
        by_value[value].append(name)
    maps.key_symbols = {value: choose_best_symbol(names) for value, names in by_value.items()}
    return maps


def move_x(hor: int) -> int:
    return ((hor & 0xFFFF) << 16)


def move_y(vert: int) -> int:
    return vert & 0xFFFF


def parse_binding_sequence(tokens: list[str], binding_cells_by_label: dict[str, int]) -> list[str]:
    bindings: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("&"):
            raise PatchError(f"Unexpected token in bindings block: {token}")
        label = token[1:]
        if label not in binding_cells_by_label:
            raise PatchError(f"Unknown behavior label in keymap template: {label}")
        end = i + 1
        while end < len(tokens) and not tokens[end].startswith("&"):
            end += 1
        bindings.append(" ".join(tokens[i:end]))
        i = end
    return bindings


def parse_binding_blocks(keymap_text: str, binding_cells_by_label: dict[str, int]) -> list[BindingBlock]:
    blocks: list[BindingBlock] = []
    search_pos = 0
    start_re = re.compile(r"^[ \t]*bindings\s*=\s*<\s*$", re.M)
    end_re = re.compile(r"^[ \t]*>\s*;\s*$", re.M)
    while True:
        start_match = start_re.search(keymap_text, search_pos)
        if not start_match:
            break
        content_start = start_match.end()
        end_match = end_re.search(keymap_text, content_start)
        if not end_match:
            raise PatchError("Unterminated bindings block in keymap file")
        block_text = keymap_text[content_start:end_match.start()]
        line_counts: list[int] = []
        bindings: list[str] = []
        indent = "            "
        for line in block_text.splitlines():
            stripped = strip_comments(line).strip()
            if not stripped:
                continue
            indent_match = re.match(r"^(\s*)", line)
            if indent_match:
                indent = indent_match.group(1) or indent
            line_bindings = parse_binding_sequence(stripped.split(), binding_cells_by_label)
            line_counts.append(len(line_bindings))
            bindings.extend(line_bindings)
        blocks.append(
            BindingBlock(
                start=content_start,
                end=end_match.start(),
                line_counts=line_counts,
                indent=indent,
                bindings=bindings,
            )
        )
        search_pos = end_match.end()
    return blocks


def choose_symbol(preferred: dict[int, str], fallback: dict[int, str], value: int) -> str:
    return preferred.get(value) or fallback.get(value) or str(value)


def reverse_style_overrides(keymap_path: Path, blocks: list[BindingBlock], maps: SymbolMaps) -> None:
    if not keymap_path.exists():
        return
    for block in blocks:
        for binding in block.bindings:
            parts = binding.split()
            label = parts[0][1:]
            args = parts[1:]
            try:
                if label == "kp" and args:
                    value = maps.key_symbols_lookup_expr(args[0])  # type: ignore[attr-defined]
                elif label == "mkp" and args:
                    value = maps.pointing_lookup_expr(args[0])  # type: ignore[attr-defined]
                elif label in {"mmv", "msc"} and args:
                    value = maps.pointing_lookup_expr(args[0])  # type: ignore[attr-defined]
                else:
                    continue
            except Exception:
                continue
            if label == "kp":
                maps.key_symbols[value] = args[0]
            elif label == "mkp":
                maps.mouse_button_symbols[value] = args[0]
            elif label == "mmv":
                maps.move_symbols[value] = args[0]
            elif label == "msc":
                maps.scroll_symbols[value] = args[0]


def attach_style_lookup_helpers(maps: SymbolMaps, workspace_root: Path | None, keymap_path: Path) -> None:
    if workspace_root is None:
        maps.key_symbols_lookup_expr = lambda expr: int(expr, 0)  # type: ignore[attr-defined]
        maps.pointing_lookup_expr = lambda expr: int(expr, 0)  # type: ignore[attr-defined]
        return
    extra_macros = {
        "ZMK_POINTING_DEFAULT_MOVE_VAL": str(extract_pointing_defaults(keymap_path)[0]),
        "ZMK_POINTING_DEFAULT_SCRL_VAL": str(extract_pointing_defaults(keymap_path)[1]),
    }
    headers = [
        workspace_root / "zmk/app/include/dt-bindings/zmk/hid_usage_pages.h",
        workspace_root / "zmk/app/include/dt-bindings/zmk/hid_usage.h",
        workspace_root / "zmk/app/include/dt-bindings/zmk/modifiers.h",
        workspace_root / "zmk/app/include/dt-bindings/zmk/keys.h",
        workspace_root / "zmk/app/include/dt-bindings/zmk/pointing.h",
    ]
    macros = read_object_macros(headers, extra_macros=extra_macros)
    values = evaluate_macros(macros)
    maps.key_symbols_lookup_expr = lambda expr: values[expr]  # type: ignore[attr-defined]
    maps.pointing_lookup_expr = lambda expr: values[expr]  # type: ignore[attr-defined]


def decode_axis_value(value: int) -> tuple[int, int]:
    x = (value >> 16) & 0xFFFF
    y = value & 0xFFFF
    if x & 0x8000:
        x -= 0x10000
    if y & 0x8000:
        y -= 0x10000
    return x, y


def format_bt_args(param1: int, param2: int, maps: SymbolMaps) -> list[str]:
    if (param1, 0) in maps.bt_commands and param2 == 0:
        return [maps.bt_commands[(param1, 0)]]
    if (param1, None) in maps.bt_commands:
        return [maps.bt_commands[(param1, None)], str(param2)]
    return [str(param1), str(param2)]


def format_rgb_args(param1: int, param2: int, maps: SymbolMaps) -> list[str]:
    if (param1, 0) in maps.rgb_commands and param2 == 0:
        return [maps.rgb_commands[(param1, 0)]]
    return [str(param1), str(param2)]


def format_axis_arg(value: int, predefined: dict[int, str]) -> str:
    if value in predefined:
        return predefined[value]
    x, y = decode_axis_value(value)
    return f"MOVE({x}, {y})"


def format_binding(binding: RuntimeBinding, behavior: BehaviorInfo, maps: SymbolMaps) -> str:
    if behavior.token is None:
        raise UnknownBehaviorError(
            f"Cannot map behavior {behavior.local_id} ({behavior.display_name}) back to a source token"
        )
    token = behavior.token
    args: list[str]
    if token == "kp":
        args = [choose_symbol({}, maps.key_symbols, binding.param1)]
    elif token == "mkp":
        args = [choose_symbol({}, maps.mouse_button_symbols, binding.param1)]
    elif token in {"mo", "to", "tog", "sl"}:
        args = [str(binding.param1)]
    elif token == "bt":
        args = format_bt_args(binding.param1, binding.param2, maps)
    elif token == "out":
        args = [maps.output_symbols.get(binding.param1, str(binding.param1))]
    elif token == "rgb_ug":
        args = format_rgb_args(binding.param1, binding.param2, maps)
    elif token == "mmv":
        args = [format_axis_arg(binding.param1, maps.move_symbols)]
    elif token == "msc":
        args = [format_axis_arg(binding.param1, maps.scroll_symbols)]
    elif token in {"trans", "none", "caps_word", "key_repeat", "soft_off", "studio_unlock", "sys_reset", "bootloader"}:
        args = []
    else:
        args = []
        arg_cells = behavior.binding_cells or 0
        if arg_cells >= 1:
            args.append(str(binding.param1))
        if arg_cells >= 2:
            args.append(str(binding.param2))
    return " ".join([f"&{token}", *args])


def format_layer_bindings(bindings: list[str], template: BindingBlock | None = None) -> str:
    if template and sum(template.line_counts) == len(bindings):
        lines = []
        offset = 0
        for count in template.line_counts:
            chunk = bindings[offset : offset + count]
            lines.append(f"{template.indent}{'  '.join(chunk)}")
            offset += count
        return "\n".join(lines)
    return "\n".join(f"            {binding}" for binding in bindings)


def update_keymap_file(
    keymap_path: Path,
    template_blocks: list[BindingBlock],
    layers: list[RuntimeLayer],
    behaviors: dict[int, BehaviorInfo],
    maps: SymbolMaps,
) -> str:
    text = keymap_path.read_text(encoding="utf-8")
    if len(template_blocks) != len(layers):
        raise PatchError(
            f"Template layer count ({len(template_blocks)}) does not match runtime layer count ({len(layers)})"
        )
    replacements: list[tuple[int, int, str]] = []
    for block, layer in zip(template_blocks, layers):
        rendered = [format_binding(binding, behaviors[binding.behavior_id], maps) for binding in layer.bindings]
        if block.bindings and len(block.bindings) != len(rendered):
            raise PatchError(
                f"Layer binding count mismatch: template has {len(block.bindings)}, runtime has {len(rendered)}"
            )
        replacements.append((block.start, block.end, "\n" + format_layer_bindings(rendered, block) + "\n"))
    updated = text
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    return updated


def render_layers(layers: list[RuntimeLayer], behaviors: dict[int, BehaviorInfo], maps: SymbolMaps) -> str:
    rendered_layers: list[str] = []
    for layer in layers:
        bindings = [format_binding(binding, behaviors[binding.behavior_id], maps) for binding in layer.bindings]
        header = f"// layer_id={layer.id} name={layer.name or '(unnamed)'}"
        rendered_layers.append(header)
        rendered_layers.append("bindings = <")
        rendered_layers.append(format_layer_bindings(bindings))
        rendered_layers.append(">;")
        rendered_layers.append("")
    return "\n".join(rendered_layers).rstrip() + "\n"


def auto_keymap_path(repo_root: Path) -> Path:
    return repo_root / "config/eyelash_sofle.keymap"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extract the current live keymap from a ZMK Studio-enabled board"
    )
    parser.add_argument("--port", required=True, help="Serial device for the Studio-enabled left half, e.g. /dev/ttyACM0")
    parser.add_argument(
        "--keymap-file",
        default=str(auto_keymap_path(repo_root)),
        help="Source keymap file to use as the formatting template",
    )
    parser.add_argument(
        "--workspace-root",
        help="Optional full ZMK workspace root used for richer behavior and symbol mapping",
    )
    parser.add_argument(
        "--raw-json-out",
        default="studio-keymap-extract.json",
        help="Where to write the raw extracted keymap metadata JSON",
    )
    parser.add_argument(
        "--snippet-out",
        help="Optional file to write the rendered layer binding snippet to",
    )
    parser.add_argument(
        "--update-keymap",
        action="store_true",
        help="Replace the bindings blocks in --keymap-file in place",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate to configure on the Studio CDC ACM device",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Read timeout for Studio RPC responses in seconds",
    )
    return parser.parse_args()


def build_behavior_catalog(workspace_root: Path | None, keymap_path: Path) -> tuple[dict[str, BehaviorDef], dict[str, int]]:
    definition_paths: list[Path] = []
    if workspace_root is not None:
        definition_paths.extend(sorted((workspace_root / "zmk/app/dts/behaviors").glob("*.dtsi")))
    definition_paths.append(keymap_path)
    definitions = parse_behavior_definitions(definition_paths)
    return build_behavior_lookup(definitions)


def map_behavior_to_token(
    display_name: str, behavior_lookup: dict[str, BehaviorDef]
) -> tuple[str | None, int | None]:
    normalized = normalize_name(display_name)
    definition = behavior_lookup.get(normalized)
    if definition is None:
        return None, None
    return definition.label, definition.binding_cells


def main() -> int:
    try:
        args = parse_args()
        repo_root = Path(__file__).resolve().parents[1]
        keymap_path = Path(args.keymap_file).expanduser().resolve()
        workspace_root = (
            Path(args.workspace_root).expanduser().resolve()
            if args.workspace_root
            else discover_workspace_root(repo_root)
        )

        behavior_lookup, binding_cells_by_label = build_behavior_catalog(workspace_root, keymap_path)
        maps = build_symbol_maps(workspace_root, keymap_path)
        attach_style_lookup_helpers(maps, workspace_root, keymap_path)

        template_blocks: list[BindingBlock] = []
        if keymap_path.exists():
            keymap_text = keymap_path.read_text(encoding="utf-8")
            template_blocks = parse_binding_blocks(keymap_text, binding_cells_by_label)
            reverse_style_overrides(keymap_path, template_blocks, maps)

        with RawSerial(args.port, baud=args.baud) as serial_port:
            client = StudioRpcClient(serial_port, timeout=args.timeout)
            device_info = client.get_device_info()
            layers = client.get_keymap()

            behavior_ids = sorted({binding.behavior_id for layer in layers for binding in layer.bindings})
            behaviors: dict[int, BehaviorInfo] = {
                0: BehaviorInfo(local_id=0, display_name="None", token="none", binding_cells=0)
            }
            for behavior_id in behavior_ids:
                if behavior_id == 0:
                    continue
                details = client.get_behavior_details(behavior_id)
                token, binding_cells = map_behavior_to_token(details["display_name"], behavior_lookup)
                behaviors[behavior_id] = BehaviorInfo(
                    local_id=behavior_id,
                    display_name=str(details["display_name"]),
                    token=token,
                    binding_cells=binding_cells,
                )

        raw_payload = {
            "device_info": device_info,
            "workspace_root": str(workspace_root) if workspace_root else None,
            "keymap_file": str(keymap_path),
            "layers": [
                {
                    "id": layer.id,
                    "name": layer.name,
                    "bindings": [
                        {
                            "behavior_id": binding.behavior_id,
                            "behavior_display_name": behaviors[binding.behavior_id].display_name,
                            "behavior_token": behaviors[binding.behavior_id].token,
                            "param1": binding.param1,
                            "param2": binding.param2,
                        }
                        for binding in layer.bindings
                    ],
                }
                for layer in layers
            ],
        }
        raw_json_path = Path(args.raw_json_out).expanduser()
        raw_json_path.parent.mkdir(parents=True, exist_ok=True)
        raw_json_path.write_text(json.dumps(raw_payload, indent=2) + "\n", encoding="utf-8")

        snippet_text = render_layers(layers, behaviors, maps)
        if args.snippet_out:
            snippet_path = Path(args.snippet_out).expanduser()
            snippet_path.parent.mkdir(parents=True, exist_ok=True)
            snippet_path.write_text(snippet_text, encoding="utf-8")

        if args.update_keymap:
            updated = update_keymap_file(keymap_path, template_blocks, layers, behaviors, maps)
            keymap_path.write_text(updated, encoding="utf-8")

        sys.stdout.write(snippet_text)
        sys.stderr.write(f"Wrote raw extract to {raw_json_path}\n")
        if args.update_keymap:
            sys.stderr.write(f"Updated {keymap_path}\n")
        return 0
    except (OSError, TimeoutError, StudioRpcError, PatchError, UnknownBehaviorError) as exc:
        sys.stderr.write(f"extract_zmk_studio_keymap.py: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
