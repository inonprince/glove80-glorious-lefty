#!/usr/bin/env python3
"""
Sync Kanata layer mappings from a ZMK Glove80 keymap.

Default behavior is a dry run:
- Parse `kanata.kbd` and `config/glove80.keymap`.
- Map Kanata key positions to ZMK positions using the Dvorak base layer.
- Propose updates for existing Kanata layers: cursor/symbol/number/function.
- Print a change report.

Use `--write` to apply changes to the Kanata file.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


SYNC_LAYER_MAP = {
    "cursor": "Cursor",
    "symbol": "Symbol",
    "number": "Number",
    "function": "Function",
    "dvorak": "Dvorak",
}

DEFAULT_SYNC_LAYERS = ("cursor", "symbol", "number", "function")

KANATA_ALIAS_BY_ZMK_LAYER = {
    "cursor": "@lk-cur",
    "symbol": "@lk-sym",
    "number": "@lk-num",
    "function": "@lk-fn",
}

# ZMK keycode -> Kanata token
KEYCODE_MAP = {
    "GRAVE": "grv",
    "SQT": "'",
    "COMMA": ",",
    "DOT": ".",
    "SLASH": "/",
    "FSLH": "/",
    "SEMI": ";",
    "MINUS": "-",
    "EQUAL": "=",
    "LBKT": "[",
    "RBKT": "]",
    "BSLH": "\\",
    "TAB": "tab",
    "SPACE": "spc",
    "ESC": "esc",
    "DEL": "del",
    "DELETE": "del",
    "INS": "ins",
    "INSERT": "ins",
    "BSPC": "bspc",
    "BACKSPACE": "bspc",
    "RET": "ret",
    "ENTER": "ret",
    "LEFT": "left",
    "RIGHT": "rght",
    "UP": "up",
    "DOWN": "down",
    "PG_DN": "pgdn",
    "PG_UP": "pgup",
    "PAGE_UP": "pgup",
    "HOME": "home",
    "END": "end",
    "LSHFT": "lsft",
    "RSHFT": "rsft",
    "LALT": "lalt",
    "RALT": "ralt",
    "LCTL": "lctl",
    "RCTL": "rctl",
    "LGUI": "lmet",
    "RGUI": "rmet",
    "N0": "0",
    "N1": "1",
    "N2": "2",
    "N3": "3",
    "N4": "4",
    "N5": "5",
    "N6": "6",
    "N7": "7",
    "N8": "8",
    "N9": "9",
    "LEFT_PINKY_MOD": "lctl",
    "LEFT_RINGY_MOD": "lalt",
    "LEFT_MIDDY_MOD": "lmet",
    "LEFT_INDEX_MOD": "lsft",
    "RIGHT_PINKY_MOD": "rctl",
    "RIGHT_RINGY_MOD": "ralt",
    "RIGHT_MIDDY_MOD": "rmet",
    "RIGHT_INDEX_MOD": "rsft",
}

MACRO_MAP = {
    "_UNDO": "M-z",
    "_REDO": "M-S-z",
    "_CUT": "M-x",
    "_COPY": "M-c",
    "_PASTE": "M-v",
    "_FIND": "M-f",
    "_FIND_NEXT": "M-g",
    "_FIND_PREV": "M-S-g",
    "_HOME": "M-left",
    "_END": "M-rght",
    "_C(A)": "M-a",
    "_C(N)": "M-n",
    "_C(S)": "M-s",
    "_C(W)": "M-w",
    "_C(Q)": "M-q",
    "_C(K)": "M-k",
    "_C(H)": "M-h",
    "_C(L)": "M-l",
}

# Aliases in kanata dvorak layer -> tap output for base matching
KANATA_BASE_ALIAS_TAP = {
    "@a_c": "a",
    "@o_a": "o",
    "@e_m": "e",
    "@u_s": "u",
    "@h_s": "h",
    "@t_m": "t",
    "@n_a": "n",
    "@s_c": "s",
    "@cap": "esc",
    "@lt": "tab",
    "@lbs": "bspc",
    "@spc": "spc",
    "@ren": "ret",
    "@rdn": "del",
}

# Prefer these Kanata source keys when multiple keys share the same tap output.
PREFERRED_KEY_FOR_TAP = {
    "tab": "lalt",   # @lt
    "bspc": "lmet",  # @lbs
    "ret": "rmet",   # @ren
}

LEFT_COL_ORDER = [6, 5, 4, 3, 2, 1]
RIGHT_COL_ORDER = [1, 2, 3, 4, 5, 6]


@dataclasses.dataclass(frozen=True)
class LayerBlock:
    name: str
    start: int
    end: int
    tokens: list[str]
    row_lengths: list[int]


@dataclasses.dataclass(frozen=True)
class KeyPosition:
    hand: str
    row: int | None = None
    col: int | None = None
    thumb: int | None = None

    @property
    def is_thumb(self) -> bool:
        return self.thumb is not None


@dataclasses.dataclass(frozen=True)
class Change:
    layer: str
    source_key: str
    base_key: str
    position: KeyPosition
    zmk_index: int
    old: str
    new: str
    zmk_binding: str


class ParseError(RuntimeError):
    pass


def compute_layer_key_stats(old_tokens: list[str], new_tokens: list[str]) -> tuple[int, int, int]:
    """
    Return (moved, added, removed) key-instance counts for a layer.

    - `_` is treated as empty and excluded from counts.
    - moved: same key token exists before/after but at different position.
    - added: key instances that only exist in new layer.
    - removed: key instances that only exist in old layer.
    """
    old_positions: dict[str, list[int]] = defaultdict(list)
    new_positions: dict[str, list[int]] = defaultdict(list)

    for i, token in enumerate(old_tokens):
        if token != "_":
            old_positions[token].append(i)
    for i, token in enumerate(new_tokens):
        if token != "_":
            new_positions[token].append(i)

    moved = 0
    added = 0
    removed = 0
    for token in set(old_positions) | set(new_positions):
        old_pos = old_positions.get(token, [])
        new_pos = new_positions.get(token, [])

        old_count = len(old_pos)
        new_count = len(new_pos)
        common = min(old_count, new_count)

        if common:
            unchanged_at_same_pos = len(set(old_pos) & set(new_pos))
            moved += common - unchanged_at_same_pos
        if new_count > old_count:
            added += new_count - old_count
        elif old_count > new_count:
            removed += old_count - new_count

    return moved, added, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync Kanata layer mappings from config/glove80.keymap"
    )
    parser.add_argument("--kanata", default="kanata.kbd", help="Path to Kanata file")
    parser.add_argument(
        "--zmk", default="config/glove80.keymap", help="Path to ZMK keymap file"
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        default=list(DEFAULT_SYNC_LAYERS),
        help="Kanata layers to sync (default: cursor symbol number function)",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Also sync dvorak base layer (off by default)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes back to the Kanata file",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Only print final status line",
    )
    parser.add_argument(
        "--no-thumb",
        action="store_true",
        help="Do not update thumb assignments",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise ParseError(f"Failed to read {path}: {exc}") from exc


def extract_paren_body(text: str, body_start: int) -> tuple[str, int]:
    depth = 1
    i = body_start
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[body_start:i], i
        i += 1
    raise ParseError("Unbalanced parentheses while parsing Kanata file")


def tokenize_space_separated(text: str) -> list[str]:
    return re.findall(r"[^\s()]+", text)


def parse_rows(body: str) -> list[int]:
    row_lengths: list[int] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(";;"):
            continue
        row_tokens = tokenize_space_separated(line)
        if row_tokens:
            row_lengths.append(len(row_tokens))
    return row_lengths


def parse_kanata_file(path: Path) -> tuple[str, list[str], list[int], dict[str, LayerBlock]]:
    text = read_text(path)

    m_src = re.search(r"\(defsrc\s*", text)
    if not m_src:
        raise ParseError("Could not find (defsrc ...) in Kanata file")
    defsrc_body, _ = extract_paren_body(text, m_src.end())
    defsrc_tokens = tokenize_space_separated(defsrc_body)
    defsrc_rows = parse_rows(defsrc_body)

    layers: dict[str, LayerBlock] = {}
    for match in re.finditer(r"\(deflayer\s+([A-Za-z0-9_-]+)\s*", text):
        name = match.group(1)
        body, close_index = extract_paren_body(text, match.end())
        tokens = tokenize_space_separated(body)
        row_lengths = parse_rows(body)
        if sum(row_lengths) != len(tokens):
            row_lengths = list(defsrc_rows)
        layers[name] = LayerBlock(
            name=name,
            start=match.start(),
            end=close_index + 1,
            tokens=tokens,
            row_lengths=row_lengths,
        )

    if not layers:
        raise ParseError("Could not find any (deflayer ...) blocks in Kanata file")
    return text, defsrc_tokens, defsrc_rows, layers


def parse_zmk_layers(path: Path) -> dict[str, list[str]]:
    text = read_text(path)
    result: dict[str, list[str]] = {}
    for match in re.finditer(r"\blayer_([A-Za-z0-9_]+)\s*\{\s*bindings\s*=\s*<", text):
        layer_name = match.group(1)
        start = match.end()
        end = text.find(">;", start)
        if end == -1:
            continue
        body = text[start:end]
        clean_lines = [line.split("//", 1)[0] for line in body.splitlines()]
        tokens: list[str] = []
        for line in clean_lines:
            line = line.strip()
            if not line:
                continue
            for part in re.split(r"\s{2,}", line):
                part = part.strip()
                if part:
                    tokens.append(part)
        result[layer_name] = tokens
    return result


def convert_kp_code(code: str) -> str | None:
    if code in MACRO_MAP:
        return MACRO_MAP[code]
    shifted = re.fullmatch(r"LS\((.+)\)", code)
    if shifted:
        inner = convert_kp_code(shifted.group(1))
        return None if inner is None else f"S-{inner}"
    if code in KEYCODE_MAP:
        return KEYCODE_MAP[code]
    letter = re.fullmatch(r"[A-Z]", code)
    if letter:
        return code.lower()
    fn = re.fullmatch(r"F(\d+)", code)
    if fn:
        return f"f{fn.group(1)}"
    return None


def convert_zmk_binding(binding: str) -> tuple[str | None, str | None]:
    binding = binding.strip()

    if binding in {"&none", "&trans"}:
        return "_", None

    kp = re.fullmatch(r"&kp\s+(.+)", binding)
    if kp:
        converted = convert_kp_code(kp.group(1))
        return converted, None if converted is not None else "unsupported keycode"

    sk = re.fullmatch(r"&sk\s+([A-Za-z0-9_]+)", binding)
    if sk:
        converted = convert_kp_code(sk.group(1))
        return converted, None if converted is not None else "unsupported sticky key"

    tog = re.fullmatch(r"&tog\s+LAYER_([A-Za-z0-9_]+)", binding)
    if tog:
        layer_key = tog.group(1).lower()
        alias = KANATA_ALIAS_BY_ZMK_LAYER.get(layer_key)
        return alias, None if alias is not None else "unsupported layer toggle"

    thumb = re.fullmatch(r"&thumb\s+LAYER_[A-Za-z0-9_]+\s+([A-Za-z0-9_]+)", binding)
    if thumb:
        converted = convert_kp_code(thumb.group(1))
        return converted, None if converted is not None else "unsupported thumb keycode"

    space = re.fullmatch(r"&space\s+LAYER_[A-Za-z0-9_]+\s+([A-Za-z0-9_]+)", binding)
    if space:
        converted = convert_kp_code(space.group(1))
        return converted, None if converted is not None else "unsupported space keycode"

    sticky = re.fullmatch(
        r"&sticky_key_modtap\s+([A-Za-z0-9_]+)\s+[A-Za-z0-9_]+", binding
    )
    if sticky:
        converted = convert_kp_code(sticky.group(1))
        return converted, None if converted is not None else "unsupported sticky modtap"

    return None, "unsupported behavior"


def zmk_base_tap(binding: str) -> str | None:
    kp = re.fullmatch(r"&kp\s+(.+)", binding)
    if kp:
        return convert_kp_code(kp.group(1))

    left_hrm = re.fullmatch(r"&Left[A-Za-z0-9_]+\s+\(([A-Z]+),\s*LAYER_Dvorak\)", binding)
    if left_hrm:
        return left_hrm.group(1).lower()
    right_hrm = re.fullmatch(
        r"&Right[A-Za-z0-9_]+\s+\(([A-Z]+),\s*LAYER_Dvorak\)", binding
    )
    if right_hrm:
        return right_hrm.group(1).lower()

    sticky = re.fullmatch(
        r"&sticky_key_modtap\s+([A-Za-z0-9_]+)\s+[A-Za-z0-9_]+", binding
    )
    if sticky:
        return convert_kp_code(sticky.group(1))

    thumb = re.fullmatch(r"&thumb\s+LAYER_[A-Za-z0-9_]+\s+([A-Za-z0-9_]+)", binding)
    if thumb:
        return convert_kp_code(thumb.group(1))

    space = re.fullmatch(r"&space\s+LAYER_[A-Za-z0-9_]+\s+([A-Za-z0-9_]+)", binding)
    if space:
        return convert_kp_code(space.group(1))

    return None


def kanata_base_tap(token: str) -> str:
    if token in KANATA_BASE_ALIAS_TAP:
        return KANATA_BASE_ALIAS_TAP[token]
    return token


def build_position_map(
    defsrc: list[str], kanata_dvorak: list[str], zmk_dvorak: list[str]
) -> tuple[dict[int, int], list[tuple[str, str]]]:
    by_tap: dict[str, list[int]] = defaultdict(list)
    for idx, binding in enumerate(zmk_dvorak):
        tap = zmk_base_tap(binding)
        if tap is not None:
            by_tap[tap].append(idx)

    positions_by_tap: dict[str, list[int]] = defaultdict(list)
    for pos, token in enumerate(kanata_dvorak):
        tap = kanata_base_tap(token)
        positions_by_tap[tap].append(pos)

    mapped: dict[int, int] = {}
    unmapped: list[tuple[str, str]] = []
    for tap, positions in positions_by_tap.items():
        indices = by_tap.get(tap, [])
        if len(indices) != 1:
            for pos in positions:
                unmapped.append((defsrc[pos], tap))
            continue

        chosen_pos: int | None = None
        preferred_key = PREFERRED_KEY_FOR_TAP.get(tap)
        if preferred_key is not None:
            for pos in positions:
                if defsrc[pos] == preferred_key:
                    chosen_pos = pos
                    break
        if chosen_pos is None:
            chosen_pos = positions[0]

        mapped[chosen_pos] = indices[0]
        for pos in positions:
            if pos != chosen_pos:
                unmapped.append((defsrc[pos], tap))
    return mapped, unmapped


def build_zmk_index_positions() -> dict[int, KeyPosition]:
    positions: dict[int, KeyPosition] = {}
    idx = 0

    def add_row(row: int, left_cols: list[int], right_cols: list[int]) -> None:
        nonlocal idx
        for col in left_cols:
            positions[idx] = KeyPosition(hand="left", row=row, col=col)
            idx += 1
        for col in right_cols:
            positions[idx] = KeyPosition(hand="right", row=row, col=col)
            idx += 1

    add_row(1, [6, 5, 4, 3, 2], [2, 3, 4, 5, 6])
    add_row(2, [6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6])
    add_row(3, [6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6])
    add_row(4, [6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6])

    for col in [6, 5, 4, 3, 2, 1]:
        positions[idx] = KeyPosition(hand="left", row=5, col=col)
        idx += 1
    for thumb in [1, 2, 3]:
        positions[idx] = KeyPosition(hand="left", thumb=thumb)
        idx += 1
    for thumb in [3, 2, 1]:
        positions[idx] = KeyPosition(hand="right", thumb=thumb)
        idx += 1
    for col in [1, 2, 3, 4, 5, 6]:
        positions[idx] = KeyPosition(hand="right", row=5, col=col)
        idx += 1

    for col in [6, 5, 4, 3, 2]:
        positions[idx] = KeyPosition(hand="left", row=6, col=col)
        idx += 1
    for thumb in [4, 5, 6]:
        positions[idx] = KeyPosition(hand="left", thumb=thumb)
        idx += 1
    for thumb in [6, 5, 4]:
        positions[idx] = KeyPosition(hand="right", thumb=thumb)
        idx += 1
    for col in [2, 3, 4, 5, 6]:
        positions[idx] = KeyPosition(hand="right", row=6, col=col)
        idx += 1

    if idx != 80:
        raise ParseError(f"Internal error: built {idx} ZMK positions, expected 80")
    return positions


def format_position(position: KeyPosition) -> str:
    if position.is_thumb:
        return f"{position.hand} hand t{position.thumb}"
    return f"{position.hand} hand r{position.row}c{position.col}"


def build_coord_to_zidx(zmk_index_positions: dict[int, KeyPosition]) -> dict[tuple, int]:
    coord_to_idx: dict[tuple, int] = {}
    for idx, pos in zmk_index_positions.items():
        coord_to_idx[(pos.hand, pos.row, pos.col, pos.thumb)] = idx
    return coord_to_idx


def _format_cell(token: str, width: int) -> str:
    if token == "":
        return " " * width
    if len(token) > width:
        return token[: width - 1] + "~"
    return token.ljust(width)


def _join_cells(tokens: list[str], width: int) -> str:
    return " ".join(f"|{_format_cell(token, width)}|" for token in tokens)


def render_layer_ascii(
    layer_name: str,
    layer_tokens: list[str],
    mapped_positions: dict[int, int],
    coord_to_zidx: dict[tuple, int],
    cell_width: int = 8,
) -> str:
    # Project Kanata tokens onto ZMK matrix positions where mapping exists.
    zmk_token_by_idx = {idx: "." for idx in range(80)}
    for src_pos, zidx in mapped_positions.items():
        zmk_token_by_idx[zidx] = layer_tokens[src_pos]

    def token_at(
        hand: str,
        row: int | None = None,
        col: int | None = None,
        thumb: int | None = None,
    ) -> str:
        idx = coord_to_zidx.get((hand, row, col, thumb))
        if idx is None:
            # No physical key at this matrix location (e.g. r1c1/r6c1).
            return ""
        return zmk_token_by_idx.get(idx, ".")

    lines: list[str] = [f"  {layer_name}:"]
    for row in range(1, 7):
        left = [token_at("left", row=row, col=col) for col in LEFT_COL_ORDER]
        right = [token_at("right", row=row, col=col) for col in RIGHT_COL_ORDER]
        lines.append(
            f"    {'LH r' + str(row):<7} {_join_cells(left, cell_width)}    "
            f"{_join_cells(right, cell_width)} {'RH r' + str(row):>7}"
        )

    left_top = [token_at("left", thumb=t) for t in [1, 2, 3]]
    right_top = [token_at("right", thumb=t) for t in [1, 2, 3]]
    left_bottom = [token_at("left", thumb=t) for t in [4, 5, 6]]
    right_bottom = [token_at("right", thumb=t) for t in [4, 5, 6]]
    lines.append(
        f"    {'LH t1-3':<7} {_join_cells(left_top, cell_width)}    "
        f"{_join_cells(right_top, cell_width)} {'RH t1-3':>7}"
    )
    lines.append(
        f"    {'LH t4-6':<7} {_join_cells(left_bottom, cell_width)}    "
        f"{_join_cells(right_bottom, cell_width)} {'RH t4-6':>7}"
    )
    return "\n".join(lines)


def render_layer(name: str, tokens: list[str], row_lengths: list[int]) -> str:
    if sum(row_lengths) != len(tokens):
        raise ParseError(
            f"Layer {name}: row length mismatch (rows={sum(row_lengths)}, tokens={len(tokens)})"
        )
    lines = [f"(deflayer {name}"]
    i = 0
    for row_len in row_lengths:
        row = " ".join(tokens[i : i + row_len])
        lines.append(f"  {row}")
        i += row_len
    lines.append(")")
    return "\n".join(lines)


def apply_layer_updates(
    text: str, layers: dict[str, LayerBlock], updates: dict[str, list[str]]
) -> str:
    out = text
    replacements: list[tuple[int, int, str]] = []
    for layer_name, new_tokens in updates.items():
        block = layers[layer_name]
        replacement = render_layer(layer_name, new_tokens, block.row_lengths)
        replacements.append((block.start, block.end, replacement))

    for start, end, replacement in sorted(replacements, key=lambda t: t[0], reverse=True):
        out = out[:start] + replacement + out[end:]
    return out


def main() -> int:
    args = parse_args()
    kanata_path = Path(args.kanata)
    zmk_path = Path(args.zmk)

    layers_to_sync = list(args.layers)
    if args.include_base and "dvorak" not in layers_to_sync:
        layers_to_sync.append("dvorak")

    unknown = [name for name in layers_to_sync if name not in SYNC_LAYER_MAP]
    if unknown:
        raise ParseError(
            f"Unknown layer(s): {', '.join(unknown)}. Known: {', '.join(sorted(SYNC_LAYER_MAP))}"
        )

    kanata_text, defsrc, _, kanata_layers = parse_kanata_file(kanata_path)
    zmk_layers = parse_zmk_layers(zmk_path)
    zmk_index_positions = build_zmk_index_positions()
    coord_to_zidx = build_coord_to_zidx(zmk_index_positions)

    required_kanata_layers = set(layers_to_sync) | {"dvorak"}
    missing_kanata = [name for name in required_kanata_layers if name not in kanata_layers]
    if missing_kanata:
        raise ParseError(f"Kanata file missing layer(s): {', '.join(sorted(missing_kanata))}")

    required_zmk_layers = {SYNC_LAYER_MAP[name] for name in required_kanata_layers}
    missing_zmk = [name for name in required_zmk_layers if name not in zmk_layers]
    if missing_zmk:
        raise ParseError(f"ZMK file missing layer(s): {', '.join(sorted(missing_zmk))}")

    kanata_dvorak = kanata_layers["dvorak"].tokens
    zmk_dvorak = zmk_layers["Dvorak"]
    mapped_positions, unmapped_keys = build_position_map(defsrc, kanata_dvorak, zmk_dvorak)
    zmk_base_key_by_index = {
        idx: zmk_base_tap(binding) for idx, binding in enumerate(zmk_dvorak)
    }

    updates: dict[str, list[str]] = {}
    changes: list[Change] = []
    unsupported_bindings: Counter[str] = Counter()
    layer_stats: dict[str, tuple[int, int, int]] = {}

    for kanata_layer_name in layers_to_sync:
        zmk_layer_name = SYNC_LAYER_MAP[kanata_layer_name]
        kanata_block = kanata_layers[kanata_layer_name]
        zmk_tokens = zmk_layers[zmk_layer_name]
        if len(kanata_block.tokens) != len(defsrc):
            raise ParseError(
                f"Kanata layer {kanata_layer_name} has {len(kanata_block.tokens)} tokens; expected {len(defsrc)}"
            )
        if len(zmk_tokens) != 80:
            raise ParseError(
                f"ZMK layer {zmk_layer_name} has {len(zmk_tokens)} tokens; expected 80"
            )

        new_tokens = list(kanata_block.tokens)
        for pos, zidx in mapped_positions.items():
            position = zmk_index_positions[zidx]
            if args.no_thumb and position.is_thumb:
                continue
            converted, reason = convert_zmk_binding(zmk_tokens[zidx])
            if converted is None:
                unsupported_bindings[f"{zmk_layer_name}:{zmk_tokens[zidx]} ({reason})"] += 1
                continue
            old = new_tokens[pos]
            if old != converted:
                new_tokens[pos] = converted
                changes.append(
                    Change(
                        layer=kanata_layer_name,
                        source_key=defsrc[pos],
                        base_key=zmk_base_key_by_index.get(zidx) or defsrc[pos],
                        position=position,
                        zmk_index=zidx,
                        old=old,
                        new=converted,
                        zmk_binding=zmk_tokens[zidx],
                    )
                )
        updates[kanata_layer_name] = new_tokens
        layer_stats[kanata_layer_name] = compute_layer_key_stats(
            kanata_block.tokens, new_tokens
        )

    if not args.no_report:
        print(f"Kanata file: {kanata_path}")
        print(f"ZMK file:    {zmk_path}")
        print(f"Mapped positions: {len(mapped_positions)}/{len(defsrc)}")
        if args.no_thumb:
            print("Thumb updates: disabled (--no-thumb)")
        if unmapped_keys:
            unmapped_desc = ", ".join(f"{key}({tap})" for key, tap in unmapped_keys)
            print(f"Unmapped source keys: {unmapped_desc}")
        print()

        change_counts = Counter(change.layer for change in changes)
        for name in layers_to_sync:
            moved, added, removed = layer_stats.get(name, (0, 0, 0))
            print(
                f"{name}: {change_counts.get(name, 0)} change(s), "
                f"moved={moved}, new={added}, removed={removed}"
            )
        print()

        print("ASCII layers (post-sync projected view; blank=no physical key, '.'=no mapped source key):")
        for layer_name in layers_to_sync:
            if layer_name not in updates:
                continue
            print(
                render_layer_ascii(
                    layer_name=layer_name,
                    layer_tokens=updates[layer_name],
                    mapped_positions=mapped_positions,
                    coord_to_zidx=coord_to_zidx,
                )
            )
        print()

        if changes:
            print("Changed assignments by layer:")
            for layer_name in layers_to_sync:
                layer_changes = [c for c in changes if c.layer == layer_name]
                if not layer_changes:
                    continue
                print(f"  {layer_name}:")
                for change in sorted(layer_changes, key=lambda c: c.zmk_index):
                    print(
                        f"    {format_position(change.position)} ({change.base_key}): "
                        f"{change.old} -> {change.new} (from `{change.zmk_binding}`)"
                    )
            print()
        else:
            print("No changes needed.")

        if unsupported_bindings:
            print("Unsupported ZMK bindings kept unchanged:")
            for binding, count in unsupported_bindings.most_common():
                print(f"  {count}x {binding}")
            print()

    if args.write and changes:
        new_text = apply_layer_updates(kanata_text, kanata_layers, updates)
        kanata_path.write_text(new_text)
        print(f"Wrote {len(changes)} update(s) to {kanata_path}")
    elif args.write:
        print("Nothing to write.")
    else:
        print("Dry run complete. Re-run with --write to apply changes.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ParseError as err:
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(2)
