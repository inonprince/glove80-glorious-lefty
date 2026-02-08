#!/usr/bin/env python3
"""Update KLE layout exports to match a new Glove80 keymap JSON.

Reads KLE files from `--kle-in` (unmodified reference) and writes updated
layouts to `--kle-out` without changing geometry. When keys are moved
"as-is" between the old and new keymaps, this script moves the full KLE
content (label + style) for that key.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROW_MAP: Dict[int, Dict[str, List[Optional[int]]]] = {
    1: {"left": [0, 1, 2, 3, 4, None], "right": [None, 5, 6, 7, 8, 9]},
    2: {"left": [10, 11, 12, 13, 14, 15], "right": [16, 17, 18, 19, 20, 21]},
    3: {"left": [22, 23, 24, 25, 26, 27], "right": [28, 29, 30, 31, 32, 33]},
    4: {"left": [34, 35, 36, 37, 38, 39], "right": [40, 41, 42, 43, 44, 45]},
    5: {"left": [46, 47, 48, 49, 50, 51], "right": [58, 59, 60, 61, 62, 63]},
    6: {"left": [64, 65, 66, 67, 68, None], "right": [None, 75, 76, 77, 78, 79]},
}

X_LEFT = [1, 2, 3, 4, 5, 6]
X_RIGHT = [14.25, 15.25, 16.25, 17.25, 18.25, 19.25]

THUMB_INDICES: Dict[Tuple[str, int], Tuple[int, int]] = {
    ("L", 25): (52, 69),
    ("L", 35): (53, 70),
    ("L", 45): (54, 71),
    ("R", 25): (57, 74),
    ("R", 35): (56, 73),
    ("R", 45): (55, 72),
}

THUMB_IGNORE_LABELS = {"T1", "T2", "T3", "T4", "T5", "T6"}

EPS = 1e-6

# KLE properties that apply to a single key only and should not persist.
# `d` (decal) and `n` (nub) are the known culprits that must not leak.
NON_STICKY_STYLE_KEYS = {"d", "n", "l", "i"}


@dataclass
class Key:
    label: str
    x: float
    y: float
    w: float
    h: float
    r: float
    rx: float
    ry: float
    x2: float
    y2: float
    w2: float
    h2: float
    style: Dict[str, Any]


@dataclass
class KLELayout:
    rows: List[List[Key]]
    elements: List[Tuple[str, Any]]  # ("meta", obj) or ("row", row_idx)


@dataclass
class KeyContent:
    label: str
    style: Dict[str, Any]


class ParseState:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.w = 1.0
        self.h = 1.0
        self.x2 = 0.0
        self.y2 = 0.0
        self.w2 = 1.0
        self.h2 = 1.0
        self.r = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.style: Dict[str, Any] = {}
        self.oneshot_style: Dict[str, Any] = {}

    def reset_key_size(self) -> None:
        self.w = 1.0
        self.h = 1.0
        self.x2 = 0.0
        self.y2 = 0.0
        self.w2 = 1.0
        self.h2 = 1.0

    def reset_oneshot_style(self) -> None:
        self.oneshot_style = {}


def parse_kle(path: Path) -> KLELayout:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[List[Key]] = []
    elements: List[Tuple[str, Any]] = []

    state = ParseState()
    first_row = True

    for elem in data:
        if not isinstance(elem, list):
            elements.append(("meta", elem))
            continue

        row_idx = len(rows)
        elements.append(("row", row_idx))

        row_keys: List[Key] = []
        state.x = 0.0
        if first_row:
            state.y = 0.0
            first_row = False
        else:
            state.y += 1.0

        for item in elem:
            if isinstance(item, dict):
                if "r" in item or "rx" in item or "ry" in item:
                    if "r" in item:
                        state.r = item["r"]
                    if "rx" in item:
                        state.rx = item["rx"]
                    if "ry" in item:
                        state.ry = item["ry"]
                    # Rotation resets the cursor to the rotation origin.
                    state.x = state.rx
                    state.y = state.ry

                if "x" in item:
                    state.x += item["x"]
                if "y" in item:
                    state.y += item["y"]

                if "w" in item:
                    state.w = item["w"]
                if "h" in item:
                    state.h = item["h"]
                if "x2" in item:
                    state.x2 = item["x2"]
                if "y2" in item:
                    state.y2 = item["y2"]
                if "w2" in item:
                    state.w2 = item["w2"]
                if "h2" in item:
                    state.h2 = item["h2"]

                for key, value in item.items():
                    if key in {"x", "y", "w", "h", "x2", "y2", "w2", "h2", "r", "rx", "ry"}:
                        continue
                    if key in NON_STICKY_STYLE_KEYS:
                        state.oneshot_style[key] = value
                    else:
                        state.style[key] = value

                # In KLE, setting `f` resets the per-label font array `fa`.
                if "f" in item and "fa" not in item:
                    state.style.pop("fa", None)
            else:
                combined_style = dict(state.style)
                combined_style.update(state.oneshot_style)
                row_keys.append(
                    Key(
                        label=item,
                        x=state.x,
                        y=state.y,
                        w=state.w,
                        h=state.h,
                        r=state.r,
                        rx=state.rx,
                        ry=state.ry,
                        x2=state.x2,
                        y2=state.y2,
                        w2=state.w2,
                        h2=state.h2,
                        style=combined_style,
                    )
                )
                state.x += state.w
                state.reset_key_size()
                state.reset_oneshot_style()

        rows.append(row_keys)

    return KLELayout(rows=rows, elements=elements)


def serialize_kle(layout: KLELayout) -> List[Any]:
    output: List[Any] = []

    class SerState:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.r = 0.0
            self.rx = 0.0
            self.ry = 0.0

    state = SerState()
    first_row = True

    def near_zero(val: float) -> bool:
        return abs(val) < EPS

    row_iter = iter(layout.rows)
    row_cache = list(layout.rows)

    for elem_type, elem in layout.elements:
        if elem_type == "meta":
            output.append(elem)
            continue

        row_idx = elem
        row = row_cache[row_idx]

        state.x = 0.0
        if first_row:
            state.y = 0.0
            first_row = False
        else:
            state.y += 1.0

        row_items: List[Any] = []

        for key in row:
            # Rotation block if needed.
            if (
                abs(key.r - state.r) > EPS
                or abs(key.rx - state.rx) > EPS
                or abs(key.ry - state.ry) > EPS
            ):
                rot_block: Dict[str, Any] = {}
                if abs(key.r - state.r) > EPS:
                    rot_block["r"] = key.r
                if abs(key.rx - state.rx) > EPS:
                    rot_block["rx"] = key.rx
                if abs(key.ry - state.ry) > EPS:
                    rot_block["ry"] = key.ry
                row_items.append(rot_block)
                state.r = key.r
                state.rx = key.rx
                state.ry = key.ry
                state.x = state.rx
                state.y = state.ry

            dx = key.x - state.x
            dy = key.y - state.y

            props: Dict[str, Any] = {}
            props.update(key.style)

            if not near_zero(dx):
                props["x"] = dx
            if not near_zero(dy):
                props["y"] = dy
            if abs(key.w - 1.0) > EPS:
                props["w"] = key.w
            if abs(key.h - 1.0) > EPS:
                props["h"] = key.h
            if abs(key.x2) > EPS:
                props["x2"] = key.x2
            if abs(key.y2) > EPS:
                props["y2"] = key.y2
            if abs(key.w2 - 1.0) > EPS:
                props["w2"] = key.w2
            if abs(key.h2 - 1.0) > EPS:
                props["h2"] = key.h2

            if props:
                row_items.append(props)
            row_items.append(key.label)

            state.x = key.x + key.w
            state.y = key.y

        output.append(row_items)

    return output


def signature(slot: Dict[str, Any]) -> str:
    return json.dumps(slot, sort_keys=True, separators=(",", ":"))


def normalized_layer_name(layer_name: str) -> str:
    return layer_name.lower().replace("_", "-")


def layer_name_from_filename(stem: str, layer_names: List[str]) -> Optional[str]:
    if stem.startswith("base-layer-diagram-"):
        suffix = stem[len("base-layer-diagram-") :]
        for name in layer_names:
            if name.lower() == suffix.lower():
                return name
        return suffix
    if stem == "base-layer-diagram":
        return "Dvorak" if "Dvorak" in layer_names else (layer_names[0] if layer_names else None)

    if stem.endswith("-layer-diagram"):
        base = stem[: -len("-layer-diagram")]
    elif stem.endswith("-diagram"):
        base = stem[: -len("-diagram")]
    else:
        base = stem

    base_norm = base.lower()
    for name in layer_names:
        if normalized_layer_name(name) == base_norm:
            return name
    return None


def find_x_index(x: float, positions: List[float]) -> Optional[int]:
    for idx, pos in enumerate(positions):
        if abs(x - pos) < 0.02:
            return idx
    return None


def map_indices(layout: KLELayout) -> Tuple[Dict[int, Key], List[str]]:
    mapped: Dict[int, Key] = {}
    warnings: List[str] = []

    all_keys = [key for row in layout.rows for key in row]

    for key in all_keys:
        if abs(key.r) > EPS:
            continue
        row = math.floor(key.y + EPS)
        if row not in ROW_MAP:
            continue
        left_idx = find_x_index(key.x, X_LEFT)
        right_idx = find_x_index(key.x, X_RIGHT)
        if left_idx is None and right_idx is None:
            continue
        if left_idx is not None:
            idx = ROW_MAP[row]["left"][left_idx]
        else:
            idx = ROW_MAP[row]["right"][right_idx]
        if idx is None:
            continue
        mapped[idx] = key

    # Thumbs (rotated keys). Ignore placeholder T1..T6 labels.
    rotated = [
        key
        for key in all_keys
        if abs(key.r) > EPS and key.label not in THUMB_IGNORE_LABELS
    ]

    groups: Dict[Tuple[str, int], List[Key]] = {}
    for key in rotated:
        rot_key = int(round(key.r / 5.0) * 5)
        if rot_key == 0:
            continue
        side = "L" if rot_key > 0 else "R"
        group_key = (side, abs(rot_key))
        groups.setdefault(group_key, []).append(key)

    for group_key, keys in groups.items():
        keys_sorted = sorted(keys, key=lambda k: k.y)
        if len(keys_sorted) < 2:
            warnings.append(
                f"thumb group {group_key} expected 2 keys, found {len(keys_sorted)}"
            )
            continue
        if len(keys_sorted) > 2:
            warnings.append(
                f"thumb group {group_key} expected 2 keys, found {len(keys_sorted)} (using top/bottom)"
            )
        if group_key not in THUMB_INDICES:
            warnings.append(f"thumb group {group_key} not in index map")
            continue
        idx_top, idx_bottom = THUMB_INDICES[group_key]
        mapped[idx_top] = keys_sorted[0]
        mapped[idx_bottom] = keys_sorted[-1]

    return mapped, warnings


def build_move_map(old_sigs: List[str], new_sigs: List[str]) -> Dict[int, int]:
    old_positions: Dict[str, List[int]] = {}
    new_positions: Dict[str, List[int]] = {}

    for idx, sig in enumerate(old_sigs):
        old_positions.setdefault(sig, []).append(idx)
    for idx, sig in enumerate(new_sigs):
        new_positions.setdefault(sig, []).append(idx)

    move_map: Dict[int, int] = {}
    for sig, new_idxs in new_positions.items():
        old_idxs = old_positions.get(sig, [])
        for old_idx, new_idx in zip(old_idxs, new_idxs):
            move_map[new_idx] = old_idx

    return move_map


def build_kp_label_map(sig_to_content: Dict[str, KeyContent], sig_to_slot: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    kp_map: Dict[str, str] = {}
    for sig, content in sig_to_content.items():
        slot = sig_to_slot.get(sig)
        if not slot:
            continue
        if slot.get("value") != "&kp":
            continue
        params = slot.get("params") or []
        if not params:
            continue
        keycode = params[0].get("value")
        if isinstance(keycode, str) and keycode not in kp_map:
            kp_map[keycode] = content.label
    return kp_map


def keycode_to_label(keycode: str, kp_label_map: Dict[str, str]) -> str:
    keycode = keycode.strip()
    if keycode in kp_label_map:
        return kp_label_map[keycode]

    fallback_map = {
        "ESC": "Escape",
        "RET": "Enter",
        "ENTER": "Enter",
        "BSPC": "Back space",
        "BACKSPACE": "Back space",
        "DEL": "Delete",
        "DELETE": "Delete",
        "SPACE": "Space",
        "TAB": "Tab",
    }
    return fallback_map.get(keycode, keycode)


def generate_label(slot: Dict[str, Any], kp_label_map: Dict[str, str]) -> str:
    value = slot.get("value")
    params = slot.get("params") or []

    if value == "&none":
        return ""

    if value == "&kp" and params:
        keycode = params[0].get("value", "")
        if isinstance(keycode, str):
            return keycode_to_label(keycode, kp_label_map)

    if value == "Custom" and params:
        expr = params[0].get("value", "")
        if not isinstance(expr, str):
            return "Custom"
        expr = expr.strip()

        # Custom shortcut labels used in cursor layer.
        ctrl_map = {
            "&kp _C(A)": "Select all",
            "&kp _C(S)": "Save",
            "&kp _C(N)": "New",
            "&kp _C(W)": "Close",
            "&kp _C(Q)": "Quit",
        }
        if expr in ctrl_map:
            return ctrl_map[expr]

        parts = re.split(r"\s+", expr)
        if parts:
            head = parts[0]
            if head in {"&thumb", "&space"} and len(parts) >= 3:
                layer = parts[1]
                key = parts[2]
                layer_name = layer.replace("LAYER_", "")
                key = key.strip("(),")
                return f"{keycode_to_label(key, kp_label_map)}\n\n\n\n{layer_name}"

        # Fallback: shorten custom expression.
        if expr:
            return expr

    return "Custom"


def update_layout(
    layout: KLELayout,
    old_layer_idx: int,
    new_layer_idx: int,
    old_layers: List[List[Dict[str, Any]]],
    new_layers: List[List[Dict[str, Any]]],
    sig_to_content: Dict[str, KeyContent],
    kp_label_map: Dict[str, str],
) -> Tuple[Dict[str, int], List[str]]:
    index_to_key, warnings = map_indices(layout)

    old_layer = old_layers[old_layer_idx]
    new_layer = new_layers[new_layer_idx]

    old_sigs = [signature(slot) for slot in old_layer]
    new_sigs = [signature(slot) for slot in new_layer]

    move_map = build_move_map(old_sigs, new_sigs)

    old_content_by_idx: Dict[int, KeyContent] = {
        idx: KeyContent(label=key.label, style=dict(key.style))
        for idx, key in index_to_key.items()
    }

    stats = {
        "updated": 0,
        "moved": 0,
        "reused": 0,
        "from_global": 0,
        "generated": 0,
        "missing_in_kle": 80 - len(index_to_key),
    }

    for idx, key in index_to_key.items():
        new_sig = new_sigs[idx]
        content: Optional[KeyContent] = None

        if idx in move_map:
            src_idx = move_map[idx]
            content = old_content_by_idx.get(src_idx)
            stats["reused"] += 1
            if src_idx != idx:
                stats["moved"] += 1

        if content is None:
            content = sig_to_content.get(new_sig)
            if content is not None:
                stats["from_global"] += 1

        if content is None:
            label = generate_label(new_layer[idx], kp_label_map)
            style = old_content_by_idx.get(idx, KeyContent("", {})).style
            content = KeyContent(label=label, style=dict(style))
            stats["generated"] += 1

        key.label = content.label
        key.style = dict(content.style)
        stats["updated"] += 1

    return stats, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update KLE exports to match a new keymap JSON.")
    parser.add_argument(
        "--old-keymap",
        dest="old_keymap",
        default="Glorious Engrammer v42-rc9 (unmodified-reference).json",
        help="Path to the reference keymap JSON (unmodified).",
    )
    parser.add_argument(
        "--new-keymap",
        dest="new_keymap",
        default="Glorious Engrammer v42-rc9 (working-copy).json",
        help="Path to the updated keymap JSON (working copy).",
    )
    parser.add_argument(
        "--old",
        dest="old_keymap",
        help="(Deprecated) Same as --old-keymap.",
    )
    parser.add_argument(
        "--new",
        dest="new_keymap",
        help="(Deprecated) Same as --new-keymap.",
    )
    parser.add_argument(
        "--kle-in",
        default="kle_layout-unmodified-reference",
        help="Directory with reference KLE JSON files.",
    )
    parser.add_argument(
        "--kle-out",
        default="kle_layouts",
        help="Directory to write updated KLE JSON files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    old_path = Path(args.old_keymap)
    new_path = Path(args.new_keymap)
    kle_in = Path(args.kle_in)
    kle_out = Path(args.kle_out)

    if not old_path.exists():
        print(f"Missing old keymap: {old_path}", file=sys.stderr)
        return 1
    if not new_path.exists():
        print(f"Missing new keymap: {new_path}", file=sys.stderr)
        return 1
    if not kle_in.exists():
        print(f"Missing KLE input directory: {kle_in}", file=sys.stderr)
        return 1

    old_json = json.loads(old_path.read_text(encoding="utf-8"))
    new_json = json.loads(new_path.read_text(encoding="utf-8"))

    old_layer_names = old_json.get("layer_names", [])
    new_layer_names = new_json.get("layer_names", [])
    if new_layer_names != old_layer_names:
        print("Warning: layer_names differ between old and new keymaps.")

    old_layers = old_json.get("layers", [])
    new_layers = new_json.get("layers", [])

    if len(old_layers) != len(new_layers):
        print("Warning: layer count differs between old and new keymaps.")

    kle_out.mkdir(parents=True, exist_ok=True)

    old_name_to_idx = {name: i for i, name in enumerate(old_layer_names)}
    new_name_to_idx = {name: i for i, name in enumerate(new_layer_names)}

    # Build global signature -> content map from old KLE files.
    sig_to_content: Dict[str, KeyContent] = {}
    sig_to_slot: Dict[str, Dict[str, Any]] = {}

    for path in sorted(kle_in.glob("*.json")):
        layer_name = layer_name_from_filename(path.stem, old_layer_names)
        if layer_name not in old_name_to_idx:
            print(f"Skipping {path.name}: cannot map to layer name", file=sys.stderr)
            continue
        layer_idx = old_name_to_idx[layer_name]
        layout = parse_kle(path)
        index_to_key, _ = map_indices(layout)

        for idx, key in index_to_key.items():
            if layer_idx >= len(old_layers) or idx >= len(old_layers[layer_idx]):
                continue
            slot = old_layers[layer_idx][idx]
            sig = signature(slot)
            if sig not in sig_to_content:
                sig_to_content[sig] = KeyContent(label=key.label, style=dict(key.style))
                sig_to_slot[sig] = slot

    kp_label_map = build_kp_label_map(sig_to_content, sig_to_slot)

    # Update each KLE file.
    print("KLE update summary:")
    for path in sorted(kle_in.glob("*.json")):
        layer_name = layer_name_from_filename(path.stem, old_layer_names)
        if layer_name not in old_name_to_idx:
            print(f"- {path.name}: skipped (unknown layer)")
            continue

        old_layer_idx = old_name_to_idx[layer_name]
        new_layer_idx = new_name_to_idx.get(layer_name)
        if new_layer_idx is None:
            print(f"- {path.name}: skipped (layer missing in new keymap)")
            continue
        if old_layer_idx >= len(old_layers) or new_layer_idx >= len(new_layers):
            print(f"- {path.name}: skipped (layer index out of range)")
            continue

        layout = parse_kle(path)
        stats, warnings = update_layout(
            layout,
            old_layer_idx,
            new_layer_idx,
            old_layers,
            new_layers,
            sig_to_content,
            kp_label_map,
        )

        output_data = serialize_kle(layout)
        out_path = kle_out / path.name
        out_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

        warn_text = "" if not warnings else f" warnings={len(warnings)}"
        print(
            f"- {path.name}: updated={stats['updated']} moved={stats['moved']} "
            f"reused={stats['reused']} from_global={stats['from_global']} "
            f"generated={stats['generated']} missing_in_kle={stats['missing_in_kle']}{warn_text}"
        )
        for warning in warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
