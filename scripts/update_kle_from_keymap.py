#!/usr/bin/env python3
"""Update KLE layout exports by diffing two ZMK .keymap files.

This script mirrors update_kle_layouts.py but uses ZMK devicetree keymap
files (glove80.keymap) instead of the JSON export. It preserves geometry,
and moves full key content (label + style) when bindings move.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from update_kle_layouts import (
    KLELayout,
    KeyContent,
    build_move_map,
    layer_name_from_filename,
    map_indices,
    parse_kle,
    serialize_kle,
)


@dataclass
class KeymapLayer:
    name: str
    bindings: List[List[str]]


COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
COMMENT_LINE_RE = re.compile(r"//.*?$", re.MULTILINE)


def strip_comments(text: str) -> str:
    text = COMMENT_BLOCK_RE.sub("", text)
    text = COMMENT_LINE_RE.sub("", text)
    return text


def extract_brace_block(text: str, start_brace: int) -> Tuple[str, int]:
    depth = 0
    i = start_brace
    in_string = False
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
            i += 1
            continue
        if in_string:
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_brace + 1 : i], i
        i += 1
    raise ValueError("Unmatched brace while parsing keymap block.")


def extract_keymap_block(text: str) -> str:
    match = re.search(r"\bkeymap\b\s*\{", text)
    if not match:
        raise ValueError("Failed to find keymap { ... } block.")
    start_brace = match.end() - 1
    block, _ = extract_brace_block(text, start_brace)
    return block


def extract_bindings(block: str) -> List[List[str]]:
    match = re.search(r"\bbindings\b\s*=\s*<", block)
    if not match:
        return []
    start = match.end()
    end = block.find(">;", start)
    if end == -1:
        end = block.find(">", start)
    if end == -1:
        raise ValueError("Failed to locate end of bindings < ... >;")
    body = block[start:end].strip()
    if not body:
        return []
    tokens = [tok for tok in re.split(r"\s+", body) if tok]

    bindings: List[List[str]] = []
    current: List[str] = []
    for tok in tokens:
        if tok.startswith("&"):
            if current:
                bindings.append(current)
            current = [tok]
        else:
            if not current:
                current = [tok]
            else:
                current.append(tok)
    if current:
        bindings.append(current)

    return bindings


def parse_keymap(path: Path) -> Tuple[List[str], Dict[str, List[List[str]]]]:
    raw = path.read_text(encoding="utf-8")
    text = strip_comments(raw)
    keymap_block = extract_keymap_block(text)

    layers: List[KeymapLayer] = []
    idx = 0
    while True:
        match = re.search(r"\blayer_([A-Za-z0-9_]+)\b\s*\{", keymap_block[idx:])
        if not match:
            break
        layer_name = match.group(1)
        start_brace = idx + match.end() - 1
        block, end = extract_brace_block(keymap_block, start_brace)
        idx = end + 1
        bindings = extract_bindings(block)
        layers.append(KeymapLayer(name=layer_name, bindings=bindings))

    layer_names = [layer.name for layer in layers]
    layer_map = {layer.name: layer.bindings for layer in layers}
    return layer_names, layer_map


def binding_signature(tokens: List[str]) -> str:
    return " ".join(tokens)


def clean_layer_name(token: str) -> str:
    if token.startswith("LAYER_"):
        return token[len("LAYER_") :]
    return token


def build_kp_label_map(sig_to_content: Dict[str, KeyContent], sig_to_tokens: Dict[str, List[str]]) -> Dict[str, str]:
    kp_map: Dict[str, str] = {}
    for sig, tokens in sig_to_tokens.items():
        if not tokens or tokens[0] != "&kp":
            continue
        if len(tokens) < 2:
            continue
        keycode = tokens[1]
        if keycode not in kp_map:
            kp_map[keycode] = sig_to_content[sig].label
    return kp_map


def keycode_to_label(keycode: str, kp_label_map: Dict[str, str]) -> str:
    keycode = keycode.strip()
    if keycode in kp_label_map:
        return kp_label_map[keycode]
    fallback = {
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
    return fallback.get(keycode, keycode)


def generate_label(tokens: List[str], kp_label_map: Dict[str, str]) -> str:
    if not tokens:
        return ""
    head = tokens[0]
    if head in {"&none", "&trans"}:
        return ""
    if head == "&kp" and len(tokens) >= 2:
        return keycode_to_label(tokens[1], kp_label_map)
    if head == "&mo" and len(tokens) >= 2:
        layer = clean_layer_name(tokens[1])
        return f"\n\n\n\n{layer}"
    if head == "&tog" and len(tokens) >= 2:
        layer = clean_layer_name(tokens[1])
        return f"Toggle\n\n\n\n{layer}"
    if head == "&sk" and len(tokens) >= 2:
        mod = tokens[1]
        return f"sticky\n\n\n\n{mod}\n\n{mod}"
    if head in {"&thumb", "&space"} and len(tokens) >= 3:
        layer = clean_layer_name(tokens[1])
        key = keycode_to_label(tokens[2].strip("(),"), kp_label_map)
        return f"{key}\n\n\n\n{layer}"
    return " ".join(tokens)


def update_layout(
    layout,
    old_layer: List[List[str]],
    new_layer: List[List[str]],
    sig_to_content: Dict[str, KeyContent],
    kp_label_map: Dict[str, str],
) -> Tuple[Dict[str, int], List[str]]:
    index_to_key, warnings = map_indices(layout)

    old_sigs = [binding_signature(tokens) for tokens in old_layer]
    new_sigs = [binding_signature(tokens) for tokens in new_layer]

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


def swap_bottom_text_blocks(layout: KLELayout) -> bool:
    """Swap the two large text-block decals below the keyboard.

    Returns True if a swap was performed.
    """
    for row in layout.rows:
        decals = [
            key for key in row
            if key.style.get("d") and key.w >= 4 and key.h >= 3
        ]
        if (
            len(decals) == 2
            and all("Combo T1+T2+T3" in k.label for k in decals)
        ):
            decals[0].label, decals[1].label = decals[1].label, decals[0].label
            for d in decals:
                d.label = d.label.replace(
                    'combos left"', 'combos __tmp__"'
                ).replace(
                    'combos right"', 'combos left"'
                ).replace(
                    'combos __tmp__"', 'combos right"'
                ).replace(
                    'Typing layer', '__TMP_LAYER__'
                ).replace(
                    'Gaming layer', 'Typing layer'
                ).replace(
                    '__TMP_LAYER__', 'Gaming layer'
                ).replace(
                    'RShift (tap: sticky one-shot)', '__TMP_SHIFT__'
                ).replace(
                    'Shift (tap: sticky one-shot)', 'RShift (tap: sticky one-shot)'
                ).replace(
                    '__TMP_SHIFT__', 'Shift (tap: sticky one-shot)'
                )
                d.label = re.sub(
                    r"<li>([^<]*?) = ([^<]*?)</li>",
                    r"<li>\2 = \1</li>",
                    d.label,
                )
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update KLE exports by diffing ZMK .keymap files.")
    parser.add_argument(
        "--old-keymap",
        dest="old_keymap",
        default="sunaku/Glorious Engrammer v42-rc9 (unmodified-reference).keymap",
        help="Path to the reference keymap .keymap file (unmodified).",
    )
    parser.add_argument(
        "--new-keymap",
        dest="new_keymap",
        default="config/glove80.keymap",
        help="Path to the updated keymap .keymap file (working copy).",
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
        default="sunaku/kle-layouts-unmodified-reference",
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

    old_layer_names, old_layers = parse_keymap(old_path)
    new_layer_names, new_layers = parse_keymap(new_path)

    if old_layer_names != new_layer_names:
        print("Warning: layer names differ between old and new keymaps.")

    kle_out.mkdir(parents=True, exist_ok=True)

    # Build global signature -> content map from old KLE files.
    sig_to_content: Dict[str, KeyContent] = {}
    sig_to_tokens: Dict[str, List[str]] = {}

    for path in sorted(kle_in.glob("*.json")):
        layer_name = layer_name_from_filename(path.stem, old_layer_names)
        if layer_name not in old_layers:
            print(f"Skipping {path.name}: cannot map to layer name", file=sys.stderr)
            continue
        layer_bindings = old_layers[layer_name]

        layout = parse_kle(path)
        index_to_key, _ = map_indices(layout)

        for idx, key in index_to_key.items():
            if idx >= len(layer_bindings):
                continue
            tokens = layer_bindings[idx]
            sig = binding_signature(tokens)
            if sig not in sig_to_content:
                sig_to_content[sig] = KeyContent(label=key.label, style=dict(key.style))
                sig_to_tokens[sig] = tokens

    kp_label_map = build_kp_label_map(sig_to_content, sig_to_tokens)

    print("KLE update summary:")
    for path in sorted(kle_in.glob("*.json")):
        layer_name = layer_name_from_filename(path.stem, old_layer_names)
        if layer_name not in old_layers:
            print(f"- {path.name}: skipped (unknown layer)")
            continue
        if layer_name not in new_layers:
            print(f"- {path.name}: skipped (layer missing in new keymap)")
            continue

        old_layer = old_layers[layer_name]
        new_layer = new_layers[layer_name]

        layout = parse_kle(path)
        stats, warnings = update_layout(
            layout,
            old_layer,
            new_layer,
            sig_to_content,
            kp_label_map,
        )

        if swap_bottom_text_blocks(layout):
            stats["text_swapped"] = 1

        output_data = serialize_kle(layout)
        out_path = kle_out / path.name
        out_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

        swap_text = " text_swapped" if stats.get("text_swapped") else ""
        warn_text = "" if not warnings else f" warnings={len(warnings)}"
        print(
            f"- {path.name}: updated={stats['updated']} moved={stats['moved']} "
            f"reused={stats['reused']} from_global={stats['from_global']} "
            f"generated={stats['generated']} missing_in_kle={stats['missing_in_kle']}"
            f"{swap_text}{warn_text}"
        )
        for warning in warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
