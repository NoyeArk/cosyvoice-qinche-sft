#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除指定目录中「文本少于 4 个字符」的 .txt 及其同名音频（如 .mp3/.wav）。

用法:
  python remove_short_txt_and_audio.py <目录>
  python remove_short_txt_and_audio.py /home/zql/code/CosyVoice/qinche_data/1_1

  --dry-run  仅打印将要删除的文件，不实际删除
  --min-len N  最少保留字符数，默认 4
"""

import argparse
import os
from pathlib import Path

# 尝试删除时匹配的音频扩展名
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")


def main():
    parser = argparse.ArgumentParser(description="删除文本过短的 txt 及对应音频")
    parser.add_argument(
        "dir",
        type=str,
        nargs="?",
        default="/home/zql/code/CosyVoice/qinche_data/1_1",
        help="目标目录，默认 qinche_data/1_1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要删除的文件，不实际删除",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=4,
        help="文本至少保留的字符数，少于此数的 txt 及其音频会被删除（默认 4）",
    )
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.is_dir():
        print(f"错误：目录不存在 {root}")
        return 1

    to_remove_txt = []
    to_remove_audio = []

    for txt_path in sorted(root.glob("*.txt")):
        try:
            text = txt_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"跳过（无法读取）: {txt_path.name} - {e}")
            continue

        if len(text) < args.min_len:
            to_remove_txt.append(txt_path)
            stem = txt_path.stem
            for ext in AUDIO_EXTENSIONS:
                audio_path = root / (stem + ext)
                if audio_path.is_file():
                    to_remove_audio.append(audio_path)
                    break

    if not to_remove_txt and not to_remove_audio:
        print(f"未发现文本少于 {args.min_len} 个字符的 txt，无需删除。")
        return 0

    print(f"将删除以下文件（文本字符数 < {args.min_len}）：")
    for p in to_remove_txt:
        try:
            content = p.read_text(encoding="utf-8").strip()
            print(f"  txt:  {p.name}  ({len(content)} 字) {repr(content)[:50]}")
        except Exception:
            print(f"  txt:  {p.name}")
    for p in to_remove_audio:
        print(f"  音频: {p.name}")

    if args.dry_run:
        print("\n[--dry-run] 未实际删除。去掉 --dry-run 后执行将删除以上文件。")
        return 0

    print()
    for p in to_remove_txt + to_remove_audio:
        try:
            p.unlink()
            print(f"已删除: {p.name}")
        except Exception as e:
            print(f"删除失败 {p}: {e}")

    print(
        f"\n完成，共删除 {len(to_remove_txt)} 个 txt、{len(to_remove_audio)} 个音频。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
