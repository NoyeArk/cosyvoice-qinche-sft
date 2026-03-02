#!/usr/bin/env python3
"""
从音频文件中提取语音文本，每个音频单独保存为同名的 .txt 文件。

支持两种后端：
- funasr: 使用 SenseVoiceSmall，中文识别准确且带标点（推荐）
- whisper: 使用 Whisper，可选标点恢复

Args:
    audio_dir: 音频文件目录路径
    txt_dir: 输出 txt 文件目录，默认与音频同目录
    backend: 识别引擎，funasr（推荐）或 whisper
    model_size: Whisper 模型大小（仅 whisper 后端有效）
"""

import argparse
from pathlib import Path


def _transcribe_funasr(audio_path: Path, model) -> str:
    """使用 FunASR SenseVoice 转录，自带标点。"""
    res = model.generate(
        input=str(audio_path),
        cache={},
        batch_size_s=60,
        language="zh",
        use_itn=True,
        merge_vad=True,
        merge_length_s=15,
    )
    if res and len(res) > 0:
        item = res[0]
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        if text:
            try:
                from funasr.utils.postprocess_utils import (
                    rich_transcription_postprocess,
                )

                text = rich_transcription_postprocess(text)
            except Exception:
                pass
        return text.strip()
    return ""


def _transcribe_whisper(
    audio_path: Path, model, add_punctuation: bool, punc_model
) -> str:
    """使用 Whisper 转录，可选标点恢复。"""
    result = model.transcribe(str(audio_path), language="zh")
    text = result.get("text", "").strip()
    if add_punctuation and text and punc_model is not None:
        try:
            punc_res = punc_model.generate(input=text, cache={})
            if punc_res and len(punc_res) > 0:
                item = punc_res[0] if isinstance(punc_res[0], dict) else punc_res
                text = item.get("text", text) if isinstance(item, dict) else str(item)
        except Exception:
            pass
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="../audio",
        help="音频文件目录",
    )
    parser.add_argument(
        "--txt_dir",
        type=str,
        default=None,
        help="输出 txt 目录，默认与音频同目录",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="funasr",
        choices=["funasr", "whisper"],
        help="识别引擎：funasr（中文准确+标点）/ whisper",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小（仅 whisper 后端）",
    )
    parser.add_argument(
        "--no_punctuation",
        action="store_true",
        help="禁用标点恢复（仅 whisper 后端）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 txt 文件",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    audio_dir = Path(args.audio_dir).resolve()
    if not audio_dir.is_absolute():
        audio_dir = (script_dir / args.audio_dir).resolve()
    txt_dir = Path(args.txt_dir).resolve() if args.txt_dir else audio_dir
    if args.txt_dir and not Path(args.txt_dir).is_absolute():
        txt_dir = (script_dir / args.txt_dir).resolve()
    txt_dir.mkdir(parents=True, exist_ok=True)

    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    audio_files = sorted(
        f for f in audio_dir.iterdir() if f.is_file() and f.suffix.lower() in audio_exts
    )
    if not audio_files:
        print(f"未在 {audio_dir} 中找到音频文件")
        return

    punc_model = None
    if args.backend == "funasr":
        try:
            from funasr import AutoModel

            print("加载 FunASR SenseVoiceSmall（中文识别 + 标点）...")
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=device,
                disable_update=True,
            )

            def transcribe(path):
                return _transcribe_funasr(path, model)

        except ImportError:
            print("未安装 funasr，请运行: pip install funasr")
            print("回退到 whisper 后端...")
            args.backend = "whisper"

    if args.backend == "whisper":
        import torch
        import whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"加载 Whisper 模型: {args.model_size}")
        model = whisper.load_model(args.model_size, device=device)

        if not args.no_punctuation:
            try:
                from funasr import AutoModel

                print("加载标点恢复模型 ct-punc...")
                punc_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                punc_model = AutoModel(
                    model="damo/punc_ct-transformer_cn-en-common-vocab471067-large",
                    device=punc_device,
                    disable_update=True,
                )
            except (ImportError, AssertionError, Exception) as e:
                print(f"标点恢复不可用，将无标点输出: {e}")

        def transcribe(path):
            return _transcribe_whisper(
                path,
                model,
                add_punctuation=not args.no_punctuation,
                punc_model=punc_model,
            )

    total = len(audio_files)
    for i, audio_path in enumerate(audio_files):
        txt_path = txt_dir / (audio_path.stem + ".txt")
        if txt_path.exists() and not args.overwrite:
            print(f"[{i+1}/{total}] 跳过（已存在）: {audio_path.name}")
            continue

        print(f"[{i+1}/{total}] 转录: {audio_path.name}", flush=True)
        try:
            text = transcribe(audio_path)
            txt_path.write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"  错误: {e}", flush=True)
            txt_path.write_text(f"[转录失败: {e}]", encoding="utf-8")

    print(f"完成，txt 文件保存在: {txt_dir}", flush=True)


if __name__ == "__main__":
    main()
