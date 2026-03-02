# -*- coding: utf-8 -*-
from pydub import AudioSegment
from pydub.silence import detect_silence
import os
import uuid

# 进度条依赖
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# 根据扩展名推断音频格式
def _format_from_path(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav",):
        return "wav"
    if ext in (".mp3",):
        return "mp3"
    if ext in (".flac",):
        return "flac"
    if ext in (".ogg", ".oga"):
        return "ogg"
    if ext in (".m4a", ".aac"):
        return "m4a"
    return None


def _load_audio(file_name, audio_type=None):
    """加载音频，优先用扩展名格式，失败时尝试不指定格式由 ffmpeg 自动检测"""
    fmt = audio_type or _format_from_path(file_name)
    try:
        if fmt:
            return AudioSegment.from_file(file_name, format=fmt)
        return AudioSegment.from_file(file_name)
    except Exception as e:
        if fmt:
            # 扩展名格式失败时，尝试不指定格式
            try:
                return AudioSegment.from_file(file_name)
            except Exception:
                pass
        raise e


# 生成guid
def GUID():
    return str(uuid.uuid1()).replace("-", "")


# 分割文件
def SplitSound(
    filename, save_path, save_file_name, start_time, end_time, audio_type=None
):
    if not os.path.exists(save_path):
        try:
            os.mkdir(save_path)
        except Exception as e:
            print(e)

    sound = _load_audio(filename, audio_type)
    result = sound[start_time:end_time]
    final_name = save_path
    if not save_path.endswith("/"):
        final_name = final_name + "/"
    final_name = final_name + save_file_name

    # 按输出文件名扩展名决定导出格式
    export_fmt = (
        _format_from_path(save_file_name)
        or audio_type
        or _format_from_path(filename)
        or "wav"
    )
    result.export(final_name, format=export_fmt)


def SplitSilence(file_name, save_path, audio_type=None, show_progress=False):
    sound = _load_audio(file_name, audio_type)
    # print(len(sound))
    # print(sound.max_possible_amplitude)
    # start_end = detect_silence(sound,800,-57,1)
    start_end = detect_silence(sound, 300, -35, 1)

    # print(start_end)
    start_point = 0
    index = 1

    segments = []
    for item in start_end:
        if item[0] != 0:
            # 取空白部分的中位数
            end_point = (item[0] + item[1]) / 2
            segments.append((start_point, end_point))
            start_point = item[1]
            index += 1
        else:
            # 跳过0开头的无效段，直接推进start_point
            start_point = item[1]

    # 最后一段
    segments.append((start_point, len(sound)))

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(
            total=len(segments),
            desc=f"  [{os.path.basename(file_name)}] 切分进度",
            unit="seg",
        )
    for idx, (st, en) in enumerate(segments):
        print("%d-%d" % (st, en))
        SplitSound(file_name, save_path, str(idx + 1) + ".mp3", st, en)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()


# 支持的音频扩展名
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")


def split_all(audio_dir, save_dir, audio_type=None):
    """
    对 audio_dir 中每个音频文件按静音切分，结果保存到 save_dir 下以文件名命名的子目录中。

    Args:
        audio_dir: 源音频目录，如 Sertua
        save_dir: 结果根目录，如 qinche_data；每个音频的切分结果保存在 save_dir/<基名>/ 下
        audio_type: 可选，强制指定音频格式，None 则按扩展名推断
    """
    audio_dir = os.path.abspath(audio_dir)
    save_dir = os.path.abspath(save_dir)
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"音频目录不存在: {audio_dir}")

    os.makedirs(save_dir, exist_ok=True)

    # 只处理支持的音频文件，并排序保证顺序稳定
    names = [
        f
        for f in os.listdir(audio_dir)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ]
    names.sort()

    use_progress = tqdm is not None and len(names) > 1
    if use_progress:
        outter_pbar = tqdm(total=len(names), desc="全局进度", unit="file")
    else:
        outter_pbar = None

    for i, name in enumerate(names):
        path = os.path.join(audio_dir, name)
        if not os.path.isfile(path):
            if outter_pbar:
                outter_pbar.update(1)
            continue
        base = os.path.splitext(name)[0]
        out_sub = os.path.join(save_dir, base)
        print(f"[{i + 1}/{len(names)}] 切分: {name} -> {out_sub}/")
        try:
            SplitSilence(
                path,
                out_sub,
                audio_type=audio_type,
                show_progress=True if tqdm else False,
            )
        except Exception as e:
            print(f"  失败: {e}")
        if outter_pbar:
            outter_pbar.update(1)

    if outter_pbar:
        outter_pbar.close()


if __name__ == "__main__":
    import sys

    # 默认：Sertua 下所有音频切分到 qinche_data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cosyvoice_root = os.path.abspath(os.path.join(script_dir, "..", "CosyVoice"))
    audio_dir = os.path.join(cosyvoice_root, "Sertua")
    save_dir = os.path.join(cosyvoice_root, "qinche_data")

    if len(sys.argv) >= 3:
        audio_dir = os.path.abspath(sys.argv[1])
        save_dir = os.path.abspath(sys.argv[2])
    elif len(sys.argv) == 2:
        audio_dir = os.path.abspath(sys.argv[1])
        save_dir = os.path.join(os.path.dirname(audio_dir), "qinche_data")

    print(f"音频目录: {audio_dir}")
    print(f"输出目录: {save_dir}")
    print()
    split_all(audio_dir, save_dir)
