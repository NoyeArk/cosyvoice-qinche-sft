set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 本脚本在 /home/zql/code/cosyvoice-paimon-sft-main/script 下，项目根目录即为 SCRIPT_DIR 的上一级
COSYVOICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="${1:-$COSYVOICE_ROOT/qinche_data/1_1}"
TARGET_DIR="$(cd "$TARGET_DIR" 2>/dev/null && pwd)" || { echo "目录不存在: $TARGET_DIR"; exit 1; }

echo "=========================================="
echo "1. 提取文本（extract_text_from_audio.py）"
echo "=========================================="
echo "音频/输出目录: $TARGET_DIR"
echo ""
cd "$COSYVOICE_ROOT"
uv run python "$SCRIPT_DIR/extract_text_from_audio.py" \
  --audio_dir "$TARGET_DIR" \
  --overwrite

echo ""
echo "=========================================="
echo "2. 删除过短文本及对应音频（remove_short_txt_and_audio.py）"
echo "=========================================="
uv run python "$SCRIPT_DIR/remove_short_txt_and_audio.py" "$TARGET_DIR"

echo ""
echo "全部完成。结果目录: $TARGET_DIR"
