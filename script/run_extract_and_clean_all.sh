#!/usr/bin/env bash
# 对 qinche_data 下除 1_1、1_2 外的每个子目录执行 run_extract_and_clean.sh
#
# 用法:
#   bash run_extract_and_clean_all.sh
#   bash run_extract_and_clean_all.sh /path/to/qinche_data   # 指定 qinche_data 目录

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# qinche_data 路径已固定为 /home/zql/code/cosyvoice-paimon-sft-main/data/qinche_data，忽略参数
QINCHE_DATA="/home/zql/code/cosyvoice-paimon-sft-main/data/qinche_data"
QINCHE_DATA="$(cd "$QINCHE_DATA" 2>/dev/null && pwd)" || { echo "目录不存在: $QINCHE_DATA"; exit 1; }

SKIP_DIRS=("1_1" "1_2")
RUN_SCRIPT="$SCRIPT_DIR/run_extract_and_clean.sh"

if [[ ! -x "$RUN_SCRIPT" ]]; then
  echo "未找到或不可执行: $RUN_SCRIPT"
  exit 1
fi

shopt -s nullglob
SUBDIRS=("$QINCHE_DATA"/*/)
TOTAL=0
RUN=0

for dir in "${SUBDIRS[@]}"; do
  name=$(basename "$dir")
  if [[ "$name" == "1_1" || "$name" == "1_2" ]]; then
    echo "跳过: $name"
    ((TOTAL++)) || true
    continue
  fi
  [[ -d "$dir" ]] || continue
  ((TOTAL++)) || true
  ((RUN++)) || true
  echo ""
  echo ">>>> 执行 [$RUN] $dir"
  bash "$RUN_SCRIPT" "$dir" || { echo "失败: $dir"; exit 1; }
done

echo ""
echo "全部完成：共处理 $RUN 个子目录（跳过 1_1、1_2，共 $TOTAL 个子目录）。"
