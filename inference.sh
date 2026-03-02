#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=1
stop_stage=1

data_dir=/home/zql/code/cosyvoice-qinche-sft/data/tts-data/qinche_1
pretrained_model_dir=/home/zql/code/cosyvoice-qinche-sft/data/tts-models/CosyVoice2-0.5B
output_model_dir=/home/zql/code/cosyvoice-qinche-sft/data/tts-models/CosyVoice2-0.5B-qinche


#######################################
# 推理
#######################################
EXAMPLE_ID='4801'
task_type=cross-lingual
spk_id=qinche

### zero-shot inference
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Run zero-shot inference"
  python my_inference/zero-shot.py \
    --tts_text `pwd`/examples/my_tts_text.json \
    --model_dir $output_model_dir/llm_flow \
    --spk_id $spk_id \
    --test_data_dir $data_dir/test \
    --example_id $EXAMPLE_ID \
    --target_sr 16000 \
    --result_dir `pwd`/output/${spk_id}_inference \
    --task_type $task_type
fi

# sft inference
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Run inference after sft"
  for model in llm; do
    python my_inference/sft_inference.py \
        --spk_id $spk_id \
        --model_dir $output_model_dir/$model \
        --model $model \
        --emb_path `pwd`/data/train/spk2embedding.pt \
        --tts_text_path `pwd`/examples/my_tts_text.json \
        --output_wav_path `pwd`/output/${spk_id}_inference 
  done
fi
