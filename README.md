# CosyVoice2 微调：恋与深空秦彻语音

基于 CosyVoice2 对恋与深空中的秦彻角色语音进行 SFT（监督微调），得到可复现该音色的 TTS 模型。


## 1. 环境配置

- Python >= 3.10
- CUDA（训练与推理需 GPU）
- 依赖见 `pyproject.toml`，推荐使用虚拟环境安装：

```bash
cd /path/to/cosyvoice-qinche-sft
pip install -e .
# 或 uv / pip 根据 pyproject.toml 安装
```

运行前需执行 `source path.sh`（或由 `run.sh` 自动 source），以设置 `PYTHONPATH`（含本项目及 `third_party/Matcha-TTS`）。

## 2. 目录结构概览

```
cosyvoice-qinche-sft/
├── run.sh                 # 主流程脚本（stage 0～7）
├── path.sh                # 环境与 PYTHONPATH
├── pyproject.toml         # 依赖与项目信息
├── conf/
│   ├── cosyvoice2-qinche.yaml   # 秦彻微调配置（模型结构、数据 pipeline、训练超参）
│   ├── cosyvoice2.yaml          # 基础配置
│   └── ds_stage2.json           # DeepSpeed 配置（可选）
├── local/
│   └── prepare_data.py    # Stage 0：成对 wav/txt → Kaldi 风格表文件
├── tools/
│   ├── extract_embedding.py     # Stage 1：CAM++ 说话人嵌入
│   ├── extract_speech_token.py  # Stage 2：语音离散 token
│   ├── make_parquet_list.py     # Stage 3：生成 parquet 与 data.list
│   └── save_spk2info.py         # Stage 4：写入 spk2info.pt 到模型目录
├── cosyvoice/             # CosyVoice 核心代码（LLM / Flow / 数据处理等）
├── my_inference/           # 推理脚本（sft_inference、zero-shot 等）
├── examples/
│   └── my_tts_text.json   # 推理用文本（需包含与 spk_id 同名的 key，如 qinche）
├── data/                  # 数据与中间结果（见下方「数据与配置」）
└── exp/                   # 训练 checkpoint（由 run.sh 写入）
```

## 3. 数据与配置

### 原始数据格式

- **根目录**由 `run.sh` 中 `data_dir` 指定（示例：`data/tts-data`）。
- 其下需有 **train** 与 **test** 两个子集，每个子集内为「成对」的：
  - `*.wav`：音频（建议 24kHz，单声道；过长会被脚本或模型截断）
  - `*.normalized.txt`：与 wav 同名的文本，扩展名为 `.normalized.txt`  
  例如：`1_1000.wav` 对应 `1_1000.normalized.txt`。
- **说话人 ID**：由 `local/prepare_data.py` 从文件名推断，为**第一个下划线前的部分**（如 `1_1000` → 说话人 `1`）。若希望秦彻单独一个 ID，可统一用同一前缀（如 `qinche_xxx.wav`）。

### 推理用文本（examples/my_tts_text.json）

推理脚本会按 `spk_id` 取 JSON 中对应 key 的文本列表。当前示例为 `paimon`，做秦彻推理时需增加 `qinche` 键，例如：

```json
{
  "paimon": { ... },
  "qinche": {
    "sft_inference": [
      "你要的秦彻试听句子一。",
      "你要的秦彻试听句子二。"
    ]
  }
}
```

---

## run.sh：Stage 0～7 说明

| Stage | 说明 |
|-------|------|
| **0** | **数据准备**：将 `data_dir/{train,test}` 下成对 `.wav` 与 `.normalized.txt` 整理成 Kaldi 风格表文件（wav.scp、text、utt2spk、spk2utt），输出到 `data/train`、`data/test`。 |
| **1** | **说话人嵌入**：用 CAM++ ONNX 提取每条/每个说话人 embedding，得到 `utt2embedding.pt`、`spk2embedding.pt`。 |
| **2** | **语音离散 token**：用 speech_tokenizer_v2.onnx 生成 `utt2speech_token.pt`。 |
| **3** | **Parquet 与 data.list**：将上述特征打成 parquet，并生成 `data.list`，供训练读取。 |
| **4** | **说话人信息写入模型目录**：把当前 `spk_id` 的 embedding 写成 `spk2info.pt`，写入 `output_model_dir` 下 llm、flow、llm_flow；若目录为空会先复制预训练权重。 |
| **5** | **微调**：按 `conf/cosyvoice2-qinche.yaml` 训练 LLM（及可选 flow），使用 `data/train.data.list`、`data/dev.data.list`，推理试听写入 `output/${spk_id}_inference`。 |
| **6** | **模型平均**：对 llm / flow 的 checkpoint 做平均，得到最终 `llm.pt` / `flow.pt`。 |
| **7** | **导出**：将 `output_model_dir` 下模型导出为 JIT / ONNX，便于推理加速。 |

可通过 `stage` 与 `stop_stage` 控制范围，例如只跑数据预处理：

```bash
stage=0 stop_stage=4 bash run.sh
```

只跑微调（需先完成 0～4 并生成 parquet）：

```bash
stage=5 stop_stage=5 bash run.sh
```

---

## 快速开始（推荐流程）

1. **准备数据**：将秦彻语音按「一句一文件」切好，每个 wav 配一个同名的 `.normalized.txt`，分别放入 `$data_dir/train` 和 `$data_dir/test`。
2. **修改 run.sh**：设置 `data_dir`、`pretrained_model_dir`、`output_model_dir`、`spk_id=qinche`，以及 `stage`/`stop_stage`（如先 `0～4` 再 `5～7`）。
3. **执行预处理**：`stage=0 stop_stage=4 bash run.sh`。
4. **训练**：确认 `data/train.data.list`、`data/dev.data.list` 已生成（run.sh 中由 parquet 的 data.list 拼接），然后 `stage=5 stop_stage=5 bash run.sh`。
5. **平均与导出**：`stage=6 stop_stage=7 bash run.sh`（或按需只跑 6 或只跑 7）。
6. **推理**：使用 `my_inference/` 下脚本，或 CosyVoice 官方推理接口，指定 `output_model_dir` 与 `spk_id`；`examples/my_tts_text.json` 中需包含 key `qinche` 及对应试听文本。

## 微调结果

运行 `stage=5 stop_stage=5 bash run.sh` 后，会在 `exp/cosyvoice2/llm/torch_ddp` 下生成 checkpoint，并在 `tensorboard/cosyvoice2/llm/torch_ddp` 下生成 tensorboard 日志。

![tensorboard](img/train_loss.png)

语音复刻效果：

​<audio id="audio" controls="" preload="none">
      <source id="mp3" src="./output/qinche_inference/llm/qinche_sft_inference_0.wav">
</audio>