# Inference Module

## 代码结构

```
eval/infer/
├── infer.sh                      # 统一 bash 接口
├── infer_use.md
└── gen3r/                        # Gen3R 推理
    ├── infer_gen3r.py            # 官方 pipeline SDE 推理
    └── build_manifest.py         # 扫描处理后数据 → manifest.jsonl
```

添加新模型时，在 `eval/infer/` 下新建文件夹（如 `wan22/`），放入 `infer_wan22.py` 和 `build_manifest.py`，`infer.sh` 通过 `--model` 参数自动分发。

## 输入

读取 `data/` 模块处理后的数据（start.png, camera.txt, metadata.json），或者一个 manifest.jsonl 文件。

## 输出

所有模型输出统一格式：

```
<output_root>/<dataset>/<sample_id>/
├── gen_0.mp4 ~ gen_{N-1}.mp4    # 生成的 rollout 视频
├── start.png                     # 从输入复制
├── gt.mp4                        # 从输入复制
├── camera.txt                    # 从输入复制
├── metadata.json                 # 从输入复制
└── infer_info.json               # 推理参数记录
```

## 使用方式

### 最简用法：指定数据目录 + checkpoint

```bash
bash infer.sh \
    --model gen3r \
    --checkpoint /path/to/gen3r_checkpoints \
    --data /path/to/processed_data \
    --datasets re10k,dl3dv \
    --output /path/to/output \
    --num_gpus 4 \
    --skip_done
```

自动执行：build_manifest → torchrun 多卡推理

### 用 RL 训练的 checkpoint 评估

```bash
bash infer.sh \
    --model gen3r \
    --checkpoint /path/to/grpo_outputs/checkpoint-200 \
    --data /path/to/processed_data \
    --datasets re10k,dl3dv \
    --output /path/to/eval_step200 \
    --num_gpus 4
```

### 单个样本调试

```bash
bash infer.sh \
    --model gen3r \
    --checkpoint /path/to/gen3r_checkpoints \
    --sample_dir /path/to/processed/re10k/scene_001 \
    --output /tmp/debug
```

### 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | gen3r | 模型名（对应 infer/ 下的子文件夹） |
| `--checkpoint` | (必需) | 模型 checkpoint 路径 |
| `--output` | (必需) | 输出根目录 |
| `--data` | | 处理后的数据根目录（自动 build manifest） |
| `--datasets` | re10k,dl3dv | 逗号分隔的数据集名（配合 --data） |
| `--manifest` | | 预构建的 manifest.jsonl 路径 |
| `--sample_dir` | | 单个样本目录（调试用） |
| `--num_gpus` | 1 | GPU 数量 |
| `--num_rollouts` | 8 | 每样本生成的视频数 |
| `--num_frames` | 49 | 生成帧数 |
| `--target_size` | 560 | 分辨率 |
| `--eta` | 0.3 | SDE 噪声强度 |
| `--steps` | 50 | 去噪步数 |
| `--guidance` | 5.0 | CFG scale |
| `--shift` | 2.0 | sigma schedule shift |
| `--seed` | 42 | 随机种子基值 |
| `--skip_done` | | 跳过已完成的样本 |
| `--device_mode` | server | server=全量GPU / local=CPU offload |
| `--require_gt` | | manifest 只含有 gt.mp4 的样本 |
