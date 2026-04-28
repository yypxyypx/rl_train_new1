# Inference Module

## 代码结构

```
eval/infer/
├── infer.sh                      # 统一 bash 接口
├── infer_use.md
├── gen3r/                        # Gen3R 推理
│   ├── infer_gen3r.py            # 官方 Gen3RPipeline ODE 推理
│   ├── build_manifest.py         # 扫描数据 → manifest.jsonl
│   └── Gen3R/                    # 官方 Gen3R 仓库（含 checkpoints 软链）
└── wan2.2/                       # Wan2.2 推理
    ├── infer_wan22.py            # 官方 Wan2_2FunControlPipeline ODE 推理
    ├── build_manifest.py         # 扫描数据 → manifest.jsonl
    └── VideoX-Fun/               # 官方 VideoX-Fun 框架（拷贝，原目录不动）
```

添加新模型时，在 `eval/infer/` 下新建文件夹，放入 `infer_<model>.py` 和 `build_manifest.py`，`infer.sh` 通过 `--model` 参数自动分发。

## 推理框架说明

- **Gen3R**：调用 `gen3r.pipeline.Gen3RPipeline.__call__`（ODE，官方）。
- **Wan2.2**：调用 `videox_fun.pipeline.Wan2_2FunControlPipeline.__call__`（Flow ODE，官方）。
- **不加载 T5**：两个 pipeline 都支持传入 `prompt_embeds` / `negative_prompt_embeds`，直接读取每条样本目录下的 `prompt_embed.pt` + `neg_embed.pt`，不实例化 text encoder / tokenizer。
- **无 rollout**：官方 pipeline 是确定性 ODE，每条样本只产出一个 `gen_0.mp4`。

## 输入数据格式

每条样本目录包含：

```
<sample_dir>/
  start.png           # 首帧图像
  camera.txt          # 相机参数（OpenCV w2c，详见下文）
  gt.mp4              # GT 视频
  metadata.json       # 图像尺寸、数据集名、sample_id 等
  prompt_embed.pt     # 预编码的 T5 正向 embedding [L, 4096]
  neg_embed.pt        # 预编码的 T5 负向 embedding [L, 4096]
```

`camera.txt` 每行格式：
```
idx  fx_n fy_n cx_n cy_n  0 0  r00 r01 r02 tx  r10 r11 r12 ty  r20 r21 r22 tz
```
fx_n/fy_n/cx_n/cy_n 为按原始图像尺寸归一化的值，坐标系为 OpenCV w2c。

## 输出结构

所有模型输出统一格式，并在 `output_root` 下加一层 `ckpt_tag`：

```
<output_root>/<ckpt_tag>/<dataset>/<sample_id>/
  gen_0.mp4           # 生成视频
  start.png           # 从输入复制
  gt.mp4              # 从输入复制
  camera.txt          # 从输入复制
  metadata.json       # 从输入复制
  infer_info.json     # 推理参数记录
```

## 使用方式

### 通过中心配置文件（推荐）

```bash
# 1. 编辑 config/infer.sh（设置 MODEL、CHECKPOINT、DATA 等）
# 2. 直接运行
bash eval/infer/infer.sh
```

### 命令行参数覆盖

```bash
# Gen3R 基础模型，两个数据集各取全部 200 条
bash eval/infer/infer.sh \
    --model gen3r \
    --checkpoint /mnt/afs/visitor16/RL_Pipeline/gen3r/Gen3R/checkpoints \
    --ckpt_tag gen3r_baseline \
    --data /mnt/afs/visitor16/rl_train_new/hf_datasets/rl_data/gen3r \
    --datasets dl3dv,scannet++ \
    --output /path/to/eval_output \
    --num_gpus 4 \
    --skip_done

# Gen3R RL 训练后的 checkpoint
bash eval/infer/infer.sh \
    --model gen3r \
    --checkpoint /path/to/grpo_outputs/checkpoint-200 \
    --ckpt_tag gen3r_step200 \
    --data /mnt/afs/visitor16/rl_train_new/hf_datasets/rl_data/gen3r \
    --datasets dl3dv,scannet++ \
    --output /path/to/eval_output \
    --num_gpus 4

# Wan2.2 推理
bash eval/infer/infer.sh \
    --model wan2.2 \
    --checkpoint /mnt/afs/visitor16/rl_train_new/model/Wan2.2-Fun-5B-Control-Camera \
    --ckpt_tag wan22_baseline \
    --data /mnt/afs/visitor16/rl_train_new/hf_datasets/rl_data/wan2.2 \
    --datasets scannet++ \
    --output /path/to/eval_output \
    --num_gpus 1

# 单样本调试
bash eval/infer/infer.sh \
    --model gen3r \
    --checkpoint /path/to/ckpt \
    --sample_dir /path/to/sample \
    --output /tmp/debug

# 小规模测试（每数据集 1 条）
bash eval/infer/infer.sh \
    --model gen3r \
    --checkpoint /path/to/ckpt \
    --data /path/to/data \
    --datasets dl3dv,scannet++ \
    --output /tmp/smoke \
    --max_per_dataset 1
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | gen3r | 模型名（gen3r \| wan2.2） |
| `--checkpoint` | (必填) | 模型 checkpoint 路径 |
| `--ckpt_tag` | \${MODEL} | 输出目录中间层标识（留空=用模型名） |
| `--output` | (必填) | 输出根目录，实际写入 \${OUTPUT}/\${CKPT_TAG}/ |
| `--data` | | 数据根目录（自动 build_manifest） |
| `--datasets` | | 逗号分隔的数据集名（配合 --data） |
| `--manifest` | | 预构建的 manifest.jsonl（与 --data 互斥） |
| `--sample_dir` | | 单条样本目录（调试用） |
| `--num_gpus` | 1 | GPU 数量（>1 时用 torchrun 多卡） |
| `--max_per_dataset` | 0 | 每数据集最多取多少条（0=全部） |
| `--seed` | 42 | 随机种子 |
| `--skip_done` | | 跳过已完成样本 |
| `--master_port` | 29500 | torchrun master port |

## 各模型默认推理参数

| 参数 | Gen3R | Wan2.2 |
|------|-------|--------|
| 分辨率 | 560×560 | 1280×704 |
| 帧数 | 49 | 49（≤81） |
| 去噪步数 | 50 | 50 |
| Guidance | 5.0 | 6.0 |
| Shift | 5.0 | 5.0 |
| Sampler | FlowMatch ODE | FlowMatch ODE |
| Rollout | 1（无随机） | 1（无随机） |
