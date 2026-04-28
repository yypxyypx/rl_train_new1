# Benchmark 评测系统

## 整体流程

```
推理（infer.sh）→ 中间产物（DA3 深度估计 / VBench worker）→ 指标计算 → 汇总 summary.json
```

**一键脚本 `run_eval.sh`** 将上述三步串联，是日常使用的主入口。

---

## 代码结构

```
eval/
├── benchmark/
│   ├── run_eval.sh           # ★ 一键入口（推理 + 评测）
│   ├── run_benchmark.sh      # 仅评测入口（bash 包装器）
│   ├── run_benchmark.py      # 评测主调度器
│   ├── common/
│   │   ├── scan.py           # 扫描 output_root，枚举 gen_0.mp4
│   │   ├── intermediate.py   # 中间产物调度（DA3/VBench worker 等）
│   │   └── utils.py          # IO、conda 运行器等工具函数
│   ├── video_quality/
│   │   ├── psnr_ssim_lpips.py
│   │   └── vbench_eval.py
│   ├── reward/
│   │   ├── camera_pose.py    # 旋转/平移 AUC + 平移距离
│   │   └── depth_reprojection.py
│   └── reconstruction/
│       └── global_point_cloud.py  # Chamfer Distance / F-score
└── infer/
    ├── infer.sh              # 推理统一入口
    ├── gen3r/
    │   ├── infer_gen3r.py
    │   └── build_manifest.py
    └── wan2.2/
        ├── infer_wan22.py
        └── build_manifest.py

third_party/
├── reward_code/step_da3.py   # DA3 深度估计脚本
├── vbench/vbench_metrics.py  # VBench 三项指标
└── workers/worker_vbench.py  # VBench conda worker

model/
└── vbench_cache/
    ├── dreamsim_cache/       # DreamSim 权重（i2v_background 用）
    └── pyiqa_model/          # MUSIQ 权重（imaging_quality 用）
```

---

## 快速上手（最常用场景）

### 测试 Gen3R 模型

```bash
bash /mnt/afs/visitor16/rl_train_new/eval/benchmark/run_eval.sh \
    --model       gen3r \
    --checkpoint  /path/to/gen3r_checkpoints \
    --ckpt_tag    gen3r_baseline \
    --data        /mnt/afs/visitor16/rl_train_new/hf_datasets/rl_data/gen3r \
    --datasets    dl3dv,scannet++ \
    --output_root /mnt/afs/visitor16/rl_train_new/eval/results \
    --num_gpus    1
```

### 测试 Wan2.2 模型

```bash
bash /mnt/afs/visitor16/rl_train_new/eval/benchmark/run_eval.sh \
    --model       wan2.2 \
    --checkpoint  /path/to/Wan2.2-Fun-5B-Control-Camera \
    --ckpt_tag    wan_baseline \
    --data        /mnt/afs/visitor16/rl_train_new/hf_datasets/rl_data/wan2.2 \
    --datasets    scannet++ \
    --output_root /mnt/afs/visitor16/rl_train_new/eval/results \
    --num_gpus    1
```

### run_eval.sh 主要参数

| 参数 | 说明 |
|---|---|
| `--model` | `gen3r` 或 `wan2.2` |
| `--checkpoint` | 权重目录路径 |
| `--ckpt_tag` | 实验标签，结果存放在 `output_root/<ckpt_tag>/` 下，用于区分不同 ckpt |
| `--data` | 数据集根目录（含各数据集子目录） |
| `--datasets` | 逗号分隔，如 `dl3dv,scannet++` |
| `--output_root` | 结果根目录 |
| `--num_gpus` | 推理并行 GPU 数 |
| `--max_per_dataset N` | 每个数据集限制 N 条，用于快速冒烟验证 |
| `--skip_infer` | 跳过推理，只跑评测（视频已存在时用） |
| `--skip_bench` | 跳过评测，只跑推理 |

脚本结束后自动打印关键指标摘要，完整结果在 `output_root/<ckpt_tag>/benchmark_results/summary.json`。

---

## 数据格式要求

推理输出（或手动准备）的每个样本目录结构：

```
<output_root>/<ckpt_tag>/<dataset>/<sample_id>/
  gen_0.mp4       # 模型生成的视频（必须）
  gt.mp4          # GT 视频（必须，用于 PSNR/VBench/点云）
  start.png       # 首帧参考图（VBench i2v 指标用）
  camera.txt      # 相机参数（必须，格式见下）
  metadata.json   # 样本元信息（可选）
```

**`camera.txt` 格式**（每行一帧）：
```
idx fx_n fy_n cx_n cy_n 0 0 r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
```
- 内参 fx/fy/cx/cy 已按图像宽高归一化
- 外参为 OpenCV 坐标系下的 w2c 矩阵（旋转矩阵 + 平移向量）

---

## 评测指标说明

固定测试的 4 类指标：

| 类别 | 指标 | 关键字段 |
|---|---|---|
| 视频质量 | PSNR / SSIM / LPIPS | `psnr`, `ssim`, `lpips` |
| VBench | i2v 一致性 + 画质 | `i2v_subject`, `i2v_background`, `imaging_quality` |
| 相机轨迹 | AUC + 平移距离 | `rotation_auc30`, `translation_auc30`, `pose_auc30`, `translation_metric` |
| 全局点云 | Umeyama 对齐后的 3D 重建 | `chamfer_distance`, `fscore` |

### camera_pose 字段说明

| 字段 | 含义 |
|---|---|
| `rotation_auc30` | 旋转误差 AUC@30° |
| `translation_auc30` | 平移方向误差 AUC@30° |
| `pose_auc30` | 组合 AUC@30°（取旋转/平移方向误差中较大值，标准 Pose AUC） |
| `translation_metric` | 首帧+尺度对齐后的归一化平移距离 |

---

## 输出结构

```
<output_root>/<ckpt_tag>/
  <dataset>/<sample_id>/
    gen_0.mp4
    gt.mp4
    gen_0/
      eval/
        psnr_ssim_lpips.json
        camera_pose.json
        global_point_cloud.json
        vbench.json
      intermediates/
        da3_pred.npz       # DA3 深度估计结果
        da3_gt.npz         # GT 视频的 DA3 结果
        frames/            # 解帧后的图片（临时）
        vbench.json        # VBench worker 原始输出
  benchmark_results/
    results.jsonl          # 每条样本的详细数值
    summary.json           # 所有指标均值/中位数/标准差（主要看这个）
```

---

## 重要注意事项

### 1. 运行环境

`run_benchmark.py` 必须用 **`rl_da3`** 环境的 Python 运行（`run_benchmark.sh` 已固定），因为它包含所有评测依赖（skimage、lpips、cv2 等）。DA3 深度估计子进程也在 `rl_da3` 中运行。VBench worker 在 `vbench` 环境中运行（由 intermediate.py 自动 conda run 调用）。

### 2. DA3 中间产物与推理不能同时占用 GPU

DA3 会生成 `.npz` 压缩文件（约 60-80MB/条）。如果 DA3 与推理进程同时在同一 GPU 上运行，显存压力会导致 zlib 压缩写入损坏（CRC 错误）。`run_eval.sh` 已做串行保证（推理完成 → 评测），不要手动并行。

### 3. VBench 权重缓存路径

VBench 三项指标各自依赖本地权重：

| 指标 | 模型 | 权重路径 |
|---|---|---|
| `i2v_subject` | DINO ViT-B/16 | `model/vbench_cache/dreamsim_cache/checkpoints/dino_vitbase16_pretrain.pth` |
| `i2v_background` | DreamSim ensemble | `model/vbench_cache/dreamsim_cache/ensemble_lora/` |
| `imaging_quality` | MUSIQ | `model/vbench_cache/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth` |

三项权重均已缓存在本地，无需下载。

### 4. `--ckpt_tag` 是区分不同实验的关键

同一个 `output_root` 下，不同 `ckpt_tag` 的结果互不干扰：
```
output_root/
  gen3r_baseline/      # ckpt_tag=gen3r_baseline
  gen3r_rl_ep10/       # ckpt_tag=gen3r_rl_ep10
  wan_baseline/        # ckpt_tag=wan_baseline
```

### 5. 中间产物可复用

DA3 和 VBench 中间产物会缓存在 `intermediates/` 目录，下次运行会自动跳过已完成的文件。如需强制重算：手动删除对应 `.npz` / `.json` 文件后重跑。

---

## 仅跑评测（不推理）

如果视频已生成（`gen_0.mp4` 等已存在），可以跳过推理：

```bash
bash /mnt/afs/visitor16/rl_train_new/eval/benchmark/run_benchmark.sh \
    --output_root /mnt/afs/visitor16/rl_train_new/eval/results/gen3r_baseline \
    --metrics video_quality.psnr,reward.camera_pose,reconstruction.global \
    --align umeyama \
    --gpu 0
```

或通过 `run_eval.sh --skip_infer`：

```bash
bash run_eval.sh --skip_infer \
    --ckpt_tag gen3r_baseline \
    --output_root /mnt/afs/visitor16/rl_train_new/eval/results
```

---

## 可选指标（按需）

默认固定跑 4 类核心指标。如需额外指标，修改 `run_eval.sh` 中的 `FIXED_METRICS` 变量，或直接调用 `run_benchmark.sh --metrics <key>`：

| key | 说明 | 额外依赖 |
|---|---|---|
| `reward.depth_reprojection.global` | 全局深度重投影误差 | DA3（已有） |
| `reward.feature_sim` | DINOv2 特征相似度 | DA3 + DINOv2（需 rl_da3 环境） |
| `reconstruction.object` | 物体级点云 | DA3 + SAM3（SAM3 环境） |
