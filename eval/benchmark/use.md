# Benchmark 评测系统

## 代码结构

```
benchmark/
├── run_benchmark.sh              # 一键入口（bash 包装器）
├── run_benchmark.py              # 主调度器：指标展开 → 中间产物 → 评测 → 汇总
├── common/
│   ├── scan.py                   # 扫描 output_root，自动检测 gen_0 ~ gen_{N-1}
│   ├── intermediate.py           # 中间产物调度（DA3/SAM3/DINOv2/VideoAlign/VBench）
│   └── utils.py                  # IO、conda 运行器、parse_camera_txt
├── video_quality/
│   ├── psnr_ssim_lpips.py        # PSNR / SSIM / LPIPS
│   ├── vbench_eval.py            # VBench（从预计算 JSON 读取）
│   └── run_video_quality.py      # 独立入口
├── reward/
│   ├── camera_pose.py            # 旋转 AUC（全帧对）+ 平移距离指标
│   ├── depth_reprojection.py     # 物体级 / 全局深度重投影（三接口）
│   ├── videoalign_eval.py        # VideoAlign 评分
│   ├── feature_sim.py            # DINOv2 特征相似度
│   └── run_reward.py             # 独立入口
└── reconstruction/
    ├── global_point_cloud.py     # 全局点云一致性（含首帧对齐模式）
    ├── object_point_cloud.py     # 物体级点云一致性（含首帧对齐模式）
    └── run_reconstruction.py     # 独立入口
```

## 核心设计思路

### 1. 三阶段流水线

```
指标选择 → 中间产物生成 → 评测计算 → 结果汇总
```

- **指标选择**：用户通过 `--metrics` 指定要跑的指标（支持多级粒度），`run_benchmark.py` 将其展开为原子指标列表。
- **中间产物**：`IntermediateManager` 根据指标列表反查依赖（如 `reward.camera_pose` 需要 `da3`），取并集后按固定顺序执行：`da3 → sam3 → dinov2 → videoalign → vbench`。已存在的文件自动跳过。
- **评测计算**：各模块从中间产物读数据，计算指标，写入 `gen_X/eval/*.json`。
- **结果汇总**：读取所有 eval JSON，输出 `benchmark_results/results.jsonl` + `summary.json`。

### 2. 相机坐标系处理

所有相机数据统一为 **w2c + OpenCV**，不做跨约定转换：

| 数据来源 | 存储格式 | 使用时处理 |
|---------|---------|-----------|
| DA3 `extrinsics` | **(N,3,4) w2c** OpenCV | `np.linalg.inv` 取逆得 c2w |
| `camera.txt` | w2c (3×4) 行优先 + 归一化内参 | `parse_camera_txt` 内部取逆，返回 c2w |

两者取逆后同为 OpenCV 约定的 c2w，所有模块可以直接比较，无需 `_GT_CONVENTION_TO_OPENCV` 转换。

### 3. 指标说明

| 类别 | 指标 | 关键改动 |
|------|------|---------|
| video_quality | PSNR / SSIM / LPIPS / VBench | 复用已有代码 |
| reward.camera_pose | 旋转 AUC + 平移距离 | 旋转从逐帧相邻误差改为**全帧对 AUC**；平移用首帧对齐+尺度对齐+`-exp(mean_dist/0.3)` |
| reward.depth_reprojection | 物体级 / 全局 | 三接口：object / global / both |
| reward.videoalign | Overall / VQ / MQ / TA | 读预计算 JSON |
| reward.feature_sim | DINOv2 cosine similarity | 读预计算 JSON |
| reconstruction | 全局 / 物体级点云 CD | 支持 camera / umeyama / icp / **first_frame** 对齐 |

## 使用方式

### 方式一：通过 config.sh 配置（推荐）

编辑 `config.sh`，填好参数后直接运行：

```bash
# 1. 修改 config.sh 中的 OUTPUT_ROOT、METRICS 等参数
# 2. 直接运行
bash run_benchmark.sh
```

`config.sh` 中的主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OUTPUT_ROOT` | (必填) | 推理输出根目录 |
| `METRICS` | `all` | 要评测的指标，逗号分隔 |
| `GPU` | `0` | GPU 编号 |
| `ALIGN` | `both_align` | 重建对齐方式：camera / first_frame / both_align |
| `VBENCH_CACHE` | (空) | VBench 缓存目录，留空自动检测 |
| `SKIP_INTERMEDIATES` | `false` | 跳过中间产物生成 |
| `AGGREGATE_ONLY` | `false` | 仅汇总已有结果 |

### 方式二：命令行传参（覆盖 config.sh）

命令行参数会覆盖 `config.sh` 中的值：

```bash
# 全量评测
bash run_benchmark.sh --output_root /path/to/output --metrics all --gpu 0

# 按类别
bash run_benchmark.sh --metrics video_quality
bash run_benchmark.sh --metrics reward
bash run_benchmark.sh --metrics reconstruction

# 按子指标 / 细粒度接口
bash run_benchmark.sh --metrics reward.camera_pose
bash run_benchmark.sh --metrics reward.depth_reprojection.object

# 逗号组合
bash run_benchmark.sh --metrics video_quality.psnr,reward.camera_pose,reconstruction.global

# 重建对齐方式
bash run_benchmark.sh --metrics reconstruction --align first_frame

# 跳过中间产物 / 仅汇总
bash run_benchmark.sh --skip_intermediates
bash run_benchmark.sh --aggregate_only
```

### 输入格式要求

`output_root` 需匹配推理输出结构：

```
<output_root>/<dataset>/<sample_id>/
  gen_0.mp4 ~ gen_{N-1}.mp4
  start.png
  gt.mp4
  camera.txt
  gt_depth.npz          （可选，有则用真值深度）
  metadata.json
```

### 输出结构

每条生成视频的评测结果写入 `gen_X/eval/` 下的各 JSON 文件，最终汇总到：

```
<output_root>/benchmark_results/
  results.jsonl          # 每条 gen 视频一行
  summary.json           # 全量指标均值/中位数/标准差
```
