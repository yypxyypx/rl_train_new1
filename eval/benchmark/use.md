# Benchmark 评测系统

## 代码结构

```
eval/benchmark/
├── run_benchmark.sh              # 一键入口（bash 包装器）
├── run_benchmark.py              # 主调度器：指标展开 → 中间产物 → 评测 → 汇总
├── common/
│   ├── scan.py                   # 扫描 output_root，自动检测 gen_0 ~ gen_{N-1}
│   ├── intermediate.py           # 中间产物调度（DA3/SAM3 视频追踪/DINOv2/VideoAlign/VBench）
│   └── utils.py                  # IO、conda 运行器、parse_camera_txt
├── video_quality/
│   ├── psnr_ssim_lpips.py        # PSNR / SSIM / LPIPS
│   ├── vbench_eval.py            # VBench（从预计算 JSON 读取）
│   └── run_video_quality.py      # 独立入口
├── reward/
│   ├── camera_pose.py            # 旋转 AUC + 平移方向 AUC + Pose AUC + 平移距离
│   ├── depth_reprojection.py     # 物体级 / 全局深度重投影
│   ├── videoalign_eval.py        # VideoAlign 评分
│   ├── feature_sim.py            # DINOv2 特征相似度
│   └── run_reward.py             # 独立入口
└── reconstruction/
    ├── global_point_cloud.py     # 全局点云一致性（camera/first_frame/umeyama/icp）
    ├── object_point_cloud.py     # 物体级点云一致性（camera/first_frame/umeyama/icp）
    └── run_reconstruction.py     # 独立入口

config/
└── benchmark.sh                  # 唯一配置文件
```

## SAM3 中间产物说明

`intermediate.py` 通过 `worker_sam3.py`（conda 环境：`SAM3`）生成语义分割中间产物。
SAM3 使用**视频追踪模式**（`propagate_in_video`）：以 Qwen3-VL 识别的物体名称为文本提示，
从首帧开始向后传播 mask，跨帧保持物体一致性，比逐帧独立分割更稳定。

| 文件 | 内容 |
|------|------|
| `gt_objects.json` / `pred_objects.json` | Qwen 识别的物体名称列表 |
| `gt_label_maps.npz` / `label_maps.npz` | SAM3 视频追踪输出，`label_maps (T,H,W) int16`，0=背景，1~K=物体类别 |
| `gt_masks.npz` / `pred_masks.npz` | per-object masks `(N_obj,T,H,W) bool` + `mean_areas` |

pred 视频的物体列表复用对应 GT 视频的 `gt_objects.json`，保持类别对齐。

---

## 配置与参数总览

所有参数在 `config/benchmark.sh` 中配置，命令行参数可覆盖。

| 参数 | 配置项 | CLI 参数 | 默认值 | 作用范围 | 说明 |
|------|--------|----------|--------|----------|------|
| 输出目录 | `OUTPUT_ROOT` | `--output_root` | (必填) | 全局 | 推理输出根目录 |
| 指标选择 | `METRICS` | `--metrics` | `all` | 全局 | 逗号分隔，见下方指标表 |
| GPU 编号 | `GPU` | `--gpu` | `0` | 全局 | 设置 CUDA_VISIBLE_DEVICES |
| 设备名 | `DEVICE` | `--device` | `cuda:<GPU>` | 全局 | 覆盖 CUDA 设备名 |
| 对齐方式 | `ALIGN` | `--align` | `both_align` | reconstruction.* | 见下方对齐模式说明 |
| FPS 采样点数 | `N_FPS` | `--n_fps` | `20000` | reconstruction.* | 全局点云采样数，物体级按面积比例分配 |
| 置信度阈值 | `CONF_THRESH` | `--conf_thresh` | `0.0` | reconstruction.* | 低于此值的深度像素不参与点云 |
| VBench 缓存 | `VBENCH_CACHE` | `--vbench_cache` | (自动) | video_quality.vbench | VBench 模型缓存路径 |
| 跳过中间产物 | `SKIP_INTERMEDIATES` | `--skip_intermediates` | `false` | 全局 | 假设 DA3/SAM3 等已生成 |
| 仅汇总 | `AGGREGATE_ONLY` | `--aggregate_only` | `false` | 全局 | 不跑评测，仅汇总已有 JSON |

## 点云对齐模式（--align）

仅影响 `reconstruction.*` 指标。

### 四种基础模式

| 模式 | 对齐方式 | 内参来源 | 外参来源 | 适用场景 |
|------|---------|---------|---------|---------|
| `camera` | Umeyama 对齐相机位置 → 求 scale | pred 内参 | GT 外参 | 相机轨迹方向基本一致，仅尺度不同 |
| `first_frame` | 首帧 pose 对齐 + 轨迹长度求 scale | pred 内参 | 对齐后 pred 外参 | **最接近 RL 训练 reward 的方式**，推荐作为主指标 |
| `umeyama` | 各用自己的相机参数重投影，再 Umeyama 相似变换对齐点云 | pred 内参 | pred 外参 | 允许旋转+缩放，不假设坐标系一致 |
| `icp` | umeyama 基础上再做 ICP 精化 | pred 内参 | pred 外参 | 最精确，耗时最长 |

### 两种组合快捷

| 模式 | 展开为 | 输出 JSON 结构 |
|------|-------|--------------|
| `both_align` | camera + first_frame（**默认**） | `{"camera": {...}, "first_frame": {...}}` |
| `all_align` | camera + first_frame + umeyama + icp | `{"camera": {...}, "first_frame": {...}, "umeyama": {...}, "icp": {...}}` |

### 示例

```bash
bash eval/benchmark/run_benchmark.sh --metrics reconstruction --align camera
bash eval/benchmark/run_benchmark.sh --metrics reconstruction --align first_frame
bash eval/benchmark/run_benchmark.sh --metrics reconstruction --align umeyama
bash eval/benchmark/run_benchmark.sh --metrics reconstruction --align icp
bash eval/benchmark/run_benchmark.sh --metrics reconstruction --align both_align  # 默认
bash eval/benchmark/run_benchmark.sh --metrics reconstruction --align all_align   # 四种全跑
```

## 指标选择

### 可用指标及其输出

| 指标 key | 描述 | 输出 JSON | 关键字段 |
|----------|------|-----------|----------|
| `video_quality.psnr` | PSNR / SSIM / LPIPS | `psnr_ssim_lpips.json` | psnr, ssim, lpips |
| `video_quality.vbench` | VBench | `vbench.json` | i2v_subject, i2v_background |
| `reward.camera_pose` | 相机轨迹评估 | `camera_pose.json` | rotation_auc30, translation_auc30, pose_auc30, translation_metric |
| `reward.depth_reprojection.object` | 物体级深度重投影 | `depth_reprojection.json` | object.reward |
| `reward.depth_reprojection.global` | 全局深度重投影 | `depth_reprojection.json` | global.reward |
| `reward.videoalign` | VideoAlign | `videoalign.json` | Overall |
| `reward.feature_sim` | DINOv2 特征相似度 | `feature_sim.json` | reward_feature_sim |
| `reconstruction.global` | 全局点云一致性 | `global_point_cloud.json` | chamfer_distance, fscore |
| `reconstruction.object` | 物体级点云一致性 | `object_point_cloud.json` | summary.chamfer_distance |

### 快捷分组

| 快捷 key | 展开为 |
|----------|--------|
| `all` | 全部 9 项指标 |
| `video_quality` | psnr + vbench |
| `reward` | camera_pose + depth_reprojection.both + videoalign + feature_sim |
| `reward.depth_reprojection` | depth_reprojection.both（object + global） |
| `reconstruction` | reconstruction.both（global + object） |

### camera_pose 输出字段说明

| 字段 | 说明 |
|------|------|
| `rotation_auc30` | 纯旋转误差的 AUC@30° |
| `translation_auc30` | 纯平移方向误差的 AUC@30° |
| `pose_auc30` | 组合 AUC@30°（max(旋转,平移方向)，标准 Pose AUC） |
| `translation_metric` | 首帧对齐 + 尺度对齐后的平移距离指标 |

## 使用方式

### 方式一：通过 config 配置（推荐）

```bash
# 1. 编辑 config/benchmark.sh 中的 OUTPUT_ROOT、METRICS 等
# 2. 直接运行
bash eval/benchmark/run_benchmark.sh
```

### 方式二：命令行传参（覆盖 config）

```bash
# 全量评测
bash eval/benchmark/run_benchmark.sh --output_root /path/to/output --metrics all --gpu 0

# 按类别
bash eval/benchmark/run_benchmark.sh --metrics video_quality
bash eval/benchmark/run_benchmark.sh --metrics reward
bash eval/benchmark/run_benchmark.sh --metrics reconstruction

# 按子指标
bash eval/benchmark/run_benchmark.sh --metrics reward.camera_pose
bash eval/benchmark/run_benchmark.sh --metrics reward.depth_reprojection.object

# 逗号组合
bash eval/benchmark/run_benchmark.sh --metrics video_quality.psnr,reward.camera_pose,reconstruction.global

# 重建对齐方式
bash eval/benchmark/run_benchmark.sh --metrics reconstruction --align first_frame

# 调整点云参数
bash eval/benchmark/run_benchmark.sh --metrics reconstruction --n_fps 50000 --conf_thresh 0.5

# 跳过中间产物 / 仅汇总
bash eval/benchmark/run_benchmark.sh --skip_intermediates
bash eval/benchmark/run_benchmark.sh --aggregate_only
```

### 方式三：独立入口（单模块调试用）

每个子模块有独立 Python 入口，可以跳过主调度器直接运行：

```bash
# 仅 reward 类指标
python eval/benchmark/reward/run_reward.py --output_root /path/to/output --mode camera_pose

# 仅重建指标（可指定 n_fps）
python eval/benchmark/reconstruction/run_reconstruction.py \
    --output_root /path/to/output --mode global --align first_frame --n_fps 50000

# 仅视频质量
python eval/benchmark/video_quality/run_video_quality.py --output_root /path/to/output --mode psnr
```

## 输入格式

```
<output_root>/<dataset>/<sample_id>/
  gen_0.mp4 ~ gen_{N-1}.mp4
  start.png
  gt.mp4
  camera.txt
  gt_depth.npz          （可选，有则用真值深度）
  metadata.json
```

## 输出结构

```
<output_root>/
  <dataset>/<sample_id>/gen_X/
    eval/                          # 各指标结果
      camera_pose.json
      psnr_ssim_lpips.json
      depth_reprojection.json
      global_point_cloud.json
      object_point_cloud.json
      ...
    intermediates/                  # 中间产物（DA3/SAM3 视频追踪等）
      da3_pred.npz
      pred_masks.npz               # SAM3 视频追踪生成的 per-object masks
      label_maps.npz               # SAM3 视频追踪生成的 label_maps (T,H,W) int16
      ...
  benchmark_results/
    results.jsonl                  # 每条 gen 视频一行
    summary.json                   # 全量指标均值/中位数/标准差
```
