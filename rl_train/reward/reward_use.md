# Reward Pipeline 使用文档

## 目录结构

```
RL_code/
├── config/
│   └── reward.sh                  # 统一配置文件（两种模式均从这里读默认值）
└── rl_train/
    └── reward/
        ├── reward_use.md          # 本文档
        ├── reward.sh              # 单视频模式入口脚本
        ├── batch_reward.sh        # 批量模式入口脚本
        ├── reward_pipeline.py     # Pipeline 调度器（核心）
        ├── reward_metrics.py      # 指标计算（核心）
        └── steps/
            ├── step_da3.py        # Step 1: DA3 深度估计 + 相机位姿
            ├── step_qwen_sam3.py  # Step 2: Qwen3-VL 物体识别 + SAM3 视频追踪
            ├── step_dinov2_featup.py  # Step 3: DINOv2+FeatUp 特征相似度
            └── step_videoalign.py # Step 4: VideoAlign 视频质量评分
```

---

## 快速开始

### 方式一：修改配置文件，零参数运行（推荐）

1. 编辑 `RL_code/config/reward.sh`，填写对应变量
2. 直接运行脚本，无需传参：

```bash
# 批量模式（处理一个 sample 下的全部 rollout）
bash rl_train/reward/batch_reward.sh

# 单视频模式
bash rl_train/reward/reward.sh
```

### 方式二：命令行传参（覆盖配置文件中的值）

```bash
# 批量模式
bash rl_train/reward/batch_reward.sh \
    --sample_dir /path/to/dl3dv/0a6c01ac32127687 \
    --output_dir /path/to/reward_output \
    --rewards all \
    --gpu 1

# 单视频模式
bash rl_train/reward/reward.sh \
    --video_path /path/to/gen_0.mp4 \
    --gt_camera_txt /path/to/camera.txt \
    --work_dir /path/to/work_dir \
    --rewards all \
    --gpu 1
```

> 命令行参数会覆盖配置文件中的同名变量，两者可以混用。

---

## 配置文件说明（`config/reward.sh`）

| 变量 | 适用模式 | 说明 |
|------|----------|------|
| `VIDEO_PATH` | 单视频 | 生成视频路径（.mp4） |
| `GT_CAMERA_TXT` | 单视频 | GT 相机轨迹文件（camera.txt） |
| `WORK_DIR` | 单视频 | 中间产物和结果输出目录 |
| `SAMPLE_DIR` | 批量 | 推理输出目录（含 `infer_info.json` + `gen_k.mp4`） |
| `OUTPUT_DIR` | 批量 | 结果根目录（自动创建 `<dataset>/<sample_id>/` 子目录） |
| `REWARDS` | 共享 | 计算哪些 reward（见下表），默认 `all` |
| `GPU` | 共享 | GPU 编号，默认 `0` |
| `PROMPT` | 共享 | VideoAlign 视频描述文本 |
| `METADATA_JSON` | 共享 | 含 caption 字段的 JSON，自动提取 prompt |
| `NO_SKIP` | 共享 | 填 `--no_skip` 强制重跑所有 steps |

### `--rewards` 可选值

| 值 | 含义 | 触发的 Steps |
|----|------|-------------|
| `all` | 全部 reward | DA3 + SAM3（视频追踪）+ DINOv2 + VideoAlign |
| `geo_semantic` | 几何+语义一致性 | DA3 + SAM3（视频追踪）|
| `geo_global` | 全局几何一致性 | DA3 |
| `feature_sim` | DINOv2 特征相似度 | DA3 + DINOv2/FeatUp |
| `camera_traj` | 相机轨迹误差 | DA3 |
| `video_quality` | 视频质量评分 | VideoAlign |

多个 reward 用逗号组合，例如 `geo_semantic,camera_traj`。

---

## 输入数据格式

### 批量模式（`batch_reward.sh`）

```
sample_dir/
├── infer_info.json     # {"dataset": "dl3dv", "sample_id": "xxx", "num_rollouts": 8, ...}
├── camera.txt          # GT 相机轨迹（所有 rollout 共用）
├── gen_0.mp4           # 第 0 个 rollout 的生成视频
├── gen_1.mp4
└── ...
```

### `camera.txt` 格式

每行 19 个空格分隔的浮点数：

```
frame_idx  fx  fy  cx  cy  d1  d2  w2c[0,0] w2c[0,1] ... w2c[2,3]
```

- `fx`, `fy`, `cx`, `cy`：归一化内参（像素值 = 归一化值 × 图像宽/高）
- `d1`, `d2`：畸变系数（当前未使用）
- `w2c[0,0]...w2c[2,3]`：world-to-camera 矩阵（OpenCV 约定），行主序展开的 3×4 矩阵

---

## 输出结构

### 批量模式输出

```
output_dir/
└── dl3dv/
    └── 0a6c01ac32127687/
        ├── gen_0/
        │   ├── frames/                      # 从视频抽取的帧图像
        │   ├── intermediates/
        │   │   ├── da3_output.npz           # DA3 输出（深度/位姿/内参/置信度）
        │   │   ├── objects.json             # Qwen 识别的物体列表
        │   │   ├── label_maps.npz           # SAM3 语义标签图
        │   │   ├── feature_sim_reward.json  # DINOv2 特征相似度中间结果
        │   │   └── videoalign.json          # VideoAlign 评分
        │   ├── reward.json                  # 标量 reward 汇总（无 details）
        │   └── reward_details.json          # 完整结果（含逐帧 details）
        ├── gen_1/ ... gen_7/
        └── summary.json                     # 全部 rollout 标量 reward 汇总
```

### `reward.json` 格式

```json
{
  "reward_geo_semantic": 0.45,
  "reward_geo_global": 0.92,
  "reward_feature_sim": 0.70,
  "reward_camera_traj": -3.91,
  "reward_video_quality": -3.06,
  "reward_total": -12.34,
  "reward_total_global": -10.22
}
```

### `summary.json` 格式

```json
{
  "sample_id": "0a6c01ac32127687",
  "dataset": "dl3dv",
  "num_rollouts": 8,
  "rewards": ["geo_semantic", "geo_global", "feature_sim", "camera_traj", "video_quality"],
  "rollouts": [
    {"gen_id": 0, "reward_geo_semantic": 0.41, "reward_camera_traj": -3.5, "reward_total": -11.2, ...},
    {"gen_id": 1, ...},
    ...
  ]
}
```

---

## 代码架构

### 整体流程

```
batch_reward.sh / reward.sh
        │
        ▼
reward_pipeline.py
  ├── run_batch_pipeline()   # 批量：循环调用 run_pipeline()
  └── run_pipeline()         # 单视频：抽帧 → 调 steps → 计算 reward
        │
        ├── extract_frames()           # cv2 抽帧 → frames/
        ├── subprocess: step_da3.py    # conda env: rl_da3
        ├── subprocess: step_qwen_sam3.py  # conda env: SAM3（视频追踪模式）
        ├── subprocess: step_dinov2_featup.py  # conda env: rl_da3
        ├── subprocess: step_videoalign.py     # conda env: Videoalign
        │
        └── compute_all_rewards()      # reward_metrics.py
              ├── compute_reward_geo_semantic()
              ├── compute_reward_geo_global()
              ├── compute_reward_camera_traj()
              ├── compute_reward_video_quality()
              └── 读取 feature_sim_reward.json（feature_sim 在 step 里算完）
```

### Reward 与 Step 依赖关系

| Reward | 依赖 Steps | 中间产物 |
|--------|-----------|---------|
| `geo_semantic` | step_da3 + step_qwen_sam3（视频追踪）| `da3_output.npz` + `label_maps.npz` |
| `geo_global` | step_da3 | `da3_output.npz` |
| `camera_traj` | step_da3 | `da3_output.npz` |
| `feature_sim` | step_da3 + step_dinov2_featup | `da3_output.npz` + `feature_sim_reward.json` |
| `video_quality` | step_videoalign | `videoalign.json` |

> `step_da3` 被多个 reward 共用，pipeline 自动去重，只跑一次。

### 各 Reward 计算方式

#### `geo_semantic` — 几何+语义一致性

以第 0 帧为参考帧，对每一帧 `i`：
1. 用 DA3 预测的深度和相机位姿，将帧 `i` 的像素通过三维重投影 warp 到参考帧坐标系
2. 对比 warp 后的深度和参考帧真实深度（相对误差 → `exp(-err)` 转为 [0,1] 分数）
3. 同时对比 warp 后的 SAM3 语义标签是否与参考帧一致（前景匹配权重 1.0，背景匹配权重 0.8）
4. 对所有帧取均值

> SAM3 分割由 `step_qwen_sam3.py` 完成，使用 **SAM3 视频追踪模式**（`propagate_in_video`）：以首帧文本提示为锚点，通过帧间传播追踪物体 mask，比逐帧独立分割更稳定，覆盖率更高。

#### `geo_global` — 全局几何一致性

与 `geo_semantic` 类似，但只比较深度，不涉及语义标签。

#### `camera_traj` — 相机轨迹误差

1. **首帧对齐**：将预测轨迹的第 0 帧 pose 对齐到 GT（在 w2c 空间做变换）
2. **轨迹长度缩放**：将预测轨迹缩放至与 GT 轨迹等长
3. **旋转误差**：相邻帧间 relative rotation 的角度差（degree），取均值，越小越好
4. **平移误差**：每帧相机位置与 GT 的距离，除以 GT 轨迹总长度归一化，再取 `-exp(dist/0.3)` 转为奖励
5. 两者加权组合（默认 rot_weight=0.5, trans_weight=0.5）

> 归一化说明：`per_frame_dist` 除以 `gt_traj_len`，确保不同场景规模下数值稳定。

#### `feature_sim` — DINOv2 特征相似度

在 `step_dinov2_featup.py` 中计算（不在 `reward_metrics.py`，因为需要 GPU 特征提取）：
1. 提取每帧的 DINOv2（+FeatUp JBU 上采样器）特征图
2. 用 DA3 几何将帧 `i` 的特征 warp 到参考帧
3. 计算与参考帧特征的余弦相似度，取有效像素均值
4. `reward = 1 - mean_dissimilarity`（越高越好）

结果以 `feature_sim_reward.json` 形式保存，`compute_all_rewards()` 直接读取。

#### `video_quality` — 视频质量

直接调用 VideoAlign（VideoReward 模型）对整个视频打分，返回：
- `VQ`（Visual Quality）、`MQ`（Motion Quality）、`TA`（Text Alignment）
- `Overall`（三者之和）= 直接作为 `reward_video_quality`

### 断点续算

Pipeline 默认跳过已存在的中间产物（`skip_done=True`）。如果某个 step 已经跑过，直接复用结果，只重算后续步骤。用 `--no_skip` 可强制全部重跑。

### Conda 环境与 GPU

各 step 通过 subprocess 在独立 conda 环境中运行，GPU 通过 `CUDA_VISIBLE_DEVICES` 透传：

| Step | Conda 环境 | 说明 |
|------|-----------|------|
| step_da3 | `rl_da3` | DA3 深度估计 |
| step_dinov2_featup | `rl_da3` | DINOv2 特征相似度 |
| step_qwen_sam3 | `SAM3` | Qwen 物体识别 + SAM3 **视频追踪**（`propagate_in_video`）|
| step_videoalign | `Videoalign` | VideoAlign 视频质量评分 |

`reward_pipeline.py`（主进程）本身运行在 `rl_da3` 环境中（由 `batch_reward.sh` / `reward.sh` 调用）。

---

## 常见用法示例

### 只重算 camera_traj（其他 step 已有结果，跳过）

```bash
bash rl_train/reward/batch_reward.sh \
    --sample_dir /horizon-bucket/robot_lab/users/puxin.yan-labs/test_output1/dl3dv/0a6c01ac32127687 \
    --output_dir RL_code/rl_train/reward_output \
    --rewards camera_traj \
    --gpu 1
```

### 强制重跑所有 steps

```bash
bash rl_train/reward/batch_reward.sh \
    --sample_dir /path/to/sample \
    --output_dir /path/to/output \
    --rewards all \
    --gpu 1 \
    --no_skip
```

### 只跑单个 rollout 测试

```bash
bash rl_train/reward/reward.sh \
    --video_path /path/to/gen_0.mp4 \
    --gt_camera_txt /path/to/camera.txt \
    --work_dir /tmp/reward_test \
    --rewards geo_global,camera_traj \
    --gpu 1
```

### 消融实验：对比不同 reward 组合

通过多次调用 `batch_reward.sh --rewards <组合>` 并将结果输出到不同 `output_dir`，然后对比各自的 `summary.json`。
