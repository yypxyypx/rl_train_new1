# RL Train — 训练代码说明

## 目录结构

```
rl_train/
├── train/gen3r/              ← 主要代码（本文档对应此处）
├── reward/                   ← Reward 计算管线
└── reward_output/            ← 已有的 reward 计算结果
```

---

## gen3r/ 代码结构

```
train/gen3r/
├── train_grpo_v2.py          ← [新] 主训练入口（GRPO，torchrun 启动）
├── train_grpo.py             ← [旧] 原始训练脚本（参考用，可跑但 reward 有 bug）
│
├── config.py                 ← 所有超参数定义（dataclass + argparse）
├── model_loader_v2.py        ← [新] 精简模型加载（无 T5/geo_adapter/VGGT）
├── model_loader.py           ← [旧] 完整加载（含 T5，占 26GB）
│
├── grpo_engine.py            ← GRPO 三阶段核心：rollout / advantage / update+KL
├── grpo_core.py              ← SDE/ODE 采样步 + log_prob 计算
├── gen3r_encode.py           ← 编码工具：CLIP / VAE / Plücker / T5
│
├── reward_dispatcher.py      ← [新] 分布式 reward 调度（4 GPU 组并行）
├── reward_bridge.py          ← [旧] 串行 reward 接口（单 GPU）
│
├── t5_precompute.py          ← 离线预计算 T5 embedding（一次性，省 26GB 显存）
├── dataset_rl.py             ← 数据集（读 camera.txt + gt.mp4）
│
├── run_train.sh              ← [新] 16×5090 正式训练启动脚本
├── test_code/
│   ├── test_grpo_4090.sh     ← 端到端测试（4×4090，dry_run 模式）
│   ├── benchmark_reward.py   ← 4 GPU 并行 reward 速度测试
│   └── results/              ← benchmark 输出
└── Gen3R/                    ← Gen3R 模型代码（不动）
```

---

## 两套系统说明

### 旧系统（已验证可跑通）

**推理 → reward 计算，用于离线评估**

```bash
# Step 1: 推理生成视频
bash run_infer.sh \
    --data_root /path/to/data \
    --datasets dl3dv \
    --max_samples 4 \
    --output_dir /path/to/infer_output

# Step 2: 计算 reward（顺序，单 GPU）
bash ../../reward/batch_reward.sh \
    --sample_dir /path/to/infer_output/dl3dv/<sample_id> \
    --output_dir /path/to/reward_output \
    --rewards geo_semantic,feature_sim,video_quality \
    --gpu 0
```

调用链：`batch_reward.sh → reward_pipeline.py → step_da3 / step_qwen_sam3 / step_dinov2_featup / step_videoalign → compute_all_rewards()`

### 新系统（GRPO 训练，代码已写完，尚未完整运行过一轮）

**Step 0：预计算 T5 embedding（一次性）**

```bash
python t5_precompute.py \
    --pretrained_model_path /path/to/gen3r_ckpts \
    --data_root /path/to/data \
    --datasets re10k,dl3dv \
    --embed_dir /path/to/t5_cache
```

**Step 1：测试流程是否通（dry_run，不跑真实 reward 模型）**

```bash
bash test_code/test_grpo_4090.sh
```

**Step 2：正式训练（16×5090）**

```bash
bash run_train.sh
# 关键参数见脚本，主要设置：
#   GEN3R_MODEL_PATH, RL_DATA_ROOT, T5_EMBED_DIR, OUTPUT_DIR
```

---

## Reward 各项说明与当前状态

| Reward | 范围 | 当前状态 | conda env |
|--------|------|----------|-----------|
| `geo_semantic` | [0, 1] | ✅ 正常，约 0.4-0.5 | rl_da3 + SAM3 |
| `geo_global` | [0, 1] | ✅ 正常，约 0.89-0.94 | rl_da3 |
| `feature_sim` | [0, 1] | ✅ 正常，约 0.62-0.74 | rl_da3 |
| `camera_traj` | 理论[-1,0] | ❌ **公式 Bug**，实际输出 -1M 至 -50M | rl_da3 |
| `video_quality` | [-5, 0] | ✅ 正常，约 -3.1 至 -4.2 | Videoalign |

**camera_traj 的 Bug**：平移误差 `reward = -exp(dist / 0.3)`，当预测轨迹和 GT 偏差 >2m 时指数爆炸。训练前必须修复（改成有界函数，如 tanh 或 clip）。

---

## 显存规划（5090 32GB）

| 阶段 | 显存占用 |
|------|---------|
| Rollout（VAE decode 时峰值） | ~22-26 GB |
| Reward 计算（清空训练模型后） | ~8-14 GB（各 reward 模型） |
| GRPO Update（Transformer + Ref） | ~18-22 GB |

策略：各阶段之间主动 `.cpu()` + `torch.cuda.empty_cache()`，不同时驻留所有模型。

---

## 当前待做

1. **修 camera_traj 公式 bug**（reward_metrics.py）
2. **跑一次 dry_run 端到端**（test_grpo_4090.sh），确认梯度回传正常
3. **DA3 在空 GPU 上的速度实测**（benchmark 时 GPU 被占，OOM）
4. 正式启动 16×5090 训练
