# Wan2.2 框架端到端推理验证

这个目录用来验证 `wan2_2/` 主代码（model_loader / wan22_encode / grpo_core.run_sample_step）
的整条 rollout 链路是否正确。**不放任何业务逻辑代码**，只放启动脚本和产物。

## 用法

```bash
cd rl_train_new/rl_train/train/wan2_2_sandbox/
bash run_framework_verify.sh                          # 默认 GPU=0
GPU=3 NUM_ROLLOUTS=4 bash run_framework_verify.sh     # 改卡 / 改条数
```

脚本调用 `../wan2_2/infer_only.py`，对两条样本各产出 4 条 rollout。
seed = `seed_base + k * 1000`（默认 42 / 1042 / 2042 / 3042）保证不同 rollout 噪声不同。

## 输出结构

```
outputs/
├── verify.log                       # tee 后的全部 stdout/stderr
└── dl3dv/
    ├── 0a1b7c20a92c43c6/
    │   ├── infer_info.json
    │   ├── camera.txt               # 自动从 GT 复制
    │   └── gen_0.mp4 ... gen_3.mp4
    └── 0a78c25f77c1ba1d/
        └── ...
```

## 判读标准

成功通过：
- ✅ 8 条 mp4 全部产出，文件大小 > 0；
  ```bash
  ffprobe -v quiet -print_format json -show_streams outputs/dl3dv/0a1b7c20a92c43c6/gen_0.mp4
  ```
  能解析出 49 帧 × 1280×704 × 16 fps。
- ✅ 同一样本的 4 条 rollout 视觉上**互不相同**（不同 seed → 应有不同细节/动态）。
- ✅ 跨样本对比能看到内容明显不同（scene-aware）。
- ✅ camera.txt 自动复制成功，可直接喂给 `rl_train/reward/batch_reward.sh`。

失败排查：
- ❌ 4 条 rollout 完全一致 → 检查 `init_same_noise` 是否被错开启 / seed 复用。
- ❌ 黑屏 / 全噪点 → 检查 VAE3_8 加载、scaling_factor、decode 路径。
- ❌ 形变剧烈 / 不跟相机 → 检查：
  - `build_camera_control` 的 Plücker 构造（K_pix 单位、首帧对齐）；
  - `chunk_camera_control` 的 4-frame chunk 折叠维度顺序；
  - `encode_inpaint_conditions` 的 mask_latents / masked_video_latents。
- ❌ shape error / OOM → 看 verify.log 报错，先把 `--sampling_steps 5` 减小复现。

## 数据格式假设

样本目录由 `unified_data_process.py` 产出，需包含：
- `start.png`        — 首帧 RGB（实际推理直接从 gt.mp4 第 0 帧读，start.png 不强制）
- `gt.mp4`           — GT 视频（121 帧）
- `camera.txt`       — w2c OpenCV，归一化内参，**无 header**
- `metadata.json`    — 含 `caption` 字段（可选；缺省 fallback 到 "camera moving through a scene"）
