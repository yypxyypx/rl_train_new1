# Data Processing Module

## 代码结构

```
data/
├── datasets/                     # 数据集解析层
│   ├── __init__.py               # 注册表 DATASET_REGISTRY
│   ├── base.py                   # RawSample 定义 + 共用工具函数
│   ├── re10k.py                  # RealEstate10K 解析器
│   └── dl3dv.py                  # DL3DV 解析器
├── unified_data_process.py       # 统一转换：RawSample → 磁盘文件
├── process_data.sh               # bash 调用接口
└── data_use.md
```

## 两层流水线

```
原始数据集 ──[datasets/*.py]──→ RawSample (OpenCV c2w, 原始分辨率)
                                    │
                          [unified_data_process.py]
                                    │
                                    ▼
                    磁盘文件 (start.png, gt.mp4, camera.txt, metadata.json)
```

- **第一层**：每个数据集一个 py 文件，负责读取原始格式并将坐标系统一为 OpenCV c2w，不管分辨率
- **第二层**：`unified_data_process.py` 根据模型配置做 resize/crop、内参重算、写固定格式文件

## 使用方式

### 处理单个数据集

```bash
bash process_data.sh \
    --dataset re10k \
    --dataset_path /path/to/raw_data \
    --model gen3r \
    --output /path/to/processed \
    --skip_done
```

### 同时处理多个数据集

```bash
bash process_data.sh \
    --dataset re10k,dl3dv \
    --dataset_path /path/to/raw_data \
    --model gen3r \
    --output /path/to/processed \
    --skip_done
```

### 自定义分辨率和帧数

```bash
bash process_data.sh \
    --dataset dl3dv \
    --dataset_path /path/to/raw_data \
    --model gen3r \
    --target_size 512 \
    --num_frames 81 \
    --output /path/to/processed
```

### 全部参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `--dataset` | 是 | 数据集名称，支持逗号分隔：`re10k,dl3dv` |
| `--dataset_path` | 是 | 原始数据集根目录 |
| `--output` | 是 | 输出目录 |
| `--model` | 否 | 模型配置，默认 `gen3r`（560×560, 49帧） |
| `--target_size` | 否 | 覆盖模型默认分辨率 |
| `--num_frames` | 否 | 覆盖模型默认帧数 |
| `--sample_mode` | 否 | `fixed`（默认，从第0帧开始）或 `random` |
| `--max_samples` | 否 | 每个数据集最多处理多少样本，默认全部 |
| `--skip_done` | 否 | 跳过已有 gt.mp4 的样本 |
| `--include_depth` | 否 | 同时处理深度图（如果数据集提供） |

## 输出格式

每个样本输出到 `<output>/<dataset>/<sample_id>/`：

| 文件 | 内容 |
|------|------|
| `start.png` | 首帧图像 |
| `gt.mp4` | GT 视频 |
| `camera.txt` | 每行：`idx fx/W fy/H cx/W cy/H 0 0 <w2c 3×4>`，坐标系 OpenCV |
| `metadata.json` | 分辨率、caption、数据集来源、坐标系约定等 |
| `gt_depth.npz` | （可选）深度图 |

## 添加新数据集

1. 在 `datasets/` 下新建 `new_dataset.py`，实现：

```python
def parse(data_root: Path, max_samples: int = 0, verbose: bool = True) -> Iterator[RawSample]:
    # 读取原始数据，坐标系转为 OpenCV c2w
    yield RawSample(sample_id=..., frames=..., c2ws=..., Ks=..., ...)
```

2. 在 `datasets/__init__.py` 注册：

```python
from . import new_dataset
DATASET_REGISTRY["new_dataset"] = new_dataset.parse
```

## 添加新模型配置

在 `unified_data_process.py` 的 `MODEL_DEFAULTS` 中添加：

```python
MODEL_DEFAULTS = {
    "gen3r": {"target_size": 560, "num_frames": 49, "fps": 16},
    "wan22": {"target_size": 512, "num_frames": 81, "fps": 16},  # 新增
}
```

然后用 `--model wan22` 调用即可。
