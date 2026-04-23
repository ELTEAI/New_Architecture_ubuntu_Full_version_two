# High_performance_streaming_sensing.py 说明文档

## 1. 概述

本脚本在 **GPU** 上串联 **MediaPipe Tasks（姿态 + 双手）** 与 **PyTorch 1D-CNN+GRU** 分类器，对 `episode_001`～`episode_700` 格式的图像序列做 **端到端（E2E）抽检评测**：随机抽取若干 episode，逐帧提关键点、取末 100 帧送入网络，统计 **准确率**、**感知耗时**、**推理耗时**，并输出 **Markdown 报告**与 **堆叠柱状图**。

适用于无图形界面的服务器：Matplotlib 使用 **Agg** 后端。

---

## 2. 依赖环境

| 组件 | 用途 |
|------|------|
| Python 3.10+（建议） | 运行环境 |
| PyTorch + CUDA | `RobotGestureClassifier` 推理（脚本内写死 `device = cuda`） |
| OpenCV (`cv2`) | 读图、缩放、BGR→RGB |
| NumPy | 序列与 padding |
| MediaPipe ≥0.10（Tasks API） | `PoseLandmarker`、`HandLandmarker`，**非** Legacy `mp.solutions` |
| Matplotlib | 报表图（Agg） |

**模型文件（须与路径一致）：**

- `best_mp_gesture_model.pth`：与 `RobotGestureClassifier` 结构匹配的权重。
- `pose_landmarker_lite.task`、`hand_landmarker.task`：MediaPipe Tasks 资源文件。

---

## 3. 路径配置（脚本内常量）

当前为 **绝对路径**，换机器时请改为本机路径。

| 常量 | 说明 |
|------|------|
| `DATA_BASE_DIR` | 数据集根目录，其下为 `episode_XXX/images/` |
| `MODEL_WEIGHTS` | 1D-CNN+GRU 权重 `.pth` |
| `MODEL_POSE` | 姿态 landmarker `.task` |
| `MODEL_HANDS` | 手部 landmarker `.task`（`num_hands=2`） |
| `REPORT_DIR` | 文本报告输出目录 |
| `CHART_DIR` | 图表输出目录 |

运行时会自动 `makedirs` 报告与图表目录。

---

## 4. 数据目录约定

```
DATA_BASE_DIR/
  episode_001/
    images/
      0.jpg
      1.jpg
      ...
  episode_002/
    ...
```

- 脚本 **仅匹配后缀 `.jpg`**（大小写敏感），按文件名中 **点号前整数** 排序。
- Episode 编号取文件夹名 `episode_{ep:03d}` 中的 `ep`（1～700）。

---

## 5. 网络与特征

### 5.1 `RobotGestureClassifier`（1D-CNN + GRU）

- 输入张量形状：`(1, 100, 225)`，即 100 帧 × 每帧 75×3 展平。
- 结构要点：`Conv1d`（225→…）→ `GRU(256→128)` → 全连接 + Dropout → **13 类** logits。

### 5.2 `extract_lightning_features(pose_res, hand_res)`

- 输出 **`(75, 3)`**：
  - 索引 0–32：姿态 33 点（`x, y, visibility`）；
  - 33–53：左手 21 点（`handedness` 为 `Left`）；
  - 54–74：右手 21 点（否则写入右手槽位）。
- 依赖 **同一帧** 的 `PoseLandmarkerResult` 与 `HandLandmarkerResult`。

### 5.3 标签 `get_true_label(ep)`

按 **episode 编号区间** 映射 13 类（与数据采集协议一致），上界依次为  
100, 150, 200, …, 700；超出则归为 0 类。

---

## 6. MediaPipe 使用要点

- **API**：`mediapipe.tasks.python.vision` 的 `PoseLandmarker` / `HandLandmarker`。
- **模式**：`RunningMode.VIDEO`，每帧调用 `detect_for_video`。
- **Delegate**：`BaseOptions.Delegate.GPU`（需本机 GPU + 驱动/OpenGL ES 等环境可用；若初始化失败需自行改为 CPU 或加 try/回退，**当前脚本未实现回退**）。
- **时间戳**：`global_ts_ms` 在 **所有 episode、所有帧** 上 **严格递增**（每帧 +33 ms），否则 VIDEO 模式可能报错。

---

## 7. 单条 Episode 流程摘要

1. 读取该 episode 下全部 `.jpg`，resize 至 **480×480**，转 RGB，封装为 `mp.Image`。
2. 对每一帧：`pose_engine` 与 `hand_engine` 各 `detect_for_video` 一次，拼成 75×3 特征，得到序列。
3. 序列长度若不足 100，在 **时间维前方** 用 `np.pad(..., mode='edge')` 补到至少 100 帧。
4. 取 **最后 100 帧** `seq[-100:]`，展平为 `(1, 100, 225)`，GPU 上推理。
5. 记录该 episode 的 MediaPipe 总时间、推理时间、预测类别与置信度，与 `get_true_label(ep)` 对比。

---

## 8. 评测规模

- `TEST_N = 20`：从 **1～700** 中 **无放回随机** 抽 20 个 episode（若不足 700 仍按 `range(1,701)` 抽样）。

---

## 9. 输出物

| 输出 | 位置 / 命名 |
|------|-------------|
| Markdown 表格报告 | `REPORT_DIR/vla_e2e_report_{unix时间戳}.txt` |
| 堆叠柱状图 | `CHART_DIR/vla_e2e_bench.png`（**固定文件名**，多次运行会覆盖） |

报告含：测试时间、总体准确率、MediaPipe/推理/总耗时均值，以及每个 episode 的表格行（真实动作、预测动作、置信度、各段耗时、对错）。

图表：**绿色** 为感知段（Pose+Hands），**蓝色** 为 CNN-GRU 推理段，纵轴为毫秒。

---

## 10. 运行方式

```bash
cd /home/ubuntu/New_Architecture
source NewArc/bin/activate   # 或你的虚拟环境
python 1D_CNN_GRU_mediapipe/optimization/High_performance_streaming_sensing.py
```

须保证：CUDA 可用、数据与 `.task` / `.pth` 路径存在。

---

## 11. 已知限制与可改进点

1. **`device = torch.device("cuda")` 写死**：无 GPU 时会直接报错，可改为 `cuda if torch.cuda.is_available() else cpu`。
2. **仅 `.jpg`**：若数据为 `.png` 需改 `endswith` 或统一扩展名。
3. **Hand/Pose GPU 初始化无 try/except**：环境不支持 GPU delegate 时需在 `create_from_options` 处增加回退逻辑。
4. **图表文件名固定**：多轮实验会覆盖 `vla_e2e_bench.png`，可与报告一样加时间戳。

---

## 12. 文件位置

- 脚本：`1D_CNN_GRU_mediapipe/optimization/High_performance_streaming_sensing.py`
- 本文档：`1D_CNN_GRU_mediapipe/optimization/High_performance_streaming_sensing.md`
