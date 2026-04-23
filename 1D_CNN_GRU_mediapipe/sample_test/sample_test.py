import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import numpy as np
import os
import random
import time

import matplotlib
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心架构定义 (RobotGestureClassifier)
# ==========================================
class RobotGestureClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(RobotGestureClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=225, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.gru = nn.GRU(256, 128, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        gru_out, h_n = self.gru(x)
        final_state = h_n[-1]
        out = self.fc(final_state)
        return out

# ==========================================
# 2. 路径与配置
# ==========================================
DATA_BASE_DIR = r"/home/ubuntu/New_Architecture/data"
MODEL_WEIGHTS = r"/home/ubuntu/New_Architecture/1D_CNN_GRU_mediapipe/trained_model/best_mp_gesture_model.pth"
TASK_PATH = r"/home/ubuntu/New_Architecture/MediaPipe_Holistic/holistic_landmarker.task"
OUTPUT_BASE = r"/home/ubuntu/New_Architecture/1D_CNN_GRU_mediapipe/sample_test/output"
REPORT_DIR = os.path.join(OUTPUT_BASE, "Test_Report")
CHART_DIR = os.path.join(OUTPUT_BASE, "chart", "Normal")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ACTION_NAMES = {
    0: "招手", 1: "完全静止", 2: "推摆(退后)", 3: "左横移", 4: "右横移",
    5: "左转", 6: "右转", 7: "坐下", 8: "起立", 9: "伸懒腰",
    10: "打滚", 11: "摆姿势", 12: "拜年/作揖"
}

def get_true_label(episode_num):
    if 1 <= episode_num <= 100: return 0
    elif 101 <= episode_num <= 150: return 1
    elif 151 <= episode_num <= 200: return 2
    elif 201 <= episode_num <= 250: return 3
    elif 251 <= episode_num <= 300: return 4
    elif 301 <= episode_num <= 350: return 5
    elif 351 <= episode_num <= 400: return 6
    elif 401 <= episode_num <= 450: return 7
    elif 451 <= episode_num <= 500: return 8
    elif 501 <= episode_num <= 550: return 9
    elif 551 <= episode_num <= 600: return 10
    elif 601 <= episode_num <= 650: return 11
    elif 651 <= episode_num <= 700: return 12
    return 0

def extract_features(result):
    keypoints = np.zeros((75, 3))
    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks):
            if i >= 33:
                break
            keypoints[i] = [lm.x, lm.y, getattr(lm, 'visibility', 1.0) or 0.0]
    if result.left_hand_landmarks:
        for i, lm in enumerate(result.left_hand_landmarks):
            if 33 + i >= 54:
                break
            keypoints[33+i] = [lm.x, lm.y, 1.0]
    if result.right_hand_landmarks:
        for i, lm in enumerate(result.right_hand_landmarks):
            if 54 + i >= 75:
                break
            keypoints[54+i] = [lm.x, lm.y, 1.0]
    return keypoints


def _save_bar_chart(path: str, title: str, y_label: str, labels: list[str], values: list[float], color: str):
    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    x = range(len(labels))
    ax.bar(x, values, color=color, edgecolor="black", linewidth=0.35)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_timing_charts(
    base_stem: str,
    episodes: list[str],
    mp_times: list[float],
    infer_times: list[float],
    totals: list[float],
):
    labels = episodes
    _save_bar_chart(
        os.path.join(CHART_DIR, f"{base_stem}_mediapipe.png"),
        "各样本 MediaPipe 耗时",
        "耗时 (s)",
        labels,
        mp_times,
        "#2ecc71",
    )
    _save_bar_chart(
        os.path.join(CHART_DIR, f"{base_stem}_inference.png"),
        "各样本 小脑(1D-CNN+GRU) 推理耗时",
        "耗时 (s)",
        labels,
        infer_times,
        "#3498db",
    )
    _save_bar_chart(
        os.path.join(CHART_DIR, f"{base_stem}_total.png"),
        "各样本总耗时 (MediaPipe + 对齐裁剪 + 推理)",
        "耗时 (s)",
        labels,
        totals,
        "#9b59b6",
    )

# ==========================================
# 3. 核心执行逻辑
# ==========================================
def run_batch_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 启动随机验证任务... 设备: {device}")

    model = RobotGestureClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE
    )

    TEST_N = 20
    test_episodes = random.sample(range(1, 701), TEST_N)
    test_episodes.sort()

    results_log = []
    correct_count = 0
    mp_times: list[float] = []
    infer_times: list[float] = []
    total_times: list[float] = []
    episode_labels: list[str] = []

    print(f"🎲 已随机选择样本: {test_episodes}")

    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for ep in test_episodes:
            t_sample0 = time.time()
            ep_folder = f"episode_{ep:03d}"
            img_dir = os.path.join(DATA_BASE_DIR, ep_folder, "images")
            true_label = get_true_label(ep)

            img_files = sorted(
                [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))],
                key=lambda x: int(x.split('.')[0])
            )

            video_features = []
            t_mp0 = time.time()
            for img_name in img_files:
                img_path = os.path.join(img_dir, img_name)
                frame = cv2.imread(img_path)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                res = landmarker.detect(mp_image)
                video_features.append(extract_features(res))
            mp_s = time.time() - t_mp0

            tensor_array = np.array(video_features)
            if len(tensor_array) > 150:
                tensor_array = tensor_array[:150]
            else:
                pad = 150 - len(tensor_array)
                if len(tensor_array) > 0:
                    tensor_array = np.concatenate(
                        (tensor_array, np.tile(tensor_array[-1], (pad, 1, 1))), axis=0
                    )
                else:
                    tensor_array = np.zeros((150, 75, 3))

            if true_label == 1:
                final_input = tensor_array[-100:, :, :]
            else:
                final_input = tensor_array[50:150, :, :]

            input_tensor = torch.from_numpy(final_input).float().view(100, -1).unsqueeze(0).to(device)
            t_inf0 = time.time()
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                pred_id = torch.argmax(probs).item()
                conf = probs[pred_id].item()
            infer_s = time.time() - t_inf0

            total_s = time.time() - t_sample0

            is_correct = (pred_id == true_label)
            if is_correct:
                correct_count += 1

            episode_labels.append(ep_folder)
            mp_times.append(mp_s)
            infer_times.append(infer_s)
            total_times.append(total_s)

            log_entry = {
                "episode": ep_folder,
                "true_name": ACTION_NAMES[true_label],
                "pred_name": ACTION_NAMES[pred_id],
                "conf": f"{conf*100:.2f}%",
                "result": "✅" if is_correct else "❌",
                "mp_s": mp_s,
                "infer_s": infer_s,
                "total_s": total_s,
            }
            results_log.append(log_entry)
            print(
                f"🔄 {ep_folder}: {log_entry['result']} | MP {mp_s:.2f}s | 推理 {infer_s*1000:.1f}ms | 合计 {total_s:.2f}s"
            )

    base_filename = "random_test_report"
    counter = 1
    while True:
        final_report_path = os.path.join(REPORT_DIR, f"{base_filename}_{counter}.txt")
        if not os.path.exists(final_report_path):
            break
        counter += 1

    report_stem = f"{base_filename}_{counter}"
    n = len(test_episodes)
    accuracy = correct_count / n if n else 0.0
    avg_mp = sum(mp_times) / n if n else 0.0
    avg_inf = sum(infer_times) / n if n else 0.0
    avg_tot = sum(total_times) / n if n else 0.0

    summary_lines = [
        "==========================================",
        f"📊 随机 {n} 样本测试报告 #{counter}",
        "==========================================",
        f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"最终准确率: {accuracy*100:.2f}%",
        f"MediaPipe 平均耗时: {avg_mp:.3f} s | 推理平均: {avg_inf*1000:.1f} ms | 单样本平均总耗时: {avg_tot:.3f} s",
        "",
        "| Episode | 真实动作 | 预测动作 | 置信度 | MediaPipe(s) | 推理(s) | 总耗时(s) | 结论 |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |",
    ]
    for log in results_log:
        summary_lines.append(
            f"| {log['episode']} | {log['true_name']} | {log['pred_name']} | {log['conf']} | "
            f"{log['mp_s']:.3f} | {log['infer_s']:.4f} | {log['total_s']:.3f} | {log['result']} |"
        )

    summary_text = "\n".join(summary_lines) + "\n"

    with open(final_report_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    save_timing_charts(report_stem, episode_labels, mp_times, infer_times, total_times)

    print("\n" + summary_text)
    print(f"💾 报告已保存至: {final_report_path}")
    print(f"📈 图表已保存至: {CHART_DIR}")


if __name__ == "__main__":
    run_batch_test()
