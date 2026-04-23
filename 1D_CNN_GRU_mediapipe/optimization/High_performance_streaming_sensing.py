import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. 路径与全局配置
# ==========================================
DATA_BASE_DIR = "/home/ubuntu/New_Architecture/data"
MODEL_WEIGHTS = "/home/ubuntu/New_Architecture/1D_CNN_GRU_mediapipe/trained_model/best_mp_gesture_model.pth"
MODEL_POSE = "/home/ubuntu/New_Architecture/MediaPipe_Models/pose_landmarker_lite.task"
MODEL_HANDS = "/home/ubuntu/New_Architecture/MediaPipe_Models/hand_landmarker.task"
REPORT_DIR = "/home/ubuntu/New_Architecture/1D_CNN_GRU_mediapipe/optimization/output/Test_Report"
CHART_DIR = "/home/ubuntu/New_Architecture/1D_CNN_GRU_mediapipe/optimization/output/chart"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

ACTION_NAMES = {
    0: "招手", 1: "完全静止", 2: "推摆(退后)", 3: "左横移", 4: "右横移",
    5: "左转", 6: "右转", 7: "坐下", 8: "起立", 9: "伸懒腰",
    10: "打滚", 11: "摆姿势", 12: "拜年/作揖"
}

# ==========================================
# 2. 模型架构 (1D-CNN+GRU)
# ==========================================
class RobotGestureClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(RobotGestureClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(225, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.gru = nn.GRU(256, 128, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

# ==========================================
# 3. 核心工具函数
# ==========================================
def extract_lightning_features(pose_res, hand_res):
    keypoints = np.zeros((75, 3))
    if pose_res.pose_landmarks:
        for i, lm in enumerate(pose_res.pose_landmarks[0]):
            keypoints[i] = [lm.x, lm.y, lm.visibility]
    if hand_res.hand_landmarks:
        for i, landmarks in enumerate(hand_res.hand_landmarks):
            label = hand_res.handedness[i][0].category_name
            offset = 33 if label == "Left" else 54
            for j, lm in enumerate(landmarks[:21]):
                keypoints[offset + j] = [lm.x, lm.y, 1.0]
    return keypoints

def get_true_label(ep):
    limits = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    for i, limit in enumerate(limits):
        if ep <= limit: return i
    return 0

# ==========================================
# 4. 终极 E2E 测试主逻辑
# ==========================================
def run_e2e_lightning_test():
    device = torch.device("cuda")
    print(f"🎬 启动 VLA 全链路端到端测试... (计算设备: {device})")

    model = RobotGestureClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    def create_opts(m_path):
        is_pose = "pose" in m_path
        base_opts = python.BaseOptions(model_asset_path=m_path, delegate=python.BaseOptions.Delegate.GPU)
        if is_pose:
            return vision.PoseLandmarkerOptions(base_options=base_opts, running_mode=vision.RunningMode.VIDEO)
        else:
            return vision.HandLandmarkerOptions(base_options=base_opts, running_mode=vision.RunningMode.VIDEO, num_hands=2)

    pose_engine = vision.PoseLandmarker.create_from_options(create_opts(MODEL_POSE))
    hand_engine = vision.HandLandmarker.create_from_options(create_opts(MODEL_HANDS))

    TEST_N = 20
    test_eps = random.sample(range(1, 701), TEST_N)
    test_eps.sort()

    results_log = []
    sense_times, brain_times, total_times, episode_labels = [], [], [], []
    correct_count = 0

    print(f"✅ 环境预热成功，正在执行动作序列识别...\n")

    # 全局连续时间戳，规避倒流报错
    global_ts_ms = 0 

    for ep in test_eps:
        ep_folder = f"episode_{ep:03d}"
        img_dir = os.path.join(DATA_BASE_DIR, ep_folder, "images")
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))

        features_buffer = []
        t_sense_start = time.time()
        for i, f_name in enumerate(img_files):
            frame = cv2.imread(os.path.join(img_dir, f_name))
            rgb_frame = cv2.cvtColor(cv2.resize(frame, (480, 480)), cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            p_res = pose_engine.detect_for_video(mp_img, global_ts_ms)
            h_res = hand_engine.detect_for_video(mp_img, global_ts_ms)
            features_buffer.append(extract_lightning_features(p_res, h_res))
            
            global_ts_ms += 33 
        
        # 记录单个 Episode 的 MediaPipe 总耗时
        mp_total_s = time.time() - t_sense_start
        sense_times.append(mp_total_s)

        seq = np.array(features_buffer)
        if len(seq) < 100:
            seq = np.pad(seq, ((100-len(seq), 0), (0,0), (0,0)), mode='edge')
        
        input_tensor = torch.from_numpy(seq[-100:]).float().view(1, 100, -1).to(device)

        t_brain_start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            pred_id = torch.argmax(output).item()
            conf = torch.softmax(output, dim=1)[0][pred_id].item()
        
        # 记录单个 Episode 的推理总耗时 (转为秒)
        infer_total_s = time.time() - t_brain_start
        brain_times.append(infer_total_s)

        episode_total_s = mp_total_s + infer_total_s
        total_times.append(episode_total_s)

        true_id = get_true_label(ep)
        is_ok = (pred_id == true_id)
        if is_ok:
            correct_count += 1
            
        episode_labels.append(ep_folder)
        
        # 严格按照要求的 Markdown 格式生成行
        results_log.append(
            f"| {ep_folder} | {ACTION_NAMES[true_id]} | {ACTION_NAMES[pred_id]} | "
            f"{conf*100:.2f}% | {mp_total_s:.3f} | {infer_total_s:.4f} | {episode_total_s:.3f} | {'✅' if is_ok else '❌'} |"
        )

    # ==========================================
    # 5. 生成标准 Markdown 报告
    # ==========================================
    acc = (correct_count / TEST_N) * 100
    avg_mp = np.mean(sense_times)
    avg_inf_ms = np.mean(brain_times) * 1000 # 转换为毫秒用于头部展示
    avg_tot = np.mean(total_times)

    summary_text = (
        f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"最终准确率: {acc:.2f}%\n"
        f"MediaPipe 平均: {avg_mp:.3f} s | 推理平均: {avg_inf_ms:.1f} ms | 总计平均: {avg_tot:.3f} s\n\n"
        "| Episode | 真实动作 | 预测动作 | 置信度 | MediaPipe(s) | 推理(s) | 总耗时(s) | 结论 |\n"
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |\n"
    )
    summary_text += "\n".join(results_log)

    # 打印到控制台
    print(summary_text)

    # 保存到文本文件
    report_file = os.path.join(REPORT_DIR, f"vla_e2e_report_{int(time.time())}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # 绘制图表 (使用毫秒级别展示会更好看)
    plt.figure(figsize=(12, 6), dpi=120)
    plt.bar(episode_labels, [t*1000 for t in sense_times], label='Sensing (Pose+Hands)', color='#2ecc71')
    plt.bar(episode_labels, [t*1000 for t in brain_times], bottom=[t*1000 for t in sense_times], label='Decision (CNN-GRU)', color='#3498db')
    plt.ylabel('Latency (ms)')
    plt.title('VLA 2.0 Full Stack Latency Breakdown (RTX 4090)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "vla_e2e_bench.png"))
    
    print(f"\n✅ 报告已生成: {report_file}\n📈 图表已生成: {CHART_DIR}")

if __name__ == "__main__":
    run_e2e_lightning_test()