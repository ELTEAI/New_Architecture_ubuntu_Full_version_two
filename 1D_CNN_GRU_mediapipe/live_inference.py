import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
import urllib.request

# ==========================================
# 1. 之前炼好的大脑网络 2.0 (内嵌在脚本中方便直接运行)
# ==========================================
class RobotGestureClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(RobotGestureClassifier, self).__init__()
        self.cnn = nn.Sequential(
            # 这里的 225 是 MediaPipe 75个点 * 3维坐标
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
# 2. 核心特征提取工具 (修复版的扁平列表遍历)
# ==========================================
def extract_holistic_features(detection_result):
    keypoints = np.zeros((75, 3))
    
    # 提取身体、左手、右手
    if detection_result.pose_landmarks:
        for i, lm in enumerate(detection_result.pose_landmarks):
            keypoints[i] = [lm.x, lm.y, getattr(lm, 'visibility', 1.0)]
            
    if detection_result.left_hand_landmarks:
        for i, lm in enumerate(detection_result.left_hand_landmarks):
            keypoints[33 + i] = [lm.x, lm.y, 1.0]
            
    if detection_result.right_hand_landmarks:
        for i, lm in enumerate(detection_result.right_hand_landmarks):
            keypoints[54 + i] = [lm.x, lm.y, 1.0]
            
    return keypoints

# ==========================================
# 3. 主程序：实机摄像头串流推理
# ==========================================
def run_live_inference():
    print("⏳ 正在加载 Unitree Go2 VLA 感知系统...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 当前推理设备: {device}")

    # --- A. 加载模型 ---
    MODEL_WEIGHTS = r"/home/ubuntu/New_Architecture/1D_CNN_GRU_mediapipe/trained_model/best_mp_gesture_model.pth"
    
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ 找不到权重文件: {MODEL_WEIGHTS}")
        return

    action_model = RobotGestureClassifier(num_classes=13).to(device)
    action_model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    action_model.eval() # 切记开启推理模式
    print("🧠 意图识别大脑已上线！")

    # --- B. 准备 MediaPipe 引擎 ---
    MODEL_DIR = r"/home/ubuntu/New_Architecture/MediaPipe_Holistic"
    MP_TASK_PATH = os.path.join(MODEL_DIR, "holistic_landmarker.task")
    
    if not os.path.exists(MP_TASK_PATH):
        print("⏳ 正在下载 MediaPipe Holistic 引擎...")
        url = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
        urllib.request.urlretrieve(url, MP_TASK_PATH)

    base_options = python.BaseOptions(model_asset_path=MP_TASK_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=False
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    print("👀 视觉感知模块已上线！")

    # --- C. 动作字典映射 ---
    action_names = {
        0: "Wave (ZhaoShou)", 1: "Still (JingZhi)", 2: "Push Back (TuiBai)", 
        3: "Lateral Left (ZuoHengYi)", 4: "Lateral Right (YouHengYi)", 
        5: "Rotate Left (ZuoZhuan)", 6: "Rotate Right (YouZhuan)", 
        7: "Sit (ZuoXia)", 8: "Rise (QiLi)", 9: "Stretch (ShenLanYao)", 
        10: "Wallow (DaGun)", 11: "Pose (BaiZiShi)", 12: "Scrape (BaiNian)"
    }

    # --- D. 初始化视频流与队列 ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查设备！")
        return

    # 100 帧记忆缓冲区
    frame_buffer = deque(maxlen=100)
    for _ in range(100):
        frame_buffer.append(np.zeros((75, 3)))

    print("==================================================")
    print("🎥 实时姿态捕获已开启！请站在镜头前做动作 (按 'q' 键退出)")
    print("==================================================")

    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        # 镜像翻转，体验更好
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 1. MediaPipe 推理
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = landmarker.detect(mp_image)

        # 2. 提取特征并压入队列
        current_kpts = extract_holistic_features(detection_result)
        frame_buffer.append(current_kpts)

        # 3. 大脑网络推理
        input_data = np.array(frame_buffer) 
        input_tensor = torch.from_numpy(input_data).float()
        # 将 (100, 75, 3) 展平为 (100, 225)，并增加 Batch 维度 -> (1, 100, 225)
        input_tensor = input_tensor.view(100, -1).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = action_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_id = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_id].item()

        # 4. 可视化绘制 (极客 UI)
        # 画点
        for idx, pt in enumerate(current_kpts):
            x, y, conf = pt
            if conf > 0.2:
                px, py = int(x * w), int(y * h)
                if idx < 33: color = (0, 255, 0)     # 身体：绿
                elif idx < 54: color = (255, 0, 0)   # 左手：蓝
                else: color = (0, 0, 255)            # 右手：红
                cv2.circle(frame, (px, py), 3, color, -1)

        # 计算并画 FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 画识别结果
        if confidence > 0.70:
            action_text = f"CMD: {action_names[predicted_id]} ({confidence*100:.1f}%)"
            text_color = (0, 255, 0) # 绿色高亮
        else:
            action_text = f"CMD: Thinking... ({confidence*100:.1f}%)"
            text_color = (0, 165, 255) # 橙色过渡状态

        cv2.putText(frame, action_text, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color, 2)

        # 显示
        cv2.imshow("Unitree Go2 VLA - Live Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    run_live_inference()