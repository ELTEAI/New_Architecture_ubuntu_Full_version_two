import os
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================================
# 0. 全局日志系统配置 (指向新的 MediaPipe 目录)
# ==========================================
LOG_DIR = r"/home/ubuntu/New_Architecture/1D_CNN_GRU_mediapipe/train_log"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("train_mp_%Y%m%d_%H%M%S.log")
log_filepath = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ==========================================
# 1. 数据集加载类 (Dataset)
# ==========================================
class RobotGestureDataset(Dataset):
    def __init__(self, npy_dir):
        self.npy_dir = npy_dir
        self.files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
        
    def __len__(self):
        return len(self.files)

    def _get_label(self, episode_num):
        if 1 <= episode_num <= 100: return 0     # 招手（前进）
        elif 101 <= episode_num <= 150: return 1 # 完全静止
        elif 151 <= episode_num <= 200: return 2 # 推摆(退后)
        elif 201 <= episode_num <= 250: return 3 # 左横移
        elif 251 <= episode_num <= 300: return 4 # 右横移
        elif 301 <= episode_num <= 350: return 5 # 左转
        elif 351 <= episode_num <= 400: return 6 # 右转
        elif 401 <= episode_num <= 450: return 7 # 坐下
        elif 451 <= episode_num <= 500: return 8 # 起立
        elif 501 <= episode_num <= 550: return 9 # 伸懒腰
        elif 551 <= episode_num <= 600: return 10# 打滚
        elif 601 <= episode_num <= 650: return 11# 摆姿势
        elif 651 <= episode_num <= 700: return 12# 拜年
        else: return 0

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.npy_dir, file_name)
        
        episode_num = int(file_name.split('_')[1].split('.')[0])
        label_id = self._get_label(episode_num)
        
        data = np.load(file_path)
        tensor_data = torch.from_numpy(data).float()
        # 魔法在这里：(100, 75, 3) 会被自动展平为 (100, 225)
        tensor_data = tensor_data.view(tensor_data.shape[0], -1) 
        return tensor_data, label_id

# ==========================================
# 2. 大脑 2.0 架构: 高维 1D-CNN + GRU
# ==========================================
class RobotGestureClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(RobotGestureClassifier, self).__init__()
        self.cnn = nn.Sequential(
            # 💎 核心升级：输入通道从 51 改为 225 (75个点 * 3)
            nn.Conv1d(in_channels=225, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 100帧 -> 50帧
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # 50帧 -> 25帧
        )
        # GRU 接收更丰满的 256 维特征
        self.gru = nn.GRU(256, 128, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)    # [Batch, 225, 100]
        x = self.cnn(x)           # [Batch, 256, 25]
        x = x.permute(0, 2, 1)    # [Batch, 25, 256]
        gru_out, h_n = self.gru(x)
        final_state = h_n[-1]     # [Batch, 128]
        out = self.fc(final_state)
        return out

# ==========================================
# 3. 训练主循环与分层抽样
# ==========================================
def train_model():
    logger.info("==================================================")
    logger.info("🚀 机器狗 VLA 意图识别 (MediaPipe 高维版) 训练启动")
    logger.info(f"📁 本次训练日志将永久保存至: {log_filepath}")
    logger.info("==================================================")

    # 路径配置
    NPY_DIR = r"/home/ubuntu/New_Architecture/MediaPipe_Holistic/Processed_Tensors"
    MODEL_SAVE_DIR = r"/home/ubuntu/New_Architecture/1D_CNN_GRU_mediapipe/trained_model"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True) 
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_mp_gesture_model.pth")
    
    # 超参数
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔥 当前使用计算设备: {device}")

    full_dataset = RobotGestureDataset(NPY_DIR)
    
    logger.info("⏳ 正在分析数据集分布，执行 9:1 严格分层抽样...")
    all_labels = []
    for file_name in full_dataset.files:
        episode_num = int(file_name.split('_')[1].split('.')[0])
        label_id = full_dataset._get_label(episode_num)
        all_labels.append(label_id)
        
    indices = list(range(len(full_dataset)))

    train_indices, val_indices = train_test_split(
        indices, 
        test_size=0.1, 
        stratify=all_labels, 
        random_state=42 
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    logger.info(f"📦 数据加载闭环！总样本: {len(full_dataset)} | 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RobotGestureClassifier(num_classes=13).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 

    best_val_f1 = 0.0

    logger.info("🏃‍♂️ 开始冲刺训练...")
    total_start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        # --- [ 训练阶段 ] ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- [ 验证阶段 ] ---
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_dataset)
        val_acc = np.mean(np.array(all_val_preds) == np.array(all_val_labels))
        val_precision = precision_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        
        epoch_duration = time.time() - epoch_start_time

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1:3d}/{EPOCHS}] ({epoch_duration:.1f}s) | "
                f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} || "
                f"Val Loss: {epoch_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
            )

        # 保存最优模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"🌟 验证集 F1-Score 提升至 {best_val_f1:.4f}，已保存最新权重。")

    total_duration = time.time() - total_start_time
    logger.info("==================================================")
    logger.info(f"🎉 炼丹彻底完成！总耗时: {total_duration/60:.2f} 分钟")
    logger.info(f"🏆 历史最高验证集 F1-Score: {best_val_f1:.4f}")
    logger.info(f"💾 最优模型权重已安全降落至: {MODEL_SAVE_PATH}")
    logger.info("==================================================")

if __name__ == "__main__":
    train_model()