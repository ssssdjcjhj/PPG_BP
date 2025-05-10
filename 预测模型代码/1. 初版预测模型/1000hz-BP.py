import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 定义归一化函数（需与训练时一致）
def normalize_dataframe(df, min_max_values):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name in min_max_values:
            min_val, max_val = min_max_values[feature_name]
            result[feature_name] = (df[feature_name] - min_val) / (max_val - min_val)
    return result


# 定义数据集类（需与训练时一致）
class PredictionDataset(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.values.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32)


# 定义网络模型类（需与训练时结构完全一致）
class NumericNetwork(torch.nn.Module):
    def __init__(self, numeric_features=24):  # 24个特征（26列减去2个血压列）
        super().__init__()
        self.fc1 = torch.nn.Linear(numeric_features, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.drop1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.drop2 = torch.nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(128, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.drop3 = torch.nn.Dropout(0.1)
        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5_sbp = torch.nn.Linear(32, 1)
        self.fc5_dbp = torch.nn.Linear(32, 1)
        self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.act(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        x = self.act(self.fc4(x))
        return {'output_sbp': self.fc5_sbp(x),
                'output_dbp': self.fc5_dbp(x)}


# 配置参数
train_data_path = r"C:\Users\86130\Desktop\24_data_new - 副本.xlsx"  # 原始训练数据路径
test_data_path = r"C:\Users\86130\Desktop\1000hzPPG_Data_3.14.xlsx"  # 测试数据路径
model_dir = r"C:\Users\86130\Desktop\111\2"  # 模型保存目录
output_path = r"C:\Users\86130\Desktop\111\2\预测结果.xlsx"  # 结果保存路径

# 定义特征列（需与训练时完全一致）
feature_columns = [
    'Average Heartbeat Interval',
    'Heartbeat Period Variability',
    'R1 AC/DC Component Ratio',
    'Dicrotic Notch Component to DC Component Ratio',
    'Average Area of PPG Waveform Bounded by Rectangle',
    'Average Area Under PPG Waveform Curve',
    'Average Area Under PPG Waveform Between Dicrotic Notch and First Trough',
    'Average Area of PPG Waveform Rising Slope',
    'PPG Pulse Wave Dynamics Parameters',
    'Average DC Component',
    'R3',
    'R4',
    'R5',
    'R6',
    'sdnn',
    'sdsd',
    'rmssd',
    'pnn20',
    'pnn50',
    'hr_mad',
    'sd1',
    'sd2',
    's',
    'sd1/sd2'
]

# 从原始训练数据获取归一化参数
train_df = pd.read_excel(train_data_path, usecols=feature_columns + ['SBP', 'DBP'])
min_max_values = {col: (train_df[col].min(), train_df[col].max()) for col in train_df.columns}

# 加载并预处理测试数据
test_df = pd.read_excel(test_data_path)
test_df = test_df[feature_columns]  # 确保列顺序正确
test_df_norm = normalize_dataframe(test_df, min_max_values)

# 创建预测数据集
dataset = PredictionDataset(test_df_norm)
dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

# 初始化存储预测结果的容器
all_sbp_preds = []
all_dbp_preds = []

# 加载所有折的模型进行预测
for fold in range(1, 11):
    model_path = os.path.join(model_dir, f"model_weights_fold_{fold}.pth")

    # 初始化模型并加载权重
    model = NumericNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    fold_sbp = []
    fold_dbp = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch)
            sbp_value = outputs['output_sbp'].squeeze().tolist()
            dbp_value = outputs['output_dbp'].squeeze().tolist()
            if isinstance(sbp_value, (int, float)):
                fold_sbp.append(sbp_value)
            else:
                fold_sbp.extend(sbp_value)
            if isinstance(dbp_value, (int, float)):
                fold_dbp.append(dbp_value)
            else:
                fold_dbp.extend(dbp_value)

    all_sbp_preds.append(fold_sbp)
    all_dbp_preds.append(fold_dbp)

# 计算平均值并反归一化
sbp_preds = np.mean(all_sbp_preds, axis=0)
dbp_preds = np.mean(all_dbp_preds, axis=0)


# 反归一化函数
def denormalize(values, col_name):
    min_val, max_val = min_max_values[col_name]
    return values * (max_val - min_val) + min_val


# 转换为实际血压值
final_sbp = denormalize(sbp_preds, 'SBP')
final_dbp = denormalize(dbp_preds, 'DBP')

# 保存结果
result_df = pd.DataFrame({
    '预测收缩压(SBP)': final_sbp,
    '预测舒张压(DBP)': final_dbp
})
result_df.to_excel(output_path, index=False)

print(f"预测完成，结果已保存至：{output_path}")
print("示例预测结果：")
print(result_df.head())