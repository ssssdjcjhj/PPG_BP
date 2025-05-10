import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error,mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 归一化函数
def normalize_dataframe(df):
    # 这里简单实现为Min-Max归一化
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# 逆归一化函数
def unnormalize(value, min_value, max_value):
    return value * (max_value - min_value) + min_value


# 指定的列名，包括SBP和DBP列
columns_of_interest = [
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
    'sd1/sd2',
    'SBP',
    'DBP'
]

# 文件路径
data_file_path = r"C:\Users\86130\Desktop\24_data_new - 副本.xlsx"
output_directory = (r"C:\Users\86130\Desktop\111\2\2")
output_file_name = "blood_pressure_results.txt"
output_path = os.path.join(output_directory, output_file_name)

# 创建输出目录
os.makedirs(output_directory, exist_ok=True)

# 准备用于保存结果的文件
df = pd.read_excel(data_file_path, usecols=columns_of_interest)
df_normalized = normalize_dataframe(df)

min_max_values = {}
for feature_name in df.columns:
    max_value = df[feature_name].max()
    min_value = df[feature_name].min()
    min_max_values[feature_name] = (min_value, max_value)

with open(output_path, 'w') as output_file:
    for index, row in df_normalized.iterrows():
        sbp_label = unnormalize(row['SBP'], min_max_values['SBP'][0], min_max_values['SBP'][1])
        dbp_label = unnormalize(row['DBP'], min_max_values['DBP'][0], min_max_values['DBP'][1])
        output_file.write(f"Subject: {index}, Data: {row.drop(['SBP', 'DBP']).to_dict()}, SBP Label: {sbp_label}, DBP Label: {dbp_label}\n")

print(f"Results saved to {output_path}")

# 输入网络相关列与血压的相关性 斯皮尔曼相关系数分析 但斯皮尔曼相关系数不能捕捉模型中复杂的非线性关系
df = pd.read_excel(data_file_path, usecols=columns_of_interest)
df_normalized = normalize_dataframe(df)

# 计算斯皮尔曼相关性系数
spearman_correlation = df_normalized.corr(method='spearman')

# 设置图形
sns.set(style="white")

# 绘制热力图
plt.figure(figsize=(14, 10))
sns.heatmap(spearman_correlation, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=.5, cbar_kws={"shrink":.5})
plt.title('Spearman Correlation Matrix of Blood Pressure Data', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.show()

from sklearn.ensemble import RandomForestRegressor

# 分离特征和目标变量
X = df_normalized.drop(['SBP', 'DBP'], axis=1)
y_sbp = df_normalized['SBP']
y_dbp = df_normalized['DBP']

# 数据集分割
X_train_sbp, X_test_sbp, y_train_sbp, y_test_sbp = train_test_split(X, y_sbp, test_size=0.2, random_state=42)
X_train_dbp, X_test_dbp, y_train_dbp, y_test_dbp = train_test_split(X, y_dbp, test_size=0.2, random_state=42)

# 随机森林模型
model_sbp = RandomForestRegressor(n_estimators=100, random_state=42)
model_sbp.fit(X_train_sbp, y_train_sbp)

model_dbp = RandomForestRegressor(n_estimators=100, random_state=42)
model_dbp.fit(X_train_dbp, y_train_dbp)

# 获取特征重要性并排序
feature_importances_sbp = pd.Series(model_sbp.feature_importances_, index=X_train_sbp.columns).sort_values(ascending=False)
feature_importances_dbp = pd.Series(model_dbp.feature_importances_, index=X_train_dbp.columns).sort_values(ascending=False)

print("SBP特征重要性数据类型:", feature_importances_sbp.dtype)
print("SBP特征重要性取值范围: 最小值 =", feature_importances_sbp.min(), "最大值 =", feature_importances_sbp.max())
print("DBP特征重要性数据类型:", feature_importances_dbp.dtype)
print("DBP特征重要性取值范围: 最小值 =", feature_importances_dbp.min(), "最大值 =", feature_importances_dbp.max())

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances_sbp, y=feature_importances_sbp.index, palette="viridis")
plt.title('Feature Importances for SBP')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances_dbp, y=feature_importances_dbp.index, palette="viridis")
plt.title('Feature Importances for DBP')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_file_path, columns_of_interest):
        self.data = pd.read_excel(data_file_path, usecols=columns_of_interest)
        self.data = normalize_dataframe(self.data)  # 确保数据在加载时就进行归一化
        self.features = self.data.drop(['SBP', 'DBP'], axis=1).values
        self.sbp_labels = self.data['SBP'].values
        self.dbp_labels = self.data['DBP'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        sbp_label = torch.tensor(self.sbp_labels[idx], dtype=torch.float32)
        dbp_label = torch.tensor(self.dbp_labels[idx], dtype=torch.float32)
        return feature, sbp_label, dbp_label


# 定义网络模型类
# 定义网络模型类（修改后）
class NumericNetwork(nn.Module):
    def __init__(self, numeric_features):
        super(NumericNetwork, self).__init__()
        # 加深网络结构：输入层 → 256 → 128 → 64 → 32
        self.fc1 = nn.Linear(numeric_features, 256)
        self.bn1 = nn.BatchNorm1d(256)  # 批归一化
        self.drop1 = nn.Dropout(0.3)    # Dropout层
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(64, 32)
        # 输出分支
        self.fc5_sbp = nn.Linear(32, 1)  # SBP输出层
        self.fc5_dbp = nn.Linear(32, 1)  # DBP输出层
        self.act = nn.LeakyReLU(0.1)     # 替换ReLU为LeakyReLU

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.act(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        x = self.act(self.fc4(x))
        output_sbp = self.fc5_sbp(x)
        output_dbp = self.fc5_dbp(x)
        return {'output_sbp': output_sbp, 'output_dbp': output_dbp}


def visualize_layer_outputs(layer_outputs):
    pass  # 这里可以实现具体的可视化函数


def print_gradients(net):
    pass  # 这里可以实现打印梯度的函数


# 数据加载和K折交叉验证
all_folds_loss = {}
dataset = CustomDataset(data_file_path, columns_of_interest)
kf = KFold(n_splits=10)

# K折交叉
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"Starting fold {fold + 1}")
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    # 设置 drop_last=True
    train_loader = DataLoader(train_subset, batch_size=20, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=20, shuffle=False, drop_last=True)

    # 后续训练代码保持不变
    model = NumericNetwork(numeric_features=len(columns_of_interest) - 2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 1000
    max_grad_norm = 0.01  # 设置梯度裁剪的最大范数 防止梯度爆炸
    fold_loss = []  # 初始化存储每个epoch损失的列表

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0  # 损失函数计数器
        for batch_idx, (numeric_data, sbp_label, dbp_label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            outputs = model(numeric_data)
            prediction_sbp = outputs['output_sbp']  # 从模型输出字典中提取最终预测结果
            prediction_dbp = outputs['output_dbp']  # 从模型输出字典中提取最终预测结果
            loss_sbp = criterion(prediction_sbp.squeeze(), sbp_label)
            loss_dbp = criterion(prediction_dbp.squeeze(), dbp_label)
            loss = loss_sbp + loss_dbp
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            average_epoch_loss = epoch_loss / len(train_loader)
            fold_loss.append(average_epoch_loss)
            all_folds_loss[fold + 1] = fold_loss
            print(f"Fold {fold + 1}, Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item()}")
            print("Input data:", numeric_data)
            print("SBP Labels:", sbp_label)
            print("DBP Labels:", dbp_label)
        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Average Epoch Loss: {epoch_loss / len(train_loader)}")

    torch.save(model.state_dict(), os.path.join(output_directory, f'model_weights_fold_{fold + 1}.pth'))

for fold, losses in all_folds_loss.items():
    plt.plot(losses, label=f'Fold {fold}')

plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.title('Loss During Training Across Folds')
plt.legend()
plt.show()

# 模型评估
all_sbp_predictions = []
all_dbp_predictions = []
all_sbp_actuals = []
all_dbp_actuals = []

# 假设这里划分了测试集，你也可以使用剩余未参与训练和验证的数据
_, test_subset = train_test_split(dataset, test_size=0.2, random_state=42)
test_loader = DataLoader(test_subset, batch_size=20, shuffle=False)

# 收集所有实际值
for _, sbp_label, dbp_label in test_loader:
    all_sbp_actuals.extend(sbp_label.tolist())
    all_dbp_actuals.extend(dbp_label.tolist())

# 加载每个折的模型并进行预测
for fold in range(1, 11):
    model = NumericNetwork(numeric_features=len(columns_of_interest) - 2)
    model.load_state_dict(torch.load(os.path.join(output_directory, f'model_weights_fold_{fold}.pth')))
    model.eval()

    fold_sbp_predictions = []
    fold_dbp_predictions = []
    with torch.no_grad():
        for numeric_data, _, _ in test_loader:
            outputs = model(numeric_data)
            fold_sbp_predictions.extend(outputs['output_sbp'].squeeze().tolist())
            fold_dbp_predictions.extend(outputs['output_dbp'].squeeze().tolist())

    all_sbp_predictions.append(fold_sbp_predictions)
    all_dbp_predictions.append(fold_dbp_predictions)

# 计算平均预测值
avg_sbp_predictions = np.mean(all_sbp_predictions, axis=0)
avg_dbp_predictions = np.mean(all_dbp_predictions, axis=0)

# 计算SBP和DBP的评价指标
r2_sbp = r2_score(all_sbp_actuals, avg_sbp_predictions)
r2_dbp = r2_score(all_dbp_actuals, avg_dbp_predictions)

mae_sbp = mean_absolute_error(all_sbp_actuals, avg_sbp_predictions)
mae_dbp = mean_absolute_error(all_dbp_actuals, avg_dbp_predictions)

mse_sbp = mean_squared_error(all_sbp_actuals, avg_sbp_predictions)
mse_dbp = mean_squared_error(all_dbp_actuals, avg_dbp_predictions)

rmse_sbp = np.sqrt(mse_sbp)
rmse_dbp = np.sqrt(mse_dbp)

mape_sbp = mean_absolute_percentage_error(all_sbp_actuals, avg_sbp_predictions)
mape_dbp = mean_absolute_percentage_error(all_dbp_actuals, avg_dbp_predictions)

print(f"SBP MAE: {mae_sbp}")
print(f"DBP MAE: {mae_dbp}")
print(f"SBP MSE: {mse_sbp}")
print(f"DBP MSE: {mse_dbp}")
print(f"SBP RMSE: {rmse_sbp}")
print(f"DBP RMSE: {rmse_dbp}")
print(f"SBP R²: {r2_sbp}")
print(f"DBP R²: {r2_dbp}")
print(f"SBP MAPE: {mape_sbp}")
print(f"DBP MAPE: {mape_dbp}")