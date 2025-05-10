import numpy as np
import pywt
import math
from tqwt_tools.tqwt import tqwt, itqwt  # 导入TQWT和逆变换
import matplotlib.pyplot as plt
from scipy.signal import periodogram


# 读取PPG信号数据
def read_ppg_signal(file_path):
    """
    读取PPG信号数据
    """
    with open(file_path, 'r') as file:
        data = file.read().split()
    return np.array(data, dtype=float)


# 小波阈值去噪
def wavelet_denoise(signal, wavelet='db4', level=5, threshold_method='soft'):
    """
    对信号进行小波去噪
    """
    import pywt
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745  # 噪声标准差估计
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))  # 通用阈值
    coeffs[1:] = [pywt.threshold(c, threshold, mode=threshold_method) for c in coeffs[1:]]
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal[:len(signal)]  # 确保输出长度与输入一致


# 归一化到[-1, 1]范围
def normalize_signal(signal):
    """
    归一化信号到[-1, 1]范围
    """
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    return 2 * (signal - signal_min) / (signal_max - signal_min) - 1


# 步骤2: 计算Jmax
def calculate_Jmax(Q, N, r=3, max_Jmax=40):
    """
    计算 TQWT 的最大分解级数 Jmax

    Args:
        Q (float): Q因子
        N (int): 信号长度
        r (float): 冗余度（默认3）
        max_Jmax (int): Jmax的最大限制值（默认40）

    Returns:
        int: 最大分解级数 Jmax
    """
    beta = 2 / (Q + 1)
    alpha = 1 - beta / r
    Jmax = math.floor(np.log((beta * N) / 8) / np.log(1 / alpha))
    return min(Jmax, max_Jmax)


# 步骤3: 计算小波系数矩阵
def calculate_wavelet_coefficients(signal, Q, r=3):
    """
    使用 TQWT 计算小波系数矩阵

    Args:
        signal (np.ndarray): 输入信号（需为偶数长度）
        Q (float): Q因子
        r (float): 冗余度（默认3）

    Returns:
        np.ndarray: 小波系数矩阵，形状为 (Jmax, max_coeff_length)
    """
    # 检查输入信号合法性
    if len(signal) % 2 != 0 or signal.ndim != 1:
        raise ValueError("Input signal must be 1D and even-length")

    N = len(signal)
    Jmax = calculate_Jmax(Q, N, r)

    # 执行 TQWT 变换
    wavelet_coeffs = tqwt(signal, Q, r, Jmax)

    # 构建系数矩阵（处理不同长度子带）
    max_length = max(len(coeff) for coeff in wavelet_coeffs)
    wavelet_matrix = []

    for coeff in wavelet_coeffs:
        pad_width = max_length - len(coeff)
        padded_coeff = np.pad(coeff, (0, pad_width), mode='constant')
        wavelet_matrix.append(padded_coeff)

    return np.array(wavelet_matrix)


# 步骤4: 计算近似L0范数（基于阈值）
def calculate_approx_l0_norm(wavelet_matrix, threshold=1e-5):
    """
    计算小波系数矩阵的近似L0范数（基于阈值），并通过非零元占比衡量稀疏性

    Args:
        wavelet_matrix (np.ndarray): 小波系数矩阵
        threshold (float): 阈值，小于该值的系数视为零

    Returns:
        float: 非零元占比（稀疏性度量）
    """
    all_coeffs_flattened = wavelet_matrix.flatten()
    total_elements = len(all_coeffs_flattened)  # 矩阵总元素数
    non_zero_count = np.sum(np.abs(all_coeffs_flattened) > threshold)  # 非零元数量
    sparsity_ratio = non_zero_count / total_elements  # 非零元占比
    return sparsity_ratio


# 步骤5: 计算能量 - 熵比R(Q)
def calculate_energy_entropy_ratio(wavelet_matrix):
    """
    计算小波系数矩阵的能量 - 熵比R(Q)

    Args:
        wavelet_matrix (np.ndarray): 小波系数矩阵

    Returns:
        float: 能量 - 熵比R(Q)
    """
    R_Q = []
    for J in range(len(wavelet_matrix)):
        c = wavelet_matrix[J]
        E_j = np.sum(np.abs(c) ** 2)  # 能量
        if E_j == 0:
            H_j = 0
        else:
            ratio = np.abs(c) ** 2 / E_j
            ratio[ratio == 0] = np.finfo(float).eps  # 避免log(0)
            H_j = -np.sum(ratio * np.log2(ratio))  # 熵
        if H_j == 0:
            R_j = 0
        else:
            R_j = E_j / H_j  # 能量 - 熵比
        R_Q.append(R_j)
    return np.max(R_Q)  # 返回每一层的最大值


# SALSA算法实现
def salsa_denoise(signal, QL, QH, r=3, max_iter=50, lambda1=0.1, lambda2=0.1, tol=1e-4):
    """
    SALSA算法实现双字典稀疏分解去噪
    Args:
        signal: 输入信号
        QL: 低频Q因子
        QH: 高频Q因子
        r: 冗余度
        max_iter: 最大迭代次数
        lambda1: 低频正则化参数
        lambda2: 高频正则化参数
        tol: 收敛阈值
    Returns:
        denoised: 去噪信号
        low_comp: 低频分量
        high_comp: 高频分量
    """
    Jmax_L = calculate_Jmax(QL, len(signal), r)
    Jmax_H = calculate_Jmax(QH, len(signal), r)

    # 初始系数矩阵
    wL = [np.zeros_like(coeff) for coeff in tqwt(signal, QL, r, Jmax_L)]
    wH = [np.zeros_like(coeff) for coeff in tqwt(signal, QH, r, Jmax_H)]

    # 辅助变量和参数初始化
    mu = 1.0  # 拉格朗日乘子的更新步长
    gamma = 1.0  # 收缩参数
    zL = [np.zeros_like(coeff) for coeff in wL]
    zH = [np.zeros_like(coeff) for coeff in wH]
    uL = [np.zeros_like(coeff) for coeff in wL]
    uH = [np.zeros_like(coeff) for coeff in wH]

    prev_residual = np.inf
    for it in range(max_iter):
        # 更新低频分量
        residual = signal - itqwt(wH, QH, r, len(signal))
        wL = tqwt(residual, QL, r, Jmax_L)
        wL = [w - (1 / mu) * u for w, u in zip(wL, uL)]
        wL = [soft_thresholding(w, lambda1 / mu) for w in wL]
        zL = [w + (1 / mu) * u for w, u in zip(wL, uL)]

        # 更新高频分量
        residual = signal - itqwt(wL, QL, r, len(signal))
        wH = tqwt(residual, QH, r, Jmax_H)
        wH = [w - (1 / mu) * u for w, u in zip(wH, uH)]
        wH = [soft_thresholding(w, lambda2 / mu) for w in wH]
        zH = [w + (1 / mu) * u for w, u in zip(wH, uH)]

        # 更新拉格朗日乘子
        uL = [u + mu * (w - z) for u, w, z in zip(uL, wL, zL)]
        uH = [u + mu * (w - z) for u, w, z in zip(uH, wH, zH)]

        # 计算残差
        current_residual = np.linalg.norm(signal - itqwt(wL, QL, r, len(signal)) - itqwt(wH, QH, r, len(signal)))
        if abs(prev_residual - current_residual) < tol:
            break
        prev_residual = current_residual

    # 重构分量
    low_comp = itqwt(wL, QL, r, len(signal))
    high_comp = itqwt(wH, QH, r, len(signal))
    denoised = low_comp + high_comp
    return denoised, low_comp, high_comp


def soft_thresholding(x, threshold):
    """
    软阈值函数
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def calculate_snr(original, noisy):
    """计算信噪比(SNR)"""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - noisy) ** 2)
    return 10 * np.log10(signal_power / noise_power)


def calculate_mse(original, denoised):
    """计算均方误差(MSE)"""
    return np.mean((original - denoised) ** 2)

from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# 使用Butterworth滤波器对数据进行滤波的函数 data传入函数需要调整的参数 lowcut highcut高低截止频率 fs采样率 order滤波器阶数
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# 定义全局步长
STEP_SIZE = 0.03  # 步长


# 主函数
def process_ppg_signal(file_path):
    # [原有代码保持不变...]
    # 在原代码末尾添加以下内容
    # 读取信号
    ppg_signal = read_ppg_signal(file_path)

    # 去噪
    #denoised_signal = wavelet_denoise(ppg_signal, wavelet='db4', level=5, threshold_method='soft')

    lowcut = 0.2
    highcut = 200
    sampling_rate = 1000
    order = 3
    denoised_signal = butter_bandpass_filter(ppg_signal, lowcut=0.5, highcut=20, fs=1000, order=3)
    denoised_signal = denoised_signal + 1
    # 归一化
    normalized_signal = normalize_signal(denoised_signal)
    normalized_ppg = normalize_signal(ppg_signal) + 1
    # 参数设置
    sampling_rate = 1000  # 采样率
    N = len(normalized_signal)  # 信号长度
    Q_values = np.arange(1, 10.5, 0.5)  # Q值从1到10，步长为0.5，共19个值

    # 存储结果
    approx_l0_norms = []
    R_Q_values = []

    # 遍历 Q 值并计算小波系数矩阵
    for Q in Q_values:
        #print(f"\n当前 Q 值: {Q}")
        r = 3.0  # 冗余度
        wavelet_matrix = calculate_wavelet_coefficients(normalized_signal, Q, r)

        # 计算近似L0范数
        approx_l0_norm = calculate_approx_l0_norm(wavelet_matrix, threshold=1e-5)
        approx_l0_norms.append(approx_l0_norm)

        # 计算能量 - 熵比R(Q)
        R_Q = calculate_energy_entropy_ratio(wavelet_matrix)
        R_Q_values.append(R_Q)

    # 选择最优的Qmax（稀疏性最高，即非零元占比最小）
    Qmax = Q_values[np.argmin(approx_l0_norms)]
    print("\nQmax (稀疏性最高，非零元占比最小):", Qmax)

    # 计算步长为 STEP_SIZE 的Q值对应的R(Q)
    Q_step_values = np.arange(1, Qmax + STEP_SIZE, STEP_SIZE)  # 使用全局步长
    R_Q_values_step = []
    for Q in Q_step_values:
        wavelet_matrix = calculate_wavelet_coefficients(normalized_signal, Q)
        R_Q = calculate_energy_entropy_ratio(wavelet_matrix)
        R_Q_values_step.append(R_Q)

    # 选择低Q因子集和高Q因子集的边界
    QB_index = np.argmin(R_Q_values_step)  # R(Q)最小的Q值
    QL_index = np.argmax(R_Q_values_step[:QB_index + 1])  # QB之前的R(Q)最大值
    QH_index = np.argmax(R_Q_values_step[QB_index:]) + QB_index  # QB之后的R(Q)最大值

    # 输出Q_boundary
    QB = Q_step_values[QB_index]
    QL = Q_step_values[QL_index]
    QH = Q_step_values[QH_index]
    print("QB:", QB)
    print("QL:", QL)
    print("QH:", QH)
    print("R(Q) at QB:", R_Q_values_step[QB_index])

    # 使用SALSA算法进行去噪
    denoised, low_comp, high_comp = salsa_denoise(normalized_signal, QL, QH,
                                                  lambda1=0.1, lambda2=0.1, max_iter=50)
    denoised = denoised + 1

    # 评估指标
    snr = calculate_snr(normalized_ppg, denoised)
    mse = calculate_mse(normalized_ppg, denoised)
    print("\nDenoising effect evaluation:")
    print(f"SNR: {snr:.2f} dB")
    print(f"MSE: {mse:.6f}")

    # 绘制结果
    # 图 1：原始波形
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.title("Original Signal")
    plt.plot(normalized_ppg, label='Original', alpha=0.6)
    plt.legend()

    # 图 2：滤波后的波形和原波形重叠对比
    plt.subplot(412)
    plt.title("Comparison between original and denoised signals")
    plt.plot(normalized_ppg, label='Original', alpha=0.6)
    plt.plot(denoised, label='Denoised', linewidth=1.5)
    plt.legend()

    # 图 3：滤波后的波形
    plt.subplot(413)
    plt.title("Denoised Signal")
    plt.plot(denoised, color='darkorange')


    plt.subplot(414)
    f_orig, Pxx_orig = periodogram(normalized_signal, fs=1000)
    f_den, Pxx_den = periodogram(denoised, fs=1000)
    plt.semilogy(f_orig, Pxx_orig, label='Original')
    plt.semilogy(f_den, Pxx_den, label='Denoised')
    plt.title("Power spectral density comparison")
    plt.xlabel('Frequency [Hz]')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return denoised, low_comp, high_comp


# 运行代码
#file_path = r"C:\Users\86130\Desktop\1000Hz自测集\jqz3.15\data1.txt"
#denoised, low_comp, high_comp = process_ppg_signal(file_path)