import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

# --- 中文显示修复 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ----------------------------------------------------

# --- 1. 定义文件和文件夹路径 ---
SPECTRA_FILE_PATH = r'D:\python-study\自己的论文\数据\原始-nir-data.xlsx'
CHEM_FILE_PATH = r'D:\python-study\自己的论文\数据\化学成分-data.xlsx'
# 定义模型统一保存的文件夹
MODEL_SAVE_DIR = r'D:\python-study\自己的论文\模型\PLS\乘法散射校正（MSC）'


# --- 2. 定义自定义预处理模块 (已更新为MSC) ---

class MSCPreprocessor(BaseEstimator, TransformerMixin):
    """
    一个scikit-learn兼容的转换器，用于应用乘法散射校正 (MSC)。
    该转换器会在fit阶段从训练数据中学习平均光谱，
    然后在transform阶段使用该平均光谱校正所有数据，以防止数据泄露。
    """

    def __init__(self):
        self.mean_spectrum_ = None

    def fit(self, X, y=None):
        # 从训练数据中计算并存储平均光谱
        print("MSC: Calculating mean spectrum from training data...")
        self.mean_spectrum_ = np.mean(X, axis=0)
        return self

    def transform(self, X, y=None):
        # 检查是否已经fit过（即平均光谱是否存在）
        if self.mean_spectrum_ is None:
            raise RuntimeError("MSC preprocessor has not been fitted yet.")

        print("MSC: Applying correction to data...")
        corrected_spectra = np.zeros_like(X)
        for i in range(X.shape[0]):
            spectrum = X[i, :]
            # 对每个光谱与平均光谱做一元线性回归
            fit = np.polyfit(self.mean_spectrum_, spectrum, 1)
            # 应用校正：(spectrum - intercept) / slope
            corrected_spectra[i, :] = (spectrum - fit[1]) / fit[0]

        return corrected_spectra


# --- 3. 主程序：循环构建、训练、评估和保存模型 ---

if __name__ == '__main__':
    try:
        # --- 步骤 1: 一次性加载所有数据 ---
        print("开始加载光谱和化学数据...")
        spectra_df = pd.read_excel(SPECTRA_FILE_PATH)
        X_raw = spectra_df.iloc[:, 5:].values

        chem_df = pd.read_excel(CHEM_FILE_PATH)
        # 获取从第2列开始的所有化学成分的名称（列标题）
        chem_names = chem_df.columns[1:14]
        print(f"数据加载成功。光谱点数: {X_raw.shape[1]}。将要处理的化学成分数量: {len(chem_names)}")
        print("待处理列表:", list(chem_names))

        # --- 步骤 2: 循环处理每一个化学成分 ---
        for chem_name in chem_names:
            print(f"\n{'=' * 20} 正在处理: {chem_name} {'=' * 20}")

            # 提取当前化学成分的y值
            y = chem_df[chem_name].values

            if X_raw.shape[0] != y.shape[0]:
                print(f"警告: {chem_name} 的样本数 ({y.shape[0]}) 与光谱样本数 ({X_raw.shape[0]}) 不匹配，跳过此成分。")
                continue

            # 划分当前成分的训练集和测试集
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_raw, y, test_size=0.3, random_state=42
            )

            # 构建模型流水线 (使用新的MSC预处理器)
            pipeline = Pipeline([
                ('preprocessor', MSCPreprocessor()),
                ('regressor', PLSRegression(n_components=15))
            ])

            # 训练流水线
            print(f"开始为 {chem_name} 训练模型...")
            pipeline.fit(X_train_raw, y_train)

            # 在测试集上进行预测
            y_pred = pipeline.predict(X_test_raw)

            # 评估模型性能
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            print(f"--- {chem_name} 模型性能评估 ---")
            print(f"决定系数 (R²): {r2:.4f}")
            print(f"均方根误差 (RMSE): {rmse:.4f}")
            print("----------------------------------")

            # 结果可视化
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', label='Prediction samples')
            lims = [np.min([y_test, y_pred]), np.max([y_test, y_pred])]
            plt.plot(lims, lims, 'r--', linewidth=2, label='Prefect predictions (y=x)')
            # 动态设置图表标题 (更新为MSC)
            plt.title(f' ', fontsize=16)
            plt.xlabel('true values', fontsize=12)
            plt.ylabel('predicted values', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)

            # --- 已修改：采用更稳健的方式定位文本框 ---
            data_range = lims[1] - lims[0]
            text_x = lims[0] + data_range * 0.05  # 从左边框偏移5%
            text_y = lims[1] - data_range * 0.05  # 从上边框偏移5%
            plt.text(text_x, text_y, f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}',
                     fontsize=14, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            plt.show()

            # 保存整个训练好的流水线
            # 动态生成模型文件名 (更新为MSC)
            model_filename = f"{chem_name}_MSC_PLS.joblib"
            model_save_path = os.path.join(MODEL_SAVE_DIR, model_filename)

            if not os.path.exists(MODEL_SAVE_DIR):
                os.makedirs(MODEL_SAVE_DIR)

            joblib.dump(pipeline, model_save_path)
            print(f"模型已成功保存至: {model_save_path}")

        print("\n所有化学成分处理完毕！")

    except FileNotFoundError as e:
        print(f"错误: 找不到文件。请检查文件路径是否正确。\n详细信息: {e}")
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生了一个意外错误: {e}")

