# ------------------------------------------------------------------------------
# Lasso-Cox Proportional Hazards 生存分析模型
# 关键点: 使用交叉验证寻找最优惩罚项 (penalizer)
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class CoxSurvival:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.selected_features_ = None
        self.optimal_penalizer_ = None

    def load_data(self, csv_path):
        """加载并预处理CSV数据"""
        print("正在加载数据...")
        df = pd.read_csv(csv_path)
        
        # 假设: 第2列是事件(event), 第3列是时间(duration), 第4列开始是特征
        X = df.iloc[:, 3:].copy()
        e = df.iloc[:, 1].values.astype(int)
        t = np.round(df.iloc[:, 2].values).astype(np.int32)
        
        # 处理缺失值
        if X.isnull().any().any():
            print("使用均值填充缺失值...")
            X = X.fillna(X.mean())
            
        print(f"数据形状: {X.shape}")
        print(f"事件发生数: {np.sum(e)}")
        
        # 创建生存数据DataFrame
        survival_df = X.copy()
        survival_df['duration'] = t
        survival_df['event'] = e
        
        # 过滤无效的时间值（<=0的时间）
        valid_mask = survival_df['duration'] > 0
        if not valid_mask.all():
            print(f"发现 {np.sum(~valid_mask)} 个无效时间值（<=0），将被过滤掉")
            survival_df = survival_df[valid_mask]
        
        return survival_df


    def train(self, survival_df, test_size=0.3):
        """使用最优penalizer训练Lasso-Cox模型"""
        print("\n开始训练最终的Lasso-Cox模型...")
        
        train_df, test_df = train_test_split(
            survival_df, test_size=test_size, random_state=42, stratify=survival_df['event']
        )
        
        feature_cols = [col for col in train_df.columns if col not in ['duration', 'event']]
        
        # 在训练集上fit_transform, 在测试集上transform
        train_df_scaled = train_df.copy()
        test_df_scaled = test_df.copy()
        self.scaler = StandardScaler() # 重新初始化scaler以在最终训练集上拟合
        train_df_scaled[feature_cols] = self.scaler.fit_transform(train_df[feature_cols])
        test_df_scaled[feature_cols] = self.scaler.transform(test_df[feature_cols])
        
        # 使用最优penalizer训练模型
        self.model = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
        self.model.fit(train_df_scaled, duration_col='duration', event_col='event')
        
        # 评估模型
        train_c_index = self.model.concordance_index_
        test_c_index = self.model.score(test_df_scaled, scoring_method="concordance_index")
        
        print(f"训练集 C-index: {train_c_index:.4f}")
        print(f"测试集 C-index: {test_c_index:.4f}")
        
        # 提取被选中的特征
        self.selected_features_ = self.model.params_[self.model.params_ != 0].index.tolist()
        print(len(self.selected_features_))
        
        # 保存模型
        self.save_model()
        
        return test_c_index

    def print_selected_features_summary(self):
        """打印被Lasso选中的特征及其系数"""
        if self.model is None or not self.selected_features_:
            print("模型未训练或没有特征被选中。")
            return
            
        print("\n=== Lasso-Cox模型选中的特征 ===")
        print(f"总共选中了 {len(self.selected_features_)} 个特征。")
        
        summary = self.model.summary.loc[self.selected_features_]
        print(summary[['coef', 'exp(coef)', 'p']])
        
    def plot_coefficients(self):
        """绘制选中特征的回归系数"""
        if self.model is None or not self.selected_features_:
            print("模型未训练或没有特征被选中。")
            return
            
        coefficients = self.model.params_.loc[self.selected_features_].sort_values()
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if x > 0 else 'blue' for x in coefficients]
        plt.barh(coefficients.index, coefficients.values, color=colors)
        plt.xlabel('Cox Regression Coefficient (log hazard ratio)')
        plt.title('Coefficients of Features Selected by Lasso-Cox')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('lasso_cox_coefficients.png', dpi=300)
        plt.show()
    
    def save_model(self, save_path='logs/models/lasso_cox.pkl'):
        """保存训练好的模型"""
        if self.model is None:
            print("没有训练好的模型可以保存！")
            return
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型和相关组件
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'selected_features': self.selected_features_,
            'optimal_penalizer': self.optimal_penalizer_,
            'feature_names': [col for col in self.model.params_.index if col not in ['duration', 'event']]
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Lasso-Cox模型已保存到: {save_path}")
    
    def load_model(self, load_path='logs/models/lasso_cox.pkl'):
        """加载训练好的模型"""
        if not os.path.exists(load_path):
            print(f"❌ 模型文件不存在: {load_path}")
            return False

        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.selected_features_ = model_data.get('selected_features_')
            # self.optimal_penalizer_ = model_data.get('optimal_penalizer_')
            self.feature_names = model_data.get('feature_names')  # ✅ 加这行

            print(f"✅ Lasso-Cox模型已从 {load_path} 加载成功")
            # print(f"   最优penalizer: {self.optimal_penalizer_}")
            print(f"   选中特征数: {len(self.selected_features_) if self.selected_features_ else 0}")
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False
    
    def predict(self, X):
        """使用训练好的模型进行预测"""
        if self.model is None:
            print("❌ 模型未训练或未加载！")
            return None
        
        try:
            # 如果输入是DataFrame，先提取特征
            if isinstance(X, pd.DataFrame):
                feature_cols = [col for col in X.columns if col not in ['duration', 'event']]
                X_features = X[feature_cols]
            else:
                X_features = pd.DataFrame(X)
            
            # 特征标准化
            X_scaled = self.scaler.transform(X_features)
            
            # 创建DataFrame用于预测
            feature_names = [f'feature_{i}' for i in range(X_scaled.shape[1])]
            test_df = pd.DataFrame(X_scaled, columns=feature_names)
            
            # 预测风险分数
            risk_scores = self.model.predict_partial_hazard(test_df).values.flatten()
            
            return risk_scores
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None

def main():
    csv_path = ''
    model_path = 'logs/models/lasso_cox.pkl'
    
    # 1. 创建模型实例
    cox_ph = CoxSurvival()
    
    # 检查是否已有训练好的模型
    if os.path.exists(model_path):
        print(f"🔍 发现已训练的模型: {model_path}")
        choice = input("是否使用已有模型？(y/n): ").lower()
        
        if choice == 'y':
            if cox_ph.load_model(model_path):
                print("✅ 模型加载成功，跳过训练步骤")
                # 打印模型信息
                cox_ph.print_selected_features_summary()
                return cox_ph
            else:
                print("❌ 模型加载失败，将重新训练")
    
    print("🚀 开始训练新模型...")
    
    # 2. 加载数据
    survival_df = cox_ph.load_data(csv_path)
    
    if survival_df.empty:
        print("数据加载后为空，程序终止。")
        return
        
    # 3. 使用交叉验证寻找最优penalizer
    # 如果特征非常多，这一步可能会比较慢
    # optimal_penalizer = lasso_cox.find_optimal_penalizer(survival_df, k_folds=5)
    
    # 4. 使用最优penalizer训练最终模型
    test_c_index = cox_ph.train(survival_df)
    
    # 5. 展示结果
    cox_ph.print_selected_features_summary()
    cox_ph.plot_coefficients()
    
    print("\n=== Lasso-Cox 生存分析完成 ===")
    # print(f"最优 Penalizer: {optimal_penalizer:.4f}")
    print(f"最终测试集 C-index: {test_c_index:.4f}")
    print(f"💾 模型已保存，后续可直接加载使用")
    
    return cox_ph

def test_model_loading():
    """测试模型加载功能"""
    print("\n🧪 测试Lasso-Cox模型加载功能...")
    
    # 创建新的模型实例
    test_model = CoxSurvival()
    
    # 加载模型
    if test_model.load_model():
        print("✅ 模型加载测试成功")
        
        # 测试预测功能（使用一些示例数据）
        csv_path = 'LIPIDOGRAM2004.csv'
        survival_df = test_model.load_data(csv_path)
        
        # 使用前5个样本测试预测
        feature_cols = [col for col in survival_df.columns if col not in ['duration', 'event']]
        test_X = survival_df[feature_cols].iloc[:5]
        predictions = test_model.predict(test_X)
        
        if predictions is not None:
            print(f"\n🎯 预测测试成功，前5个样本的风险分数: {predictions}")
        else:
            print("❌ 预测测试失败")
    else:
        print("❌ 模型加载测试失败")

if __name__ == '__main__':
    model_instance = main()
    
    # 可选：测试模型加载功能
    # test_model_loading()