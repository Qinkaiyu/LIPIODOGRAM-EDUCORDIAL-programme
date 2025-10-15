# ------------------------------------------------------------------------------
# S-SVM (Support Vector Machine for Survival) 生存分析模型
# 基于Van Belle et al. (2011) "Support vector methods for survival analysis: a comparison between ranking and regression approaches"
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from scipy.optimize import minimize
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class SSVMSurvival:
    def __init__(self, C=1.0, gamma='scale', kernel='rbf', epsilon=0.1, random_state=42):
        """
        S-SVM生存分析模型
        
        参数:
        C: 正则化参数
        gamma: RBF核参数
        kernel: 核函数类型 ('rbf', 'linear', 'poly', 'sigmoid')
        epsilon: SVR的epsilon参数
        random_state: 随机种子
        """
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.epsilon = epsilon
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def load_data(self, csv_path):
        """加载CSV数据"""
        print("正在加载数据...")
        df = pd.read_csv(csv_path)
        
        # 数据预处理
        X = df.iloc[:, 3:].values.astype(np.float32)  # 特征
        e = df.iloc[:, 1].values.astype(bool)         # 事件指示
        # t = np.round(df.iloc[:, 2].values).astype(np.int32)
        t = np.round(df.iloc[:, 2].values, 2).astype(np.float32)  # 时间保留两位小数
        
        # 获取特征名
        feature_names = df.columns[3:].tolist()
        
        # # 过滤无效的时间值（<=0的时间）
        # valid_mask = t > 0
        # if not np.all(valid_mask):
        #     print(f"发现 {np.sum(~valid_mask)} 个无效时间值（<=0），将被过滤掉")
        #     X = X[valid_mask]
        #     e = e[valid_mask]
        #     t = t[valid_mask]
        
        # 处理缺失值
        if np.any(np.isnan(X)):
            print("处理缺失值...")
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
        
        print(f"数据形状: {X.shape}")
        print(f"事件发生数: {np.sum(e)}")
        print(f"删失数: {np.sum(~e)}")
        print(f"时间范围: {np.min(t):.2f} - {np.max(t):.2f}")
        
        return X, e, t, feature_names
    
    def _create_survival_target(self, t, e):
        """
        创建S-SVM的目标变量
        对于删失样本，我们使用一种特殊的编码方式
        """
        # 方法1: 直接使用观察时间，但对删失样本进行调整
        target = t.copy()
        
        # 对删失样本，我们假设真实生存时间大于观察时间
        # 这里使用一个启发式方法：将删失样本的目标时间设为观察时间的1.5倍
        censored_mask = ~e
        if np.any(censored_mask):
            max_observed_time = np.max(t[e])  # 最大的事件发生时间
            target[censored_mask] = np.maximum(
                t[censored_mask] * 1.2,  # 至少是观察时间的1.2倍
                t[censored_mask] + 0.5   # 或者观察时间加0.5年
            )
            # 但不能超过最大观察时间的2倍
            target[censored_mask] = np.minimum(target[censored_mask], max_observed_time * 2)
        
        return target
    
    def _concordance_index(self, risk_scores, t, e):
        """计算C-index"""
        n = len(risk_scores)
        concordant = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i+1, n):
                # 只考虑可比较的对
                if e[i] and t[i] <= t[j]:
                    # i发生事件且时间较早，应该有更高的风险分数
                    total_pairs += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                elif e[j] and t[j] <= t[i]:
                    # j发生事件且时间较早，应该有更高的风险分数
                    total_pairs += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
        
        if total_pairs == 0:
            return 0.5
        
        return concordant / total_pairs
    
    def feature_selection(self, X, t, k_features=20):
        """特征选择"""
        if X.shape[1] <= k_features:
            print(f"特征数量({X.shape[1]})小于等于目标数量({k_features})，跳过特征选择")
            return X, list(range(X.shape[1]))
        
        print(f"\n进行特征选择，选择前{k_features}个最重要的特征...")
        
        # 使用生存时间作为目标进行特征选择
        self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = self.feature_selector.fit_transform(X, t)
        
        # 获取选择的特征索引
        selected_indices = self.feature_selector.get_support(indices=True)
        
        print(f"选择的特征索引: {selected_indices}")
        
        return X_selected, selected_indices
    
    def train(self, X, e, t, test_size=0.3, feature_selection=True, k_features=20):
        """训练S-SVM生存模型"""
        print("\n开始训练S-SVM生存模型...")
        
        # 特征选择
        if feature_selection:
            X, selected_indices = self.feature_selection(X, t, k_features)
        else:
            selected_indices = list(range(X.shape[1]))
        
        # 创建生存目标变量
        survival_target = self._create_survival_target(t, e)
        
        # 划分训练测试集
        X_train, X_test, e_train, e_test, t_train, t_test, target_train, target_test = train_test_split(
            X, e, t, survival_target, test_size=test_size, random_state=self.random_state, stratify=e
        )
        
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练SVR模型
        self.model = SVR(
            C=self.C,
            gamma=self.gamma,
            kernel=self.kernel,
            epsilon=self.epsilon
        )
        
        print(f"使用参数: C={self.C}, gamma={self.gamma}, kernel={self.kernel}, epsilon={self.epsilon}")
        
        # 拟合模型
        self.model.fit(X_train_scaled, target_train)
        
        # 预测生存时间
        pred_train = self.model.predict(X_train_scaled)
        pred_test = self.model.predict(X_test_scaled)
        
        # 将预测的生存时间转换为风险分数（生存时间越短，风险越高）
        risk_scores_train = -pred_train  # 负号：时间越短风险越高
        risk_scores_test = -pred_test
        
        # 计算C-index
        train_c_index = self._concordance_index(risk_scores_train, t_train, e_train)
        test_c_index = self._concordance_index(risk_scores_test, t_test, e_test)
        
        # 计算回归性能指标
        train_mse = mean_squared_error(target_train, pred_train)
        test_mse = mean_squared_error(target_test, pred_test)
        
        print(f"训练集 - C-index: {train_c_index:.4f}, MSE: {train_mse:.4f}")
        print(f"测试集 - C-index: {test_c_index:.4f}, MSE: {test_mse:.4f}")
        
        # 存储结果
        self.results = {
            'train_c_index': train_c_index,
            'test_c_index': test_c_index,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'e_train': e_train,
            'e_test': e_test,
            't_train': t_train,
            't_test': t_test,
            'selected_indices': selected_indices,
            'pred_train': pred_train,
            'pred_test': pred_test,
            'risk_scores_train': risk_scores_train,
            'risk_scores_test': risk_scores_test
        }
        
        # 保存模型
        self.save_model()
        
        return test_c_index
    
    def hyperparameter_tuning(self, X, e, t, cv_folds=3):
        """超参数调优"""
        print("\n开始超参数调优...")
        
        # 特征选择和标准化
        if X.shape[1] > 20:
            X, _ = self.feature_selection(X, t, 20)
        
        survival_target = self._create_survival_target(t, e)
        X_scaled = self.scaler.fit_transform(X)
        
        # 定义参数网格
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'kernel': ['rbf', 'linear']
        }
        
        best_score = -np.inf
        best_params = None
        
        # 简化的网格搜索
        print("正在搜索最佳参数组合...")
        total_combinations = len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['epsilon']) * len(param_grid['kernel'])
        current_combination = 0
        
        for C in param_grid['C']:
            for gamma in param_grid['gamma']:
                for epsilon in param_grid['epsilon']:
                    for kernel in param_grid['kernel']:
                        current_combination += 1
                        print(f"进度: {current_combination}/{total_combinations} - 测试参数: C={C}, gamma={gamma}, epsilon={epsilon}, kernel={kernel}")
                        
                        try:
                            model = SVR(C=C, gamma=gamma, epsilon=epsilon, kernel=kernel)
                            model.fit(X_scaled, survival_target)
                            
                            # 预测并计算C-index
                            pred = model.predict(X_scaled)
                            risk_scores = -pred
                            c_index = self._concordance_index(risk_scores, t, e)
                            
                            if c_index > best_score:
                                best_score = c_index
                                best_params = {
                                    'C': C, 
                                    'gamma': gamma, 
                                    'epsilon': epsilon, 
                                    'kernel': kernel
                                }
                                print(f"  ✓ 新的最佳参数! C-index: {c_index:.4f}")
                            else:
                                print(f"  C-index: {c_index:.4f}")
                                
                        except Exception as e:
                            print(f"  ✗ 参数组合失败: {e}")
                            continue
        
        print(f"\n最佳参数: {best_params}")
        print(f"最佳 C-index: {best_score:.4f}")
        
        return best_params
    
    def plot_predictions(self):
        """绘制预测结果"""
        if self.model is None:
            print("请先训练模型！")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 预测时间 vs 实际时间
        axes[0, 0].scatter(self.results['t_train'], self.results['pred_train'], 
                          alpha=0.6, label='训练集', color='blue')
        axes[0, 0].scatter(self.results['t_test'], self.results['pred_test'], 
                          alpha=0.6, label='测试集', color='red')
        
        # 添加对角线
        min_time = min(np.min(self.results['t_train']), np.min(self.results['t_test']))
        max_time = max(np.max(self.results['pred_train']), np.max(self.results['pred_test']))
        axes[0, 0].plot([min_time, max_time], [min_time, max_time], 'k--', alpha=0.7)
        
        axes[0, 0].set_xlabel('实际生存时间')
        axes[0, 0].set_ylabel('预测生存时间')
        axes[0, 0].set_title('预测时间 vs 实际时间')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 风险分数分布
        axes[0, 1].hist(self.results['risk_scores_train'], bins=30, alpha=0.7, 
                       label='训练集', color='blue', density=True)
        axes[0, 1].hist(self.results['risk_scores_test'], bins=30, alpha=0.7, 
                       label='测试集', color='red', density=True)
        axes[0, 1].set_xlabel('风险分数')
        axes[0, 1].set_ylabel('密度')
        axes[0, 1].set_title('风险分数分布')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 事件发生 vs 风险分数
        event_mask = self.results['e_test']
        censored_mask = ~self.results['e_test']
        
        axes[1, 0].scatter(self.results['risk_scores_test'][event_mask], 
                          self.results['t_test'][event_mask],
                          c='red', alpha=0.6, label='事件发生', marker='o')
        axes[1, 0].scatter(self.results['risk_scores_test'][censored_mask], 
                          self.results['t_test'][censored_mask],
                          c='blue', alpha=0.6, label='删失', marker='^')
        axes[1, 0].set_xlabel('风险分数')
        axes[1, 0].set_ylabel('观察时间')
        axes[1, 0].set_title('风险分数 vs 观察时间')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 残差分析
        residuals_train = self.results['pred_train'] - self.results['t_train']
        residuals_test = self.results['pred_test'] - self.results['t_test']
        
        axes[1, 1].scatter(self.results['pred_train'], residuals_train, 
                          alpha=0.6, label='训练集', color='blue')
        axes[1, 1].scatter(self.results['pred_test'], residuals_test, 
                          alpha=0.6, label='测试集', color='red')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('预测值')
        axes[1, 1].set_ylabel('残差')
        axes[1, 1].set_title('残差分析')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('s_svm_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, save_path='logs/models/s_svm.pkl'):
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
            'feature_selector': self.feature_selector,
            'results': self.results,
            'hyperparameters': {
                'C': self.C,
                'gamma': self.gamma,
                'kernel': self.kernel,
                'epsilon': self.epsilon,
                'random_state': self.random_state
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 S-SVM模型已保存到: {save_path}")
    
    def load_model(self, load_path='logs/models/s_svm.pkl'):
        """加载训练好的模型"""
        if not os.path.exists(load_path):
            print(f"❌ 模型文件不存在: {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data.get('feature_selector')
            self.results = model_data.get('results', {})
            
            # 恢复超参数
            hyperparams = model_data.get('hyperparameters', {})
            self.C = hyperparams.get('C', self.C)
            self.gamma = hyperparams.get('gamma', self.gamma)
            self.kernel = hyperparams.get('kernel', self.kernel)
            self.epsilon = hyperparams.get('epsilon', self.epsilon)
            self.random_state = hyperparams.get('random_state', self.random_state)
            
            print(f"✅ S-SVM模型已从 {load_path} 加载成功")
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
            # 特征选择（如果之前进行过）
            if self.feature_selector is not None:
                X_selected = self.feature_selector.transform(X)
            else:
                X_selected = X
            
            # 特征标准化
            X_scaled = self.scaler.transform(X_selected)
            
            # 预测生存时间
            pred_times = self.model.predict(X_scaled)
            
            # 转换为风险分数
            risk_scores = -pred_times
            
            return risk_scores
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def print_model_summary(self):
        """打印模型摘要"""
        if self.model is None:
            print("请先训练模型！")
            return
        
        print("\n" + "="*50)
        print("S-SVM 模型摘要")
        print("="*50)
        print(f"核函数: {self.model.kernel}")
        print(f"C参数: {self.model.C}")
        print(f"Gamma参数: {self.model.gamma}")
        print(f"Epsilon参数: {self.model.epsilon}")
        print(f"支持向量数量: {len(self.model.support_vectors_)}")
        print(f"支持向量比例: {len(self.model.support_vectors_) / len(self.results['X_train']):.2%}")
        
        print(f"\n性能指标:")
        print(f"训练集 C-index: {self.results['train_c_index']:.4f}")
        print(f"测试集 C-index: {self.results['test_c_index']:.4f}")
        print(f"训练集 MSE: {self.results['train_mse']:.4f}")
        print(f"测试集 MSE: {self.results['test_mse']:.4f}")

def main():
    # 数据路径
    csv_path = ''
    model_path = 'logs/"models/s_svm.pkl'
    
    # 创建模型
    ssvm_model = SSVMSurvival(C=10.0, gamma='scale', kernel='rbf', epsilon=0.1, random_state=42)
    
    # 检查是否已有训练好的模型
    if os.path.exists(model_path):
        print(f"🔍 发现已训练的模型: {model_path}")
        choice = input("是否使用已有模型？(y/n): ").lower()
        
        if choice == 'y':
            if ssvm_model.load_model(model_path):
                print("✅ 模型加载成功，跳过训练步骤")
                ssvm_model.print_model_summary()
                return ssvm_model
            else:
                print("❌ 模型加载失败，将重新训练")
    
    print("🚀 开始训练新模型...")
    
    # 加载数据
    X, e, t, feature_names = ssvm_model.load_data(csv_path)
    
    # 训练模型
    test_c_index = ssvm_model.train(X, e, t, feature_selection=True, k_features=20)
    
    # 打印模型摘要
    ssvm_model.print_model_summary()
    
    # 超参数调优（可选，比较耗时）
    print("\n是否进行超参数调优？这可能需要较长时间...")
    # best_params = ssvm_model.hyperparameter_tuning(X, e, t)
    
    # 绘制分析图
    ssvm_model.plot_predictions()
    
    print(f"\n=== S-SVM 生存分析结果 ===")
    print(f"最终测试集 C-index: {test_c_index:.4f}")
    print(f"💾 模型已保存，后续可直接加载使用")
    
    return ssvm_model

def test_model_loading():
    """测试模型加载功能"""
    print("\n🧪 测试模型加载功能...")
    
    # 创建新的模型实例
    test_model = SSVMSurvival()
    
    # 加载模型
    if test_model.load_model():
        print("✅ 模型加载测试成功")
        test_model.print_model_summary()
        
        # 测试预测功能（使用一些示例数据）
        csv_path = r""
        X, e, t, _ = test_model.load_data(csv_path)
        
        # 使用前5个样本测试预测
        test_X = X[:5]
        predictions = test_model.predict(test_X)
        
        if predictions is not None:
            print(f"\n🎯 预测测试成功，前5个样本的风险分数: {predictions}")
        else:
            print("❌ 预测测试失败")
    else:
        print("❌ 模型加载测试失败")

if __name__ == '__main__':
    model = main()
    
    # 可选：测试模型加载功能
    # test_model_loading()
