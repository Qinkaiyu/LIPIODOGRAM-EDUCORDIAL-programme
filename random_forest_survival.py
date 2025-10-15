# ------------------------------------------------------------------------------
# Random Forest 生存分析模型
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class RandomForestSurvival:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        
    def load_data(self, csv_path):
        """加载CSV数据"""
        print("正在加载数据...")
        df = pd.read_csv(csv_path)
        
        # 数据预处理
        X = df.iloc[:, 3:].values.astype(np.float32)  # 特征
        e = df.iloc[:, 1].values.astype(bool)         # 事件指示
        t = np.round(df.iloc[:, 2].values, 2).astype(np.float32)  # 时间保留两位小数
        
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
        
        return X, e, t
    
    def train(self, X, e, t, test_size=0.3):
        """训练Random Forest生存模型"""
        print("\n开始训练Random Forest生存模型...")
        
        # 划分训练测试集
        X_train, X_test, e_train, e_test, t_train, t_test = train_test_split(
            X, e, t, test_size=test_size, random_state=self.random_state, stratify=e
        )
        
        # 创建生存数据结构
        y_train = Surv.from_arrays(e_train, t_train)
        y_test = Surv.from_arrays(e_test, t_test)
        
        # 初始化模型
        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 预测
        risk_scores_train = self.model.predict(X_train)
        risk_scores_test = self.model.predict(X_test)
        
        # 计算C-index
        train_c_index = concordance_index_censored(e_train, t_train, risk_scores_train)[0]
        test_c_index = concordance_index_censored(e_test, t_test, risk_scores_test)[0]
        
        print(f"训练集 C-index: {train_c_index:.4f}")
        print(f"测试集 C-index: {test_c_index:.4f}")
        
        # 存储结果
        self.results = {
            'train_c_index': train_c_index,
            'test_c_index': test_c_index,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'risk_scores_train': risk_scores_train,
            'risk_scores_test': risk_scores_test
        }
        
        # 保存模型
        self.save_model()
        
        return test_c_index
    
    def hyperparameter_tuning(self, X, e, t):
        """超参数调优"""
        print("\n开始超参数调优...")
        
        # 划分数据
        X_train, X_test, e_train, e_test, t_train, t_test = train_test_split(
            X, e, t, test_size=0.3, random_state=self.random_state, stratify=e
        )
        
        y_train = Surv.from_arrays(e_train, t_train)
        
        # 定义参数网格
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # 网格搜索
        rf = RandomSurvivalForest(random_state=self.random_state, n_jobs=-1)
        
        best_score = -np.inf
        best_params = None
        
        # 简化的网格搜索（由于sksurv不直接支持GridSearchCV）
        for n_est in param_grid['n_estimators']:
            for max_d in param_grid['max_depth']:
                for min_split in param_grid['min_samples_split']:
                    for min_leaf in param_grid['min_samples_leaf']:
                        try:
                            model = RandomSurvivalForest(
                                n_estimators=n_est,
                                max_depth=max_d,
                                min_samples_split=min_split,
                                min_samples_leaf=min_leaf,
                                random_state=self.random_state,
                                n_jobs=-1
                            )
                            model.fit(X_train, y_train)
                            
                            # 使用OOB score作为评估指标
                            if hasattr(model, 'oob_score_') and model.oob_score_ > best_score:
                                best_score = model.oob_score_
                                best_params = {
                                    'n_estimators': n_est,
                                    'max_depth': max_d,
                                    'min_samples_split': min_split,
                                    'min_samples_leaf': min_leaf
                                }
                        except:
                            continue
        
        print(f"最佳参数: {best_params}")
        print(f"最佳得分: {best_score:.4f}")
        
        return best_params
    
    def save_model(self, save_path='logs/models/random_forest2.pkl'):
        """保存训练好的模型"""
        if self.model is None:
            print("没有训练好的模型可以保存！")
            return
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 确定模型类型
        model_type = 'survival' if hasattr(self.model, 'predict') and 'RandomSurvival' in str(type(self.model)) else 'regression'
        
        # 保存模型和相关组件
        model_data = {
            'model': self.model,
            'type': model_type,
            'results': self.results,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'random_state': self.random_state
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Random Forest模型已保存到: {save_path}")
    
    def load_model(self, load_path='logs/models/random_forest2.pkl'):
        """加载训练好的模型"""
        if not os.path.exists(load_path):
            print(f"❌ 模型文件不存在: {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_type = model_data.get('type', 'survival')
            self.results = model_data.get('results', {})
            
            # 恢复超参数
            hyperparams = model_data.get('hyperparameters', {})
            self.n_estimators = hyperparams.get('n_estimators', self.n_estimators)
            self.max_depth = hyperparams.get('max_depth', self.max_depth)
            self.random_state = hyperparams.get('random_state', self.random_state)
            
            print(f"✅ Random Forest模型已从 {load_path} 加载成功")
            print(f"   模型类型: {self.model_type}")
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
            if hasattr(self, 'model_type') and self.model_type == 'regression':
                # 回归模型：预测时间，然后转换为风险分数
                pred_times = self.model.predict(X)
                risk_scores = -pred_times
            else:
                # 生存分析模型：直接预测风险分数
                risk_scores = self.model.predict(X)
            
            return risk_scores
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    # def plot_feature_importance(self, feature_names=None):
    #     """绘制特征重要性"""
    #     if self.model is None:
    #         print("请先训练模型！")
    #         return
        
    #     importance = self.model.feature_importances_
        
    #     if feature_names is None:
    #         feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
    #     # 排序
    #     indices = np.argsort(importance)[::-1]
        
    #     plt.figure(figsize=(12, 8))
    #     plt.title("Random Forest 特征重要性")
    #     plt.bar(range(len(importance)), importance[indices])
    #     plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    #     plt.tight_layout()
    #     plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    #     plt.show()
        
    #     # 打印前10个重要特征
    #     print("\n前10个最重要的特征:")
    #     for i in range(min(10, len(importance))):
    #         idx = indices[i]
    #         print(f"{feature_names[idx]}: {importance[idx]:.4f}")

def main():
    # 数据路径
    csv_path = ""
    model_path = 'logs/models/random_forest.pkl'
    
    # 创建模型
    rf_model = RandomForestSurvival(n_estimators=350, max_depth=7, random_state=42)
    
    # 检查是否已有训练好的模型
    if os.path.exists(model_path):
        print(f"🔍 发现已训练的模型: {model_path}")
        choice = input("是否使用已有模型？(y/n): ").lower()
        
        if choice == 'y':
            if rf_model.load_model(model_path):
                print("✅ 模型加载成功，跳过训练步骤")
                # 打印模型信息
                if rf_model.results:
                    print(f"训练集 C-index: {rf_model.results.get('train_c_index', 'N/A'):.4f}")
                    print(f"测试集 C-index: {rf_model.results.get('test_c_index', 'N/A'):.4f}")
                return rf_model
            else:
                print("❌ 模型加载失败，将重新训练")
    
    print("🚀 开始训练新模型...")
    
    # 加载数据
    X, e, t = rf_model.load_data(csv_path)
    
    # 训练模型
    test_c_index = rf_model.train(X, e, t)
    
    # 超参数调优（可选，比较耗时）
    # print("\n是否进行超参数调优？这可能需要较长时间...")
    # best_params = rf_model.hyperparameter_tuning(X, e, t)
    
    # 绘制特征重要性
    #rf_model.plot_feature_importance()
    
    print(f"\n=== Random Forest 生存分析结果 ===")
    print(f"最终测试集 C-index: {test_c_index:.4f}")
    print(f"💾 模型已保存，后续可直接加载使用")
    
    return rf_model

def test_model_loading():
    """测试模型加载功能"""
    print("\n🧪 测试Random Forest模型加载功能...")
    
    # 创建新的模型实例
    test_model = RandomForestSurvival()
    
    # 加载模型
    if test_model.load_model():
        print("✅ 模型加载测试成功")
        
        # 测试预测功能（使用一些示例数据）
        csv_path = r"C:/Users/yuqinkai/Downloads/Smaple_LIP_copy.csv"
        X, e, t = test_model.load_data(csv_path)
        
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
