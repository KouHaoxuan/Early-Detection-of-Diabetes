import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.metrics import (
    precision_score,  # 精确率
    recall_score,     # 召回率
    accuracy_score,   # 准确率
    confusion_matrix, # 混淆矩阵
    f1_score,         # F1 分数
    roc_curve,        # ROC 曲线
    auc               # ROC 曲线下面积
)
from sklearn.model_selection import train_test_split


data = pd.read_csv("./data/cleaned_data.csv")

# 特征和目标变量
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# xgboost模型建立
# 创建XGBoost分类器模型
model_XGB = XGBClassifier()

# 指定评估集
eval_set = [(test_x, test_y)]

# 使用early stopping来防止过拟合
model_XGB.fit(train_x, train_y, eval_set=eval_set, verbose=False)
# 对验证集进行预测
y_pred = model_XGB.predict(test_x)

# 存储模型评估指标
scores_XGB = []

# 计算精确率
scores_XGB.append(precision_score(test_y, y_pred))

# 计算召回率
scores_XGB.append(recall_score(test_y, y_pred))

# 计算准确率:即模型预测正确的样本数与总样本数之比。准确率 = (TP + TN) / (TP + FP + TN + FN)。
scores_XGB.append(accuracy_score(test_y, y_pred))

# 计算混淆矩阵
confusion_matrix_XGB = confusion_matrix(test_y, y_pred)

# 计算F1分数
f1_score_XGB = f1_score(test_y, y_pred, labels=None, pos_label=0, average="binary", sample_weight=None)

# 对验证集进行概率预测
predictions_xgb = model_XGB.predict_proba(test_x)

# 计算ROC曲线
FPR_xgb, recall_xgb, thresholds = roc_curve(test_y, predictions_xgb[:, 1], pos_label=1)

# 计算ROC曲线下面积
area_xgb = auc(FPR_xgb, recall_xgb)

# 打印XGBoost模型结果
print('xgboost_result：\n')

# 打印混淆矩阵
print(pd.DataFrame(columns=['Prediction = 0', 'Prediction = 1'], index=['Ground True = 0', 'Ground True = 1'], data=confusion_matrix_XGB))

# 打印F1值
print("f1 Score:" + str(f1_score_XGB))

# 打印精确度和召回率
print("Precision | Recall | Accuracy：" + str(scores_XGB))

pickle.dump(model_XGB, open('./models/model_xgb.pkl', 'wb'))