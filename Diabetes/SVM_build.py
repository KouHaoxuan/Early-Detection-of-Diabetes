import pandas as pd
import pickle
from sklearn.svm import SVC
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

# SVM模型建立
# 创建MinMaxScaler对象，用于特征归一化处理
min_max_scaler = preprocessing.MinMaxScaler()

# 对训练集特征进行归一化处理
# X_train_minmax = min_max_scaler.fit_transform(train_x)
X_train_minmax = train_x
# 对验证集特征进行归一化处理
# X_test_minmax = min_max_scaler.transform(test_x)
X_test_minmax = test_x
# 创建SVM模型
model_svm = SVC(probability=True)

# 使用归一化后的训练集拟合SVM模型
model_svm.fit(X_train_minmax, train_y)

# 对归一化后的验证集进行预测
svm_pred = model_svm.predict(X_test_minmax)

# 存储模型评估指标
scores_svm = []

# 计算精确率
scores_svm.append(precision_score(test_y, svm_pred))

# 计算召回率
scores_svm.append(recall_score(test_y, svm_pred))

# 计算准确率:即模型预测正确的样本数与总样本数之比。准确率 = (TP + TN) / (TP + FP + TN + FN)。
scores_svm.append(accuracy_score(test_y, svm_pred))

# 计算混淆矩阵
confusion_matrix_svm = confusion_matrix(test_y, svm_pred)

# 计算F1分数
f1_score_svm = f1_score(test_y, svm_pred, labels=None, pos_label=1, average='binary', sample_weight=None)

# 对归一化后的验证集进行概率预测
predictions_svm = model_svm.predict_proba(X_test_minmax)

# 计算ROC曲线
FPR_svm, recall_svm, thresholds = roc_curve(test_y, predictions_svm[:, 1], pos_label=1)

# 计算ROC曲线下面积
area_svm = auc(FPR_svm, recall_svm)

# 打印SVM模型结果
print('svm_result：\n')

# 打印混淆矩阵
print(pd.DataFrame(columns=['Prediction = 0', 'Prediction = 1'], index=['Ground True = 0', 'Ground True = 1'],data=confusion_matrix_svm))

# 打印F1值
print("f1 Score:" + str(f1_score_svm))

# 打印准确率和召回率
print("Precision | Recall | Accuracy：" + str(scores_svm))

pickle.dump(model_svm, open('./models/model_svm.pkl', 'wb'))