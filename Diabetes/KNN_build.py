import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
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

# 模型训练
model_knn = KNeighborsClassifier()
model_knn.fit(train_x, train_y)
knn_pred = model_knn.predict(test_x)

# 存储模型评估指标
scores_knn = []

# 计算精确率:即模型预测为正例的样本中真正为正例的比例。精确率 = TP / (TP + FP)
scores_knn.append(precision_score(test_y, knn_pred, pos_label=1, average='binary'))

# 计算召回率:即真正例在所有实际为正例的样本中的比例。召回率 = TP / (TP + FN)
scores_knn.append(recall_score(test_y, knn_pred, pos_label=1, average='binary'))

# 计算准确率:即模型预测正确的样本数与总样本数之比。准确率 = (TP + TN) / (TP + FP + TN + FN)。
scores_knn.append(accuracy_score(test_y, knn_pred))

# 计算混淆矩阵
confusion_matrix_knn = confusion_matrix(test_y, knn_pred)

# 计算F1分数
f1_score_knn = f1_score(test_y, knn_pred, pos_label=1, average='binary')

# 对验证集进行预测，得到每一类的概率
predictions_knn = model_knn.predict_proba(test_x) # 每一类的概率

# 计算ROC曲线
FPR_knn, recall_knn, thresholds = roc_curve(test_y, predictions_knn[:,1], pos_label=1)

# 计算ROC曲线下面积
area_knn = auc(FPR_knn, recall_knn)

print('knn_result：\n')
print(pd.DataFrame(columns=['Prediction = 0', 'Prediction = 1'], index=['Ground True = 0', 'Ground True = 1'], data=confusion_matrix_knn)) # 混淆矩阵
print("f1 Score:" + str(f1_score_knn))
print("Precision | Recall | Accuracy：" + str(scores_knn))

pickle.dump(model_knn, open('./models/model_knn.pkl', 'wb'))
