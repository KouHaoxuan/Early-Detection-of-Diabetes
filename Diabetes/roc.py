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
import matplotlib.pyplot as plt



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

#pickle.dump(model_knn, open('./models/model_knn.pkl', 'wb'))

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
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

# 创建Logistic回归模型
model_logistics = LogisticRegression()

# 使用训练集拟合模型
model_logistics.fit(train_x,train_y)

# 对验证集进行预测
logistics_pred = model_logistics.predict(test_x)

# 存储模型评估指标
scores_logistics=[]

# 计算精确率
scores_logistics.append(precision_score(test_y, logistics_pred))

# 计算召回率
scores_logistics.append(recall_score(test_y, logistics_pred))

# 计算准确率:即模型预测正确的样本数与总样本数之比。准确率 = (TP + TN) / (TP + FP + TN + FN)。
scores_logistics.append(accuracy_score(test_y, logistics_pred))

# 计算混淆矩阵
confusion_matrix_logistics=confusion_matrix(test_y, logistics_pred)

# 计算F1分数
f1_score_logistics=f1_score(test_y, logistics_pred,labels=None, pos_label=1, average='binary', sample_weight=None)

# # 获取特征重要性
# importance=pd.DataFrame({"columns":list(test_x.columns), "coef":list(model_logistics.coef_.T)})

# 对验证集进行预测，得到每一类的概率
predictions_log=model_logistics.predict_proba(test_x)#每一类的概率

# 计算ROC曲线
FPR_log, recall_log, thresholds = roc_curve(test_y, predictions_log[:,1],pos_label=1)

# 计算ROC曲线下面积
area_log=auc(FPR_log,recall_log)

print('Log_result：\n')
print(pd.DataFrame(columns=['Prediction = 0', 'Prediction = 1'], index=['Ground True = 0', 'Ground True = 1'],data=confusion_matrix_logistics))# 混淆矩阵
print("f1 Score:"+str(f1_score_logistics))
print("Precision | Recall | Accuracy：" + str(scores_logistics))
# print('模型系数：\n'+str(importance))
#pickle.dump(model_logistics, open('./models/model_log.pkl', 'wb'))

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

#pickle.dump(model_svm, open('./models/model_svm.pkl', 'wb'))


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

#pickle.dump(model_XGB, open('./models/model_xgb.pkl', 'wb'))

# ROC图的绘制
# 创建画布
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# 绘制XGBoost的ROC曲线（粉色）
plt.plot(FPR_xgb, recall_xgb, color='pink', label='XGBoost_AUC = %0.3f' % area_xgb)

# 绘制SVM的ROC曲线（浅蓝色）
plt.plot(FPR_svm, recall_svm, color='lightblue', label='SVM_AUC = %0.3f' % area_svm)

# 绘制Logistic回归的ROC曲线（橙黄色）
plt.plot(FPR_log, recall_log, color='orange', label='Logistic_AUC = %0.3f' % area_log)

# 绘制knn的ROC曲线（紫色）
plt.plot(FPR_knn, recall_knn, color='red', label='KNN_AUC = %0.3f' % area_knn)

plt.legend(loc='lower right')        # 添加图例
plt.plot([0, 1], [0, 1], 'r--',color='grey')      # 绘制对角线
plt.xlim([0.0, 1.0])                 # 设置x轴范围
plt.ylim([0.0, 1.0])                 # 设置y轴范围
plt.ylabel('Recall')                 # 设置y轴标签
plt.xlabel('FPR')                    # 设置x轴标签
plt.title('ROC_before_GridSearchCV') # 设置标题
plt.savefig("ROC.png")
plt.show()                           # 显示图形
