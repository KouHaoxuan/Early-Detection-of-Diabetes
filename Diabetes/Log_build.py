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
pickle.dump(model_logistics, open('./models/model_log.pkl', 'wb'))