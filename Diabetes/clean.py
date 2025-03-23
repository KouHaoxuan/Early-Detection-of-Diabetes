import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
diabetes_data = pd.read_csv("./data/diabetes.csv")
# 简要摘要
diabetes_data.info()
diabetes_data.head()

# 统计摘要
diabetes_data.describe().T
diabetes_data.hist(figsize=(20,20), color='lightblue')
plt.savefig("distribution.png")
plt.show()


diabetes_data_02 = pd.read_csv("./data/diabetes.csv")
# 定义字段
get_col = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
# 0转为np.nan
diabetes_data_02[get_col] = diabetes_data_02[get_col].replace(0, np.nan)
# 统计np.nan的次数
# True == 1  False == 0
diabetes_data_02.isnull().sum()

# 分布形态 ->> 连续值 ->> 直方图
diabetes_data_02.hist(figsize=(20,20), color="lightpink")
plt.savefig("distribution_after.png")
plt.show()

# 填充缺失值
diabetes_data_02['Glucose'] = diabetes_data_02['Glucose'].fillna(diabetes_data_02['Glucose'].mean())
diabetes_data_02['BloodPressure'] = diabetes_data_02['BloodPressure'].fillna(diabetes_data_02['BloodPressure'].mean())
diabetes_data_02['SkinThickness'] = diabetes_data_02['SkinThickness'].fillna(diabetes_data_02['SkinThickness'].median())
diabetes_data_02['Insulin'] = diabetes_data_02['Insulin'].fillna(diabetes_data_02['Insulin'].median())
diabetes_data_02['BMI'] = diabetes_data_02['BMI'].fillna(diabetes_data_02['BMI'].median())
# 替换后的数据
diabetes_data_02.info()
diabetes_data_02.head()
diabetes_data_02.describe().T
diabetes_data_02.to_csv('./data/cleaned_data.csv', index=False)

# 创建一个大小为(12, 30)的画布
# 创建画布
fig = plt.figure(figsize=(20, 20))

# 调整子图内部间距
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# 怀孕次数
plt.subplot2grid((4, 2), (0, 0))
diabetes_data_02.Pregnancies[diabetes_data_02.Outcome == 1].plot(kind='kde', color='lightblue')
diabetes_data_02.Pregnancies[diabetes_data_02.Outcome == 0].plot(kind='kde', color='lightpink')
plt.xlabel(u"Pregnancies")  # 设置x轴标签
plt.ylabel(u"Density")
plt.title(u"Pregnancies Distribution")
plt.legend((u'Diabetic', u'Normal'), loc='best')

# 葡萄糖
plt.subplot2grid((4, 2), (0, 1))
diabetes_data_02.Glucose[diabetes_data_02.Outcome == 1].plot(kind='kde', color='lightblue')
diabetes_data_02.Glucose[diabetes_data_02.Outcome == 0].plot(kind='kde', color='lightpink')
plt.xlabel(u"Glucose")
plt.ylabel(u"Density")
plt.title(u"Glucose Distribution")
plt.legend((u'Diabetic', u'Normal'), loc='best')

# 血压
plt.subplot2grid((4, 2), (1, 0))
diabetes_data_02.BloodPressure[diabetes_data_02.Outcome == 1].plot(kind='kde', color='lightblue')
diabetes_data_02.BloodPressure[diabetes_data_02.Outcome == 0].plot(kind='kde', color='lightpink')
plt.xlabel(u"BloodPressure")
plt.ylabel(u"Density")
plt.title(u"BloodPressure Distribution")
plt.legend((u'Diabetic', u'Normal'), loc='best')

# 皮肤厚度
plt.subplot2grid((4, 2), (1, 1))
diabetes_data_02.SkinThickness[diabetes_data_02.Outcome == 1].plot(kind='kde', color='lightblue')
diabetes_data_02.SkinThickness[diabetes_data_02.Outcome == 0].plot(kind='kde', color='lightpink')
plt.xlabel(u"SkinThickness")
plt.ylabel(u"Density")
plt.title(u"SkinThickness Distribution")
plt.legend((u'Diabetic', u'Normal'), loc='best')

# 胰岛素
plt.subplot2grid((4, 2), (2, 0))
diabetes_data_02.Insulin[diabetes_data_02.Outcome == 1].plot(kind='kde', color='lightblue')
diabetes_data_02.Insulin[diabetes_data_02.Outcome == 0].plot(kind='kde', color='lightpink')
plt.xlabel(u"Insulin")
plt.ylabel(u"Density")
plt.title(u"Insulin Distribution")
plt.legend((u'Diabetic', u'Normal'), loc='best')

# BMI
plt.subplot2grid((4, 2), (2, 1))
diabetes_data_02.BMI[diabetes_data_02.Outcome == 1].plot(kind='kde', color='lightblue')
diabetes_data_02.BMI[diabetes_data_02.Outcome == 0].plot(kind='kde', color='lightpink')
plt.xlabel(u"BMI")
plt.ylabel(u"Density")
plt.title(u"BMI Distribution")
plt.legend((u'Diabetic', u'Normal'), loc='best')

# 糖尿病谱系功能
plt.subplot2grid((4, 2), (3, 0))
diabetes_data_02.DiabetesPedigreeFunction[diabetes_data_02.Outcome == 1].plot(kind='kde', color='lightblue')
diabetes_data_02.DiabetesPedigreeFunction[diabetes_data_02.Outcome == 0].plot(kind='kde', color='lightpink')
plt.xlabel(u"DiabetesPedigreeFunction")
plt.ylabel(u"Density")
plt.title(u"DiabetesPedigreeFunction Distribution")
plt.legend((u'Diabetic', u'Normal'), loc='best')

# 年龄
plt.subplot2grid((4, 2), (3, 1))
diabetes_data_02.Age[diabetes_data_02.Outcome == 1].plot(kind='kde', color='lightblue')
diabetes_data_02.Age[diabetes_data_02.Outcome == 0].plot(kind='kde', color='lightpink')
plt.xlabel(u"Age")
plt.ylabel(u"Density")
plt.title(u"Age Distribution")
plt.legend((u'Diabetic', u'Normal'), loc='best')
plt.savefig("feature comparison.png")



# 观察特征之间的相关性 ->> 特征与特征的相关性所构建出的数组
corr=diabetes_data_02.corr()
# 绘制热力图展示
plt.figure(figsize=(12,10))
ax=sns.heatmap(corr,cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True))
ax.tick_params(axis='both', which='major', labelsize=10)  # 设置x轴和y轴刻度标签的字体大小
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # 旋转x轴标签并调整水平对齐方式以避免重叠

plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.savefig("Heatmap.png")

plt.show()