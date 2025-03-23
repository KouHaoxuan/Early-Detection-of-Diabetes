import pickle
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# 加载所有模型
models = {
    "KNN": pickle.load(open("./models/model_knn.pkl", "rb")),
    "Logistic Regression": pickle.load(open("./models/model_log.pkl", "rb")),
    "SVM": pickle.load(open("./models/model_svm.pkl", "rb")),
    "XGBoost": pickle.load(open("./models/model_xgb.pkl", "rb"))
}

# 预测函数
def predict_diabetes():
    # 获取用户输入
    try:
        features = [
            float(pregnancies_entry.get()),
            float(glucose_entry.get()),
            float(blood_pressure_entry.get()),
            float(skin_thickness_entry.get()),
            float(insulin_entry.get()),
            float(bmi_entry.get()),
            float(diabetes_pedigree_entry.get()),
            float(age_entry.get())
        ]
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid number")
        return

    # 创建 DataFrame
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age']
    X = pd.DataFrame([features], columns=feature_names)

    # 获取用户选择的模型
    selected_model = models[model_combobox.get()]

    # 进行预测
    predict_outcome = selected_model.predict(X)

    # 显示结果
    if predict_outcome[0]:
        messagebox.showinfo("Result", f"⚠️ Use {model_combobox.get()} to predict：You may have diabetes, please get checked out as soon as possible")
    else:
        messagebox.showinfo("Result", f"✅ Use {model_combobox.get()} to predict：You are healthy and not at risk for diabetes")

# 创建主窗口
root = tk.Tk()
root.title("Diabetes Diagnostic Tool")

# 设置窗口大小
root.geometry("400x500")

# 在窗口边缘增加空白
root.configure(padx=20, pady=20)

# 创建输入框和标签
tk.Label(root, text="Pregnancies:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
pregnancies_entry = tk.Entry(root)
pregnancies_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

tk.Label(root, text="Glucose:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
glucose_entry = tk.Entry(root)
glucose_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

tk.Label(root, text="BloodPressure:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
blood_pressure_entry = tk.Entry(root)
blood_pressure_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

tk.Label(root, text="SkinThickness:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
skin_thickness_entry = tk.Entry(root)
skin_thickness_entry.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

tk.Label(root, text="Insulin:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
insulin_entry = tk.Entry(root)
insulin_entry.grid(row=4, column=1, padx=10, pady=5, sticky="ew")

tk.Label(root, text="BMI:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
bmi_entry = tk.Entry(root)
bmi_entry.grid(row=5, column=1, padx=10, pady=5, sticky="ew")

tk.Label(root, text="DiabetesPedigreeFunction:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
diabetes_pedigree_entry = tk.Entry(root)
diabetes_pedigree_entry.grid(row=6, column=1, padx=10, pady=5, sticky="ew")

tk.Label(root, text="Age:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
age_entry = tk.Entry(root)
age_entry.grid(row=7, column=1, padx=10, pady=5, sticky="ew")

# 创建模型选择下拉菜单
tk.Label(root, text="Select Model:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
model_combobox = ttk.Combobox(root, values=list(models.keys()))
model_combobox.grid(row=8, column=1, padx=10, pady=5, sticky="ew")
model_combobox.current(0)  # 默认选择第一个模型

# 创建预测按钮
predict_button = tk.Button(root, text="Predict", command=predict_diabetes, width=20)
predict_button.grid(row=9, column=0, columnspan=2, pady=20, sticky="ew")  # 使用 sticky="ew" 使按钮水平拉伸

# 创建退出按钮
exit_button = tk.Button(root, text="Exit", command=root.quit, width=20)
exit_button.grid(row=10, column=0, columnspan=2, pady=10, sticky="ew")  # 使用 sticky="ew" 使按钮水平拉伸

# 运行主循环
root.mainloop()