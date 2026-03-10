from sklearn.datasets import load_diabetes
import pandas as pd

# 加载糖尿病数据集
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# 创建DataFrame便于查看
feature_names = diabetes.feature_names
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# 查看数据
print(f"特征字段: {diabetes.feature_names}")
print(f"样本数: {len(df)}")
print(f"特征数: {X.shape[1]}")
print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")
print(f"目标变量均值: {y.mean():.2f}, 标准差: {y.std():.2f}")

# 显示前5行数据
print("\n数据示例（前5行）:")
print(df.head())

# 特征描述
print("\n特征描述:")
print("-" * 40)
print("age: 年龄")
print("sex: 性别")
print("bmi: 身体质量指数")
print("bp: 平均血压")
print("s1~s6: 六种血清含量指标")
print("-" * 40)
