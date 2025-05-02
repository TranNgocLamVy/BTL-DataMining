import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


df = pd.read_csv('datasets/diem_thi_thpt_2024.csv')

df.columns = ['sbd', 'toan', 'van', 'anh', 'ly', 'hoa', 'sinh', 'su', 'dia', 'gdcd', 'ma_nn']

cols = ['toan', 'van', 'anh', 'ly', 'hoa', 'sinh', 'su', 'dia', 'gdcd']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

df = df[df['toan'].notna()]

df['gioi_toan'] = df['toan'] >= 8.0

print("Trung bình các môn theo nhóm 'giỏi Toán':")
print(df.groupby('gioi_toan')[['van', 'anh']].mean())

sns.boxplot(data=df, x='gioi_toan', y='van')
plt.title("So sánh điểm Văn giữa học sinh giỏi và không giỏi Toán")
plt.xlabel("Giỏi Toán (True/False)")
plt.ylabel("Điểm Văn")
plt.show()

df_model = df[['toan', 'hoa', 'ly']].dropna()

X = df_model[['toan', 'hoa']]
y = df_model['ly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("\n Kết quả hồi quy tuyến tính dự đoán điểm Lý:")
print("Hệ số hồi quy:", model.coef_)
print("Intercept:", model.intercept_)
print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")