import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# نمایش مسیر فعلی اجرا
print("Current Directory:", os.getcwd())

# مسیر فایل CSV (مسیر دقیق خود را جایگزین کنید)
file_path = "Data/CreditPrediction.csv"

# اطمینان از وجود فایل
if not os.path.exists(file_path):
    print("فایل CSV در مسیر مشخص شده پیدا نشد!")
    exit()

# بارگذاری داده‌ها
df = pd.read_csv(file_path)
print(df.head())

# حذف ستون‌های کاملاً خالی یا تقریبا خالی (حداقل 10% داده موجود باشد)
df = df.dropna(axis=1, thresh=len(df)*0.1)

# تعریف ستون هدف
target_column = 'Credit_Limit'

# جدا کردن متغیر هدف و ویژگی‌ها
features = df.drop(target_column, axis=1)
target = df[target_column]

# پرکردن مقادیر گمشده در ستون‌های عددی با میانگین همان ستون‌ها
numeric_cols = features.select_dtypes(include=[np.number]).columns
features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].mean())

# حذف داده‌های پرت (مقادیر خارج از 3 انحراف معیار) در داده‌های عددی
z_scores = np.abs(stats.zscore(features[numeric_cols]))
filter_mask = (z_scores < 3).all(axis=1)
features = features.loc[filter_mask]
target = target.loc[filter_mask]

# تبدیل داده‌های غیر عددی با One-Hot Encoding
non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns
features = pd.get_dummies(features, columns=non_numeric_cols)

# تبدیل همه داده‌ها به نوع float برای مدل عصبی و پر کردن احتمالی مقدارهای باقی‌مانده با صفر
features = features.astype(float).fillna(0)
target = target.astype(float)

# نرمال‌سازی داده‌ها
scaler = StandardScaler()
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

# آرایه numpy آماده مدل
X = features.values.astype(np.float32)
y = target.values.astype(np.float32)

# تقسیم داده به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ساخت مدل شبکه عصبی
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# آموزش مدل
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)
# نمایش روند آموزش مدل
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Neural Network Training Performance")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()

# پیش‌بینی و ارزیابی مدل
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# Actual vs Predicted Credit Limit
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual Credit Limit")
plt.ylabel("Predicted Credit Limit")
plt.title("Actual vs. Predicted Credit Limit")

# خط ایده‌آل پیش‌بینی
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.grid(True)
plt.tight_layout()
plt.show()

print('MSE:', mse)
print('RMSE:', rmse)
# Display final model evaluation results
plt.figure(figsize=(8, 5))

plt.axis("off")

plt.text(
    0.5,
    0.75,
    "Credit Limit Prediction",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold"
)

plt.text(
    0.5,
    0.55,
    "Neural Network Model Evaluation",
    ha="center",
    va="center",
    fontsize=16
)

plt.text(
    0.5,
    0.35,
    f"MSE: {mse:,.2f}\nRMSE: {rmse:,.2f}",
    ha="center",
    va="center",
    fontsize=18
)

plt.tight_layout()
plt.show()
