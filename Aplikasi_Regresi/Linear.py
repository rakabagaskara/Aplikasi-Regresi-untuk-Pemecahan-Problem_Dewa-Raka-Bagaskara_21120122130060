import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Memuat data
data_url = r"C:\Users\Public\python-2\student_performance.csv"  # Ganti dengan path yang sesuai
data = pd.read_csv(data_url)

# Memilih kolom yang relevan
data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]

# Mengambil data yang dibutuhkan untuk analisis
X = data['Hours Studied'].values.reshape(-1, 1)
y = data['Performance Index'].values

# Regresi Linear (Metode 1)
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Plot data dan hasil regresi linear
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred_linear, color='red', label='Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Menghitung RMS error untuk model linear
rms_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
print(f'RMS Error for Linear Model: {rms_linear}')