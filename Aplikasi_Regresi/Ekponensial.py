import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Memuat data
data_url = r"C:\Users\Public\python-2\student_performance.csv"  # Ganti dengan path yang sesuai
data = pd.read_csv(data_url)

# Memilih kolom yang relevan
data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]

# Mengambil data yang dibutuhkan untuk analisis
X = data['Hours Studied'].values
y = data['Performance Index'].values

# Model Eksponensial (Metode 3)
def exponential_model(x, a, b):
    return a * np.exp(b * x)

# Memasang model eksponensial ke data
params, covariance = curve_fit(exponential_model, X, y)
a, b = params
y_pred_exponential = exponential_model(X, a, b)

# Plot data dan hasil regresi eksponensial
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred_exponential, color='green', label='Exponential Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Exponential Regression')
plt.legend()
plt.show()

# Menghitung RMS error untuk model eksponensial
rms_exponential = np.sqrt(mean_squared_error(y, y_pred_exponential))
print(f'RMS Error for Exponential Model: {rms_exponential}')