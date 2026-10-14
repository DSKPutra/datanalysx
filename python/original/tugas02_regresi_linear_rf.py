# Sumber: Lampiran kode program pada laporan Tugas 02 (Google Colab), disalin apa adanya.
# Penanda "# %% Cell N" menandai sel notebook asli.

# %% Cell 1
# ANALISIS DAN PREDIKSI TINGKAT KEPARAHAN KEGAGALAN AGEN AI
# PADA AI ECONOMY PULSE DATASET TAHUN 2022-2026
# MENGGUNAKAN ALGORITMA LINEAR DAN NON-LINEAR
 
# Nama  : [Dea Saka Kurnia Putra]
# NIM   : [261012000111]
# Kelas : [S2 TI Reg B 02]

# %% Cell 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %% Cell 5
from google.colab import files
 
uploaded = files.upload()

# %% Cell 6
df = pd.read_csv('AI_Economy_Pulse_Dataset_2022_-_Present.csv')
 
df.head()

# %% Cell 8
print("Jumlah baris dan kolom:", df.shape)
 
df.info()

# %% Cell 9
df.isnull().sum()

# %% Cell 10
print("Jumlah data duplikat:", df.duplicated().sum())

# %% Cell 12
# Rekayasa fitur: encoding kategori dan penghitungan panjang teks
difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
severity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
 
df['difficulty_score'] = df['difficulty'].map(difficulty_map)
df['severity_score'] = df['failure_severity'].map(severity_map)
df['prompt_length'] = df['prompt'].astype(str).str.split().apply(len)
df['context_length'] = df['context'].astype(str).str.split().apply(len)
 
df[['difficulty_score', 'prompt_length', 'context_length', 'severity_score']].describe()

# %% Cell 14
plt.figure(figsize=(8,5))
 
plt.scatter(df['context_length'], df['severity_score'])
 
plt.xlabel('Panjang Konteks Tugas (context_length)')
plt.ylabel('Tingkat Keparahan Kegagalan (severity_score)')
plt.title('Hubungan Panjang Konteks dengan Tingkat Keparahan Kegagalan')
 
plt.show()

# %% Cell 16
X = df[['difficulty_score', 'prompt_length', 'context_length']]
y = df['severity_score']

# %% Cell 18
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
 
print("Data training:", X_train.shape)
print("Data testing :", X_test.shape)

# %% Cell 20
linear_model = LinearRegression()
 
linear_model.fit(X_train, y_train)
 
y_pred_linear = linear_model.predict(X_test)

# %% Cell 21
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)
 
print("Linear Regression")
print("MAE  :", mae_linear)
print("RMSE :", rmse_linear)
print("R2   :", r2_linear)

# %% Cell 23
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
 
rf_model.fit(X_train, y_train)
 
y_pred_rf = rf_model.predict(X_test)

# %% Cell 24
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
 
print("Random Forest Regression")
print("MAE  :", mae_rf)
print("RMSE :", rmse_rf)
print("R2   :", r2_rf)

# %% Cell 26
hasil = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Random Forest'
    ],
    'MAE': [
        mae_linear,
        mae_rf
    ],
    'RMSE': [
        rmse_linear,
        rmse_rf
    ],
    'R2': [
        r2_linear,
        r2_rf
    ]
})
 
hasil

# %% Cell 28
plt.figure(figsize=(8,5))
 
plt.scatter(y_test, y_pred_linear)
 
plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')
plt.title('Linear Regression - Aktual vs Prediksi')
 
plt.show()

# %% Cell 29
plt.figure(figsize=(8,5))
 
plt.scatter(y_test, y_pred_rf)
 
plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')
plt.title('Random Forest - Aktual vs Prediksi')
 
plt.show()
