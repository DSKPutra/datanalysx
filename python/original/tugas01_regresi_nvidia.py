# Sumber: Lampiran kode program pada laporan Tugas 01 (Google Colab), disalin apa adanya.
# Penanda "# %% Cell N" menandai sel notebook asli.

# %% Cell 1
# =========================================
# ANALISIS PREDIKSI HARGA SAHAM NVIDIA
# MENGGUNAKAN ALGORITMA LINEAR REGRESSION
# =========================================
 
# Nama  : [Dea Saka Kurnia Putra]
# NIM   : [261012000111]
# Kelas : [S2 TI Reg B 02]

# %% Cell 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from google.colab import files
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %% Cell 3
uploaded = files.upload()

# %% Cell 4
print(list(uploaded.keys())[0])

# %% Cell 5
df = pd.read_csv('nvidia_stock_data_1999_2026.csv')

# %% Cell 6
df.head()

# %% Cell 7
print("Jumlah baris dan kolom:", df.shape)

# %% Cell 8
print("Nama kolom:")
print(df.columns.tolist())

# %% Cell 9
df.info()

# %% Cell 10
df.describe()

# %% Cell 11
print("Jumlah missing value setiap kolom:")
print(df.isnull().sum())

# %% Cell 12
print("Total missing value:", df.isnull().sum().sum())

# %% Cell 13
print("Jumlah data duplikat:", df.duplicated().sum())

# %% Cell 14
df = df.drop_duplicates()
 
print("Jumlah data setelah menghapus duplikat:", df.shape)

# %% Cell 15
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df = df.sort_values('date').reset_index(drop=True)
 
print("Rentang tanggal:", df["date"].min(), "-", df["date"].max())

# %% Cell 16
plt.figure(figsize=(8,4.7))
 
plt.plot(df['date'], df['close'], linewidth=0.9)
 
plt.xlabel('Tahun')
plt.ylabel('Harga Close (USD)')
plt.title('Pergerakan Harga Close Saham NVIDIA (1999-2026)')
 
plt.show()

# %% Cell 17
df[
    [
        'open',
        'high',
        'low',
        'volume',
        'sma_50',
        'rsi_14',
        'close'
    ]
].corr()

# %% Cell 18
##membuat visualisasi korelasi sederhana:
 
plt.figure(figsize=(8,6))
 
plt.imshow(
    df[
        [
            'open',
            'high',
            'low',
            'volume',
            'sma_50',
            'rsi_14',
            'close'
        ]
    ].corr(),
    cmap='coolwarm',
    interpolation='nearest'
)
 
plt.colorbar()
 
plt.xticks(
    range(7),
    ['Open', 'High', 'Low', 'Volume', 'SMA 50', 'RSI 14', 'Close'],
    rotation=45
)
 
plt.yticks(
    range(7),
    ['Open', 'High', 'Low', 'Volume', 'SMA 50', 'RSI 14', 'Close']
)
 
plt.title('Korelasi Variabel Numerik')
 
plt.show()

# %% Cell 19
# Kita gunakan enam variabel numerik sebagai faktor prediksi:
#Open
#High
#Low
#Volume
#SMA 50
#RSI 14
 
#Sedangkan Close menjadi target.
 
X = df[
    [
        'open',
        'high',
        'low',
        'volume',
        'sma_50',
        'rsi_14'
    ]
]
 
y = df['close']

# %% Cell 20
print("Variabel X:")
display(X.head())
 
print("\nTarget Y:")
display(y.head())

# %% Cell 21
#Kita gunakan 80% data untuk training dan 20% untuk testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
 
print("Jumlah data training:", X_train.shape[0])
print("Jumlah data testing :", X_test.shape[0])

# %% Cell 22
model = LinearRegression()
 
model.fit(X_train, y_train)

# %% Cell 23
y_pred = model.predict(X_test)

# %% Cell 24
#Kita bandingkan harga Close asli dengan harga hasil prediksi.
 
hasil_prediksi = pd.DataFrame({
    'Harga Aktual': y_test.values,
    'Harga Prediksi': y_pred
})
 
hasil_prediksi.head(10)

# %% Cell 25
mae = mean_absolute_error(y_test, y_pred)
 
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
 
r2 = r2_score(y_test, y_pred)
 
print("HASIL EVALUASI REGRESI LINEAR")
print("==============================")
print("MAE  :", mae)
print("RMSE :", rmse)
print("R²   :", r2)
 
#Cara membacanya nanti:
#MAE → rata-rata kesalahan prediksi.
#RMSE → mengukur besar kesalahan prediksi, dengan memberikan penalti lebih besar pada kesalahan yang besar.
#R² → menunjukkan seberapa besar variasi Close dapat dijelaskan oleh variabel yang digunakan model.
 
#Untuk MAE dan RMSE, semakin kecil semakin baik.
 
#Untuk R², semakin mendekati 1 semakin baik.

# %% Cell 26
plt.figure(figsize=(8,5))
 
plt.scatter(y_test, y_pred, alpha=0.5)
 
plt.xlabel('Harga Aktual')
plt.ylabel('Harga Prediksi')
plt.title('Perbandingan Harga Aktual dan Harga Prediksi')
 
plt.show()

# %% Cell 27
koefisien = pd.DataFrame({
    'Variabel': X.columns,
    'Koefisien': model.coef_
})
 
koefisien

# %% Cell 28
print("Intercept:", model.intercept_)
# Bagian ini bisa digunakan untuk melihat arah pengaruh masing-masing variabel dalam model regresi.
