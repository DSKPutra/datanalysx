"""DATANALYSX — modul komputasi Python yang dijalankan di browser (Pyodide).

Setiap submodul (t01 ... t09, eda, prep, datalab) diringkas dari skrip tugas asli
(lihat folder python/original) menjadi fungsi-fungsi kecil yang menerima data dan
parameter, lalu mengembalikan dict yang aman di-serialisasi ke JSON.

Seluruh fungsi deterministik: random_state = 42.
"""

RANDOM_STATE = 42
