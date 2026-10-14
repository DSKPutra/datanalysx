# Sumber: lampiran kode pada laporan Tugas 03; paragraf penjelasan laporan dijadikan komentar.
# Seluruh perhitungan pada laporan ini dijalankan dengan Python (tanpa pustaka statsmodels) agar proses regresi, uji asumsi, dan validasi sepenuhnya transparan dan dapat direproduksi. Kode berikut disajikan dalam lima bagian: (1) kelas OLS untuk regresi dari nol, (2) pemuatan dan pembersihan data serta pembentukan model, (3) validasi hold-out dan validasi silang 10-lipatan ala widget Test and Score, (4) analisis segmen dan efek tetap perusahaan, dan (5) contoh pembuatan satu visualisasi bergaya widget Orange.
# 1. ols.py — Kelas regresi OLS dari nol (koefisien, t, F, R², VIF, Breusch–Pagan, Durbin–Watson, jarak Cook)
import numpy as np, pandas as pd
from scipy import stats
 
class OLS:
    def __init__(s, X, y, names):
        X = np.asarray(X, float); y = np.asarray(y, float)
        s.X = np.column_stack([np.ones(len(y)), X]); s.y = y
        s.names = ['const'] + list(names)
        n, k = s.X.shape; s.n = n; s.k = k
        s.XtXi = np.linalg.inv(s.X.T @ s.X)          # (X'X)^-1
        s.b = s.XtXi @ s.X.T @ y                      # beta = (X'X)^-1 X'y
        s.fit = s.X @ s.b; s.res = y - s.fit
        s.sse = s.res @ s.res
        s.sst = ((y - y.mean()) ** 2).sum()
        s.ssr = s.sst - s.sse
        s.df = n - k; s.s2 = s.sse / s.df
        s.se = np.sqrt(np.diag(s.XtXi) * s.s2)        # galat baku koefisien
        s.t = s.b / s.se
        s.p = 2 * stats.t.sf(abs(s.t), s.df)          # uji t dua sisi
        s.r2 = 1 - s.sse / s.sst
        s.adj = 1 - (1 - s.r2) * (n - 1) / s.df
        s.F = (s.ssr / (k - 1)) / s.s2                # uji F keseluruhan model
        s.Fp = stats.f.sf(s.F, k - 1, s.df)
        tc = stats.t.ppf(.975, s.df)
        s.ci = np.c_[s.b - tc * s.se, s.b + tc * s.se]  # interval keyakinan 95%
 
    def predict(s, X):
        X = np.asarray(X, float)
        return np.column_stack([np.ones(len(X)), X]) @ s.b
 
    def dw(s):                                        # statistik Durbin-Watson
        return (np.diff(s.res) ** 2).sum() / s.sse
 
    def bp(s):                                         # uji Breusch-Pagan (varian Koenker/LM)
        e2 = s.res ** 2; Z = s.X
        g = np.linalg.lstsq(Z, e2, rcond=None)[0]
        r = 1 - ((e2 - Z @ g) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
        lm = s.n * r
        return lm, stats.chi2.sf(lm, s.k - 1)
 
    def vif(s):                                        # Variance Inflation Factor tiap prediktor
        out = []
        for j in range(1, s.k):
            o = [i for i in range(s.k) if i != j]
            Z = s.X[:, o]
            g = np.linalg.lstsq(Z, s.X[:, j], rcond=None)[0]
            r = s.X[:, j] - Z @ g
            r2 = 1 - (r @ r) / ((s.X[:, j] - s.X[:, j].mean()) ** 2).sum()
            out.append(1 / (1 - r2))
        return out
 
    def cooks(s):                                      # jarak Cook & leverage
        h = np.einsum('ij,jk,ik->i', s.X, s.XtXi, s.X)
        return s.res ** 2 / (s.k * s.s2) * h / (1 - h) ** 2, h
 
 
def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    mse = ((y - p) ** 2).mean()
    return dict(MSE=mse, RMSE=mse ** .5, MAE=abs(y - p).mean(),
                R2=1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
# 2. Pemuatan data, pembersihan, dan pembentukan model regresi
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from ols import OLS, metrics
 
raw = pd.read_csv('chip_companies_financials.csv')
# buang baris dengan revenue/R&D/capex = 0 (perusahaan rintisan belum beroperasi)
c = raw[(raw.revenue_usd_bn > 0) & (raw.rd_spend_usd_bn > 0)
        & (raw.capex_usd_bn > 0)].copy().reset_index(drop=True)
 
R, RD, CX, OM = 'revenue_usd_bn', 'rd_spend_usd_bn', 'capex_usd_bn', 'operating_margin_pct'
c['lR']  = np.log(c[R]); c['lRD'] = np.log(c[RD]); c['lCX'] = np.log(c[CX])
c['rd_int'] = c[RD] / c[R] * 100      # intensitas R&D (% pendapatan)
c['cx_int'] = c[CX] / c[R] * 100      # intensitas capex (% pendapatan)
 
# Regresi sederhana: pendapatan ~ R&D   (skala asli & log-log)
m1  = OLS(c[[RD]],   c[R],  [RD])
m1l = OLS(c[['lRD']], c.lR, ['lRD'])
 
# Regresi berganda: pendapatan ~ R&D + capex + margin   (skala asli & log-log)
m2  = OLS(c[[RD, CX, OM]],        c[R],  ['rd', 'cx', 'om'])
m2l = OLS(c[['lRD', 'lCX', OM]],  c.lR,  ['lrd', 'lcx', 'om'])
 
print('R2 sederhana (asli/log):', round(m1.r2, 3), round(m1l.r2, 3))
print('R2 berganda  (asli/log):', round(m2.r2, 3), round(m2l.r2, 3))
print('VIF berganda log-log   :', [round(v, 2) for v in m2l.vif()])
# 3. Validasi hold-out 80:20 dan validasi silang 10-lipatan (ala widget Test and Score)
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
 
# --- hold-out 80:20 ---
def split(X, y):
    a, b, ya, yb = train_test_split(X, y, test_size=.2, random_state=42)
    m = OLS(a, ya, range(a.shape[1]))
    return m, b, yb, len(a), len(b), m.predict(b)
 
s2 = split(c[[RD, CX, OM]].values, c[R].values)
print('Hold-out berganda R2 uji:', round(metrics(s2[2], s2[5])['R2'], 3))
 
# --- validasi silang 10-lipatan, delapan model dibandingkan seperti Test and Score ---
y = c[R].values
Xs, Xm, Xml = c[[RD]].values, c[[RD, CX, OM]].values, c[['lRD', 'lCX', OM]].values
kf = KFold(10, shuffle=True, random_state=42)
 
def cvpred(model, X, ylog=False):
    P = np.zeros(len(y)); folds = []
    for tr, te in kf.split(X):
        yt = np.log(y[tr]) if ylog else y[tr]
        model.fit(X[tr], yt)
        p = model.predict(X[te]); p = np.exp(p) if ylog else p
        P[te] = p; folds.append((te, p))
    return P, folds
 
def tts(m):
    return TransformedTargetRegressor(regressor=m, transformer=StandardScaler())
 
MODELS = {
    'Constant (rata-rata)':     (DummyRegressor(), Xs, False),
    'Linear – sederhana':       (LinearRegression(), Xs, False),
    'Linear – berganda':        (LinearRegression(), Xm, False),
    'Linear – berganda (log)':  (LinearRegression(), Xml, True),
    'kNN (k=5)':                (make_pipeline(StandardScaler(), KNeighborsRegressor(5)), Xm, False),
    'SGD':                      (tts(make_pipeline(StandardScaler(), SGDRegressor(max_iter=5000, tol=1e-6, random_state=0))), Xm, False),
    'SVM (RBF)':                (tts(make_pipeline(StandardScaler(), SVR(C=3.0))), Xm, False),
    'Random Forest':            (RandomForestRegressor(200, random_state=0, n_jobs=-1), Xm, False),
}
CV = {}
for k, (m, X, lg) in MODELS.items():
    P, folds = cvpred(m, X, lg)
    CV[k] = dict(pred=P, **metrics(y, P))
    print(f"{k:28s}  RMSE={CV[k]['RMSE']:.2f}  MAE={CV[k]['MAE']:.2f}  R2={CV[k]['R2']:.3f}")
 
# --- uji-t berpasangan terkoreksi (Nadeau–Bengio) untuk Model Comparison by MSE ---
from scipy import stats as st
ntest = len(y) / 10; ntrain = len(y) - ntest
def compare(a_mse_fold, b_mse_fold):
    d = a_mse_fold - b_mse_fold
    se = np.sqrt((1/10 + ntest/ntrain) * d.var(ddof=1)) or 1e-12
    t = -d.mean() / se
    return st.t.cdf(t, 9)          # peluang a lebih baik daripada b
# 4. Analisis per-segmen dan model efek tetap perusahaan (fixed effects)
# elastisitas R&D per kelompok segmen (n >= 30), untuk Gambar 18
seg_slopes = []
for g, dfg in c.groupby('grp'):
    if len(dfg) >= 30:
        m = OLS(dfg[['lRD']], dfg.lR, ['x'])
        seg_slopes.append((g, len(dfg), m.b[1], m.ci[1, 0], m.ci[1, 1], m.r2))
seg_slopes.sort(key=lambda t: -t[2])
 
# model efek tetap (within-company): menghilangkan rata-rata per perusahaan
# sebelum regresi, sehingga hanya variasi DALAM perusahaan yang dipakai
_g = c.groupby('company_name')
_xd = c.lRD - _g.lRD.transform('mean')     # ln R&D terpusatkan per perusahaan
_yd = c.lR  - _g.lR.transform('mean')      # ln pendapatan terpusatkan per perusahaan
FE_b   = (_xd @ _yd) / (_xd @ _xd)                       # slope within
_r     = _yd - FE_b * _xd
FE_df  = len(c) - c.company_name.nunique() - 1
FE_se  = np.sqrt((_r @ _r) / FE_df / (_xd @ _xd))
FE_r2w = 1 - (_r @ _r) / (_yd @ _yd)                     # R2 within
FE_r2  = 1 - (_r @ _r) / ((c.lR - c.lR.mean()) ** 2).sum()  # R2 total (dgn intersep perusahaan)
 
print('Elastisitas gabungan (pooled) :', round(m1l.b[1], 3))
print('Elastisitas within (efek tetap):', round(FE_b, 3), '  R2 within =', round(FE_r2w, 4))
# 5. Contoh pembuatan visualisasi bergaya widget Orange (Correlations)
import matplotlib.pyplot as plt
from scipy import stats
 
pairs = []
V = {'Pendapatan': R, 'R&D': RD, 'Capex': CX, 'Margin op.': OM}
ks = list(V)
for i in range(4):
    for j in range(i + 1, 4):
        r_  = c[V[ks[i]]].corr(c[V[ks[j]]])
        rho = stats.spearmanr(c[V[ks[i]]], c[V[ks[j]]])[0]
        pairs.append((f'{ks[i]} ↔ {ks[j]}', r_, rho))
pairs.sort(key=lambda t: -abs(t[1]))
 
fig, ax = plt.subplots(figsize=(6.6, 2.9))
yy = np.arange(len(pairs))[::-1]
ax.barh(yy + .18, [p[1] for p in pairs], .36, label='Pearson r')
ax.barh(yy - .18, [p[2] for p in pairs], .36, label='Spearman ρ')
ax.set_yticks(yy); ax.set_yticklabels([p[0] for p in pairs])
ax.set_title('Correlations – peringkat pasangan variabel'); ax.legend()
plt.savefig('correlations.png', dpi=200, bbox_inches='tight')
