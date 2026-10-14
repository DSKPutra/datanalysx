"""Studi Kasus 09 — Text Mining komentar publik terkait program Makan Bergizi Gratis (MBG).

PERHATIAN: laporan studi kasus 09 dan skrip textmining_mbg.py BELUM tersedia. Modul ini adalah
implementasi baru untuk lab aplikasi, mengikuti rancangan aplikasi (tahap
preprocessing teks Bahasa Indonesia standar), bukan reproduksi hasil laporan.

Etika: nama akun diganti pseudonim (hash), kata kasar disensor pada setiap kutipan,
dan hasil hanya memetakan sentimen korpus — bukan opini publik Indonesia.
Sumber daya leksikal dimuat saat runtime dari repositori aslinya (tidak diredistribusi):
  • InSet — Koto & Rahmaningtyas (2017), github.com/fajri91/InSet
  • Colloquial Indonesian Lexicon — Salsabila dkk. (2018), github.com/nasalsabila/kamus-alay
  • Stopword & stemmer Sastrawi (PySastrawi, MIT)
"""
from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from . import RANDOM_STATE
from .common import CACHE, RESOURCES, cached, get, set_frame

NEGATIONS = {"tidak", "tak", "bukan", "jangan", "belum", "enggak", "nggak", "gak", "ga", "tdk", "kagak", "ndak"}
MBG_PATTERN = re.compile(r"\b(mbg|makan bergizi|makan gratis|makan siang gratis|bgn|badan gizi)\b", re.I)
# Daftar kecil kata kasar umum untuk SENSOR tampilan (bukan untuk klasifikasi).
PROFANITY = {"anjing", "anjir", "anjrit", "bangsat", "babi", "goblok", "goblog", "tolol", "bego", "kontol", "memek",
             "ngentot", "jancok", "jancuk", "asu", "bajingan", "keparat", "brengsek", "tai", "taik", "kampret",
             "sialan", "idiot", "bodoh", "pantek", "pukimak", "setan", "iblis", "laknat", "monyet", "kunyuk"}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w.]+")
HASHTAG_RE = re.compile(r"#\w+")
NUM_RE = re.compile(r"\d+")
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF‍️]")
PUNCT_RE = re.compile(r"[^\w\s]|_")
ELONG_RE = re.compile(r"(\w)\1{2,}")


# ---------------------------------------------------------------- sumber daya leksikal
def _stopwords() -> set[str]:
    def build():
        try:
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
            sw = set(StopWordRemoverFactory().get_stop_words())
        except Exception:  # pragma: no cover - Sastrawi selalu terpasang pada lab
            sw = set()
        return sw
    return cached("res:sastrawi_stop", build)


def _stemmer():
    def build():
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        st = StemmerFactory().create_stemmer()
        # ArrayDictionary Sastrawi memakai list (pencarian O(n)); diganti set agar ~400× lebih cepat.
        # Hasil stemming identik karena hanya operasi `in` yang dipakai.
        d = st.delegatedStemmer.dictionary
        d.words = set(d.words)
        return st
    return cached("res:sastrawi_stem", build)


def slang_dict() -> dict[str, str]:
    def build():
        txt = RESOURCES.get("kamus_alay")
        if not txt:
            return {}
        df = pd.read_csv(io.StringIO(txt), usecols=["slang", "formal"]).dropna()
        df["slang"] = df["slang"].astype(str).str.lower().str.strip()
        df["formal"] = df["formal"].astype(str).str.lower().str.strip()
        return dict(zip(df["slang"], df["formal"]))
    return cached("res:kamus_alay:dict", build)


def inset() -> dict[str, int]:
    def build():
        lex: dict[str, int] = {}
        for name in ("inset_positive", "inset_negative"):
            txt = RESOURCES.get(name)
            if not txt:
                continue
            df = pd.read_csv(io.StringIO(txt), sep="\t")
            for w, s in zip(df["word"].astype(str), df["weight"].astype(int)):
                lex[w.lower().strip()] = lex.get(w.lower().strip(), 0) + int(s)
        return lex
    return cached("res:inset:dict", build)


def resources_status() -> dict:
    return {"kamus_alay": len(slang_dict()), "inset": len(inset()), "stopwords": len(_stopwords())}


# ---------------------------------------------------------------- tahap preprocessing
def filtering(text: str) -> str:
    t = URL_RE.sub(" ", str(text))
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = NUM_RE.sub(" ", t)
    t = PUNCT_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def case_folding(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> list[str]:
    return text.split()


def stopword_removal(tokens: list[str], keep_negation: bool = True) -> list[str]:
    sw = _stopwords()
    if keep_negation:
        sw = sw - NEGATIONS
    return [t for t in tokens if t not in sw]


def normalize(tokens: list[str]) -> list[str]:
    d = slang_dict()
    out = []
    for t in tokens:
        t2 = ELONG_RE.sub(r"\1\1", t)          # "bangettt" -> "bangett" sebelum dicari di kamus
        f = d.get(t, d.get(t2, t2))
        out.extend(f.split())
    return out


def stem(tokens: list[str]) -> list[str]:
    st = _stemmer()
    return [st.stem(t) or t for t in tokens]


def preprocess(text: str, order: str = "slide", keep_negation: bool = True, do_stem: bool = True,
               do_normalize: bool = True) -> dict:
    """Mengembalikan keluaran setiap tahap.
    order="slide": filtering → case folding → tokenizing → stopword → normalisasi → stemming
    order="normalize_first": filtering → case folding → tokenizing → normalisasi → stopword → stemming
      (normalisasi lebih dulu agar slang negasi seperti "gak" → "tidak" ikut dipertahankan)."""
    steps = [("Teks asli", str(text))]
    f = filtering(text)
    steps.append(("Filtering", f))
    c = case_folding(f)
    steps.append(("Case folding", c))
    toks = tokenize(c)
    steps.append(("Tokenizing", toks))
    if order == "normalize_first":
        if do_normalize:
            toks = normalize(toks)
            steps.append(("Normalisasi slang", toks))
        toks = stopword_removal(toks, keep_negation)
        steps.append(("Stopword removal", toks))
    else:
        toks = stopword_removal(toks, keep_negation)
        steps.append(("Stopword removal", toks))
        if do_normalize:
            toks = normalize(toks)
            steps.append(("Normalisasi slang", toks))
    if do_stem:
        toks = stem(toks)
        steps.append(("Stemming (Sastrawi)", toks))
    return {"steps": [{"stage": s, "output": o} for s, o in steps], "tokens": toks}


def censor(text: str) -> str:
    def rep(m):
        w = m.group(0)
        return w[0] + "*" * (len(w) - 1) if w.lower() in PROFANITY else w
    return re.sub(r"\w+", rep, str(text))


def pseudonym(name: str, salt: str = "datanalysx") -> str:
    return "user_" + hashlib.sha256((salt + str(name)).encode()).hexdigest()[:8]


# ---------------------------------------------------------------- TF-IDF (verifikasi manual)
def tfidf_sklearn(docs: list[str], smooth_idf: bool = True, norm: str | None = "l2", sublinear_tf: bool = False) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, smooth_idf=smooth_idf,
                        norm=norm, sublinear_tf=sublinear_tf)
    M = v.fit_transform(docs).toarray()
    return {"vocab": v.get_feature_names_out().tolist(), "idf": v.idf_, "matrix": M}


def tfidf_manual(docs: list[str], smooth_idf: bool = True, norm: str | None = "l2") -> dict:
    """TF mentah, DF, IDF = ln((1+N)/(1+df)) + 1 (smooth) atau ln(N/df) + 1, lalu normalisasi L2."""
    toks = [d.split() for d in docs]
    vocab = sorted({t for d in toks for t in d})
    N = len(docs)
    tf = np.array([[Counter(d)[w] for w in vocab] for d in toks], float)
    df = (tf > 0).sum(0)
    idf = np.log((1 + N) / (1 + df)) + 1 if smooth_idf else np.log(N / df) + 1
    W = tf * idf
    if norm == "l2":
        nrm = np.sqrt((W ** 2).sum(1, keepdims=True))
        W = W / np.where(nrm == 0, 1, nrm)
    return {"vocab": vocab, "tf": tf, "df": df, "idf": idf, "matrix": W}


# ---------------------------------------------------------------- korpus
def load_corpus(data_key: str, text_col: str | None = None, author_col: str | None = None,
                date_col: str | None = None, likes_col: str | None = None) -> dict:
    raw = get(data_key)
    cols = {c.lower(): c for c in raw.columns}
    text_col = text_col or cols.get("comment") or cols.get("text") or raw.columns[0]
    author_col = author_col or cols.get("author")
    date_col = date_col or cols.get("published_at") or cols.get("date")
    likes_col = likes_col or cols.get("likes")
    df = pd.DataFrame({"text": raw[text_col].astype(str)})
    if author_col:
        df["author"] = raw[author_col].astype(str).map(pseudonym)   # nama akun asli tidak disimpan
    if date_col:
        df["date"] = pd.to_datetime(raw[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    if likes_col:
        df["likes"] = pd.to_numeric(raw[likes_col], errors="coerce")
    n_full_dup = int(raw.duplicated().sum())
    n_text_dup = int(df["text"].duplicated().sum())
    uniq = df.drop_duplicates("text").reset_index(drop=True)
    uniq["n_words"] = uniq["text"].str.split().str.len()
    uniq["mentions_mbg"] = uniq["text"].str.contains(MBG_PATTERN)
    set_frame(data_key + ":corpus", uniq)
    out = {"n_rows": len(raw), "n_cols": raw.shape[1], "columns": [str(c) for c in raw.columns],
           "missing": raw.isna().sum().to_dict(), "n_full_duplicates": n_full_dup, "n_text_duplicates": n_text_dup,
           "n_unique_texts": len(uniq), "share_mbg": float(uniq["mentions_mbg"].mean()),
           "n_words": {"min": int(uniq["n_words"].min()), "median": float(uniq["n_words"].median()),
                       "max": int(uniq["n_words"].max())},
           "length_hist": np.histogram(np.log10(uniq["n_words"].clip(lower=1)), bins=30)[0].tolist(),
           "share_url": float(uniq["text"].str.contains(URL_RE).mean()),
           "share_mention": float(uniq["text"].str.contains(MENTION_RE).mean()),
           "share_emoji": float(uniq["text"].str.contains(EMOJI_RE).mean())}
    if "author" in df:
        out["n_unique_authors"] = int(df["author"].nunique())
        rep = df.groupby("text")["author"].nunique()
        out["n_texts_by_multiple_authors"] = int((rep > 1).sum())
    if "date" in df and df["date"].notna().any():
        w = df.dropna(subset=["date"]).set_index("date").resample("W-MON")["text"].count()
        out["weekly"] = {"week": [str(i.date()) for i in w.index], "count": w.values.tolist()}
        out["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    if "likes" in df:
        lk = df["likes"].dropna()
        out["likes"] = {"median": float(lk.median()), "max": float(lk.max()),
                        "hist_log1p": np.histogram(np.log1p(lk), bins=30)[0].tolist()}
    out["examples"] = [censor(t)[:280] for t in uniq["text"].sample(min(8, len(uniq)), random_state=RANDOM_STATE)]
    return out


def preprocess_corpus(data_key: str, order: str = "normalize_first", keep_negation: bool = True,
                      do_stem: bool = True, subset: str = "all", max_docs: int = 4000) -> dict:
    c = get(data_key + ":corpus")
    if subset == "mbg":
        c = c[c["mentions_mbg"]]
    c = c.head(max_docs).copy()
    stemcache: dict[str, str] = {}
    st = _stemmer() if do_stem else None
    docs = []
    for t in c["text"]:
        toks = tokenize(case_folding(filtering(t)))
        if order == "normalize_first":
            toks = stopword_removal(normalize(toks), keep_negation)
        else:
            toks = normalize(stopword_removal(toks, keep_negation))
        if st is not None:
            toks = [stemcache.setdefault(w, st.stem(w) or w) for w in toks]
        docs.append(" ".join(toks))
    c["clean"] = docs
    c = c[c["clean"].str.len() > 0].reset_index(drop=True)
    CACHE[data_key + ":clean"] = c
    raw_vocab = len({w for t in c["text"] for w in t.lower().split()})
    vocab = Counter(w for d in c["clean"] for w in d.split())
    return {"n_docs": len(c), "subset": subset, "vocab_raw": raw_vocab, "vocab_clean": len(vocab),
            "top_terms": vocab.most_common(30), "zipf": sorted(vocab.values(), reverse=True)[:500],
            "options": {"order": order, "keep_negation": keep_negation, "stem": do_stem}}


def label_lexicon(data_key: str, negation_window: int = 1) -> dict:
    """Label sentimen berbasis leksikon InSet: skor = Σ bobot kata; kata setelah negasi
    dibalik tandanya. Skor > 0 positif, < 0 negatif, = 0 netral.
    Ini label OTOMATIS, bukan ground truth."""
    c = CACHE[data_key + ":clean"]
    lex = inset()
    scores = []
    for d in c["clean"]:
        toks = d.split()
        s, flip = 0, 0
        for t in toks:
            if t in NEGATIONS:
                flip = negation_window
                continue
            w = lex.get(t, 0)
            s += -w if flip > 0 else w
            flip = max(0, flip - 1)
        scores.append(s)
    c["score"] = scores
    c["label"] = np.where(c["score"] > 0, "positif", np.where(c["score"] < 0, "negatif", "netral"))
    CACHE[data_key + ":clean"] = c
    tops = {}
    for lab in ["positif", "netral", "negatif"]:
        cnt = Counter(w for d in c.loc[c["label"] == lab, "clean"] for w in d.split())
        tops[lab] = cnt.most_common(60)
    out = {"counts": c["label"].value_counts().to_dict(), "top_terms": tops,
           "score_hist": np.histogram(np.clip(c["score"], -20, 20), bins=41, range=(-20.5, 20.5))[0].tolist(),
           "examples": {lab: [censor(t)[:200] for t in c.loc[c["label"] == lab, "text"].head(3)]
                        for lab in ["positif", "netral", "negatif"]}}
    if "date" in c and c["date"].notna().any():
        w = c.dropna(subset=["date"]).groupby([pd.Grouper(key="date", freq="W-MON"), "label"]).size().unstack(fill_value=0)
        out["weekly"] = {"week": [str(i.date()) for i in w.index],
                         **{lab: (w[lab].values.tolist() if lab in w else [0] * len(w)) for lab in ["positif", "netral", "negatif"]}}
    if "likes" in c:
        out["likes_by_label"] = c.groupby("label")["likes"].median().to_dict()
    return out


def classify(data_key: str, models: list[str] | None = None, ngram: str = "1", min_df: int = 2,
             test_size: float = 0.2) -> dict:
    """MNB, Logistic Regression, Linear SVM, KNN dengan Pipeline TF-IDF (di-fit hanya pada data latih).
    Duplikat teks sudah dibuang sebelum split; split berstrata; class_weight='balanced' bila tersedia."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import label_binarize
    from sklearn.svm import LinearSVC

    c = CACHE[data_key + ":clean"].drop_duplicates("clean")
    labels = ["negatif", "netral", "positif"]
    y = c["label"].values
    Xtr, Xte, ytr, yte = train_test_split(c["clean"].values, y, test_size=test_size, stratify=y,
                                          random_state=RANDOM_STATE)
    rng = (1, 1) if ngram == "1" else (2, 2) if ngram == "2" else (1, 2)
    all_models = {
        "Multinomial NB": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        "KNN (k=5, cosine)": KNeighborsClassifier(5, metric="cosine"),
    }
    rows, cms = [], {}
    for name in models or list(all_models):
        pipe = Pipeline([("tfidf", TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                                   ngram_range=rng, min_df=min_df)),
                         ("clf", all_models[name])])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        if hasattr(pipe, "predict_proba"):
            score = pipe.predict_proba(Xte)
        else:
            score = pipe.decision_function(Xte)
        try:
            auc = roc_auc_score(label_binarize(yte, classes=pipe.classes_), score, average="macro")
        except ValueError:
            auc = None
        rows.append({"model": name, "accuracy": accuracy_score(yte, pred),
                     "precision_macro": precision_score(yte, pred, average="macro", zero_division=0),
                     "recall_macro": recall_score(yte, pred, average="macro", zero_division=0),
                     "f1_macro": f1_score(yte, pred, average="macro", zero_division=0), "roc_auc_macro": auc})
        cms[name] = confusion_matrix(yte, pred, labels=labels)
    vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=rng, min_df=min_df)
    M = vec.fit_transform(Xtr)
    return {"rows": rows, "confusion": cms, "labels": labels, "n_train": len(ytr), "n_test": len(yte),
            "matrix_shape": list(M.shape), "sparsity": float(1 - M.nnz / (M.shape[0] * M.shape[1])),
            "warning": "Label berasal dari leksikon InSet; model pada dasarnya meniru leksikon (sirkular)."}


def topics(data_key: str, method: str = "nmf", n_topics: int = 5, n_top: int = 10) -> dict:
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    c = CACHE[data_key + ":clean"]
    if method == "lda":
        vec = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, min_df=3, max_df=0.5)
        M = vec.fit_transform(c["clean"])
        model = LatentDirichletAllocation(n_components=n_topics, random_state=RANDOM_STATE, learning_method="batch")
    else:
        vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, min_df=3, max_df=0.5)
        M = vec.fit_transform(c["clean"])
        model = NMF(n_components=n_topics, random_state=RANDOM_STATE, init="nndsvda", max_iter=400)
    W = model.fit_transform(M)
    H = model.components_
    vocab = vec.get_feature_names_out()
    B = (M > 0).astype(int).tocsc()
    dfreq = np.asarray(B.sum(0)).ravel()
    tops, coh = [], []
    for k in range(n_topics):
        idx = np.argsort(H[k])[::-1][:n_top]
        tops.append([str(vocab[i]) for i in idx])
        s = 0.0  # koherensi UMass: Σ log((D(wi,wj)+1)/D(wj))
        for a in range(1, len(idx)):
            for b in range(a):
                co = B[:, idx[a]].multiply(B[:, idx[b]]).sum()
                s += math.log((co + 1) / max(dfreq[idx[b]], 1))
        coh.append(s)
    dom = W.argmax(1)
    out = {"method": method, "n_topics": n_topics, "top_words": tops, "umass": coh,
           "umass_mean": float(np.mean(coh)), "doc_share": np.bincount(dom, minlength=n_topics) / len(dom)}
    if "label" in c:
        tab = pd.crosstab(dom, c["label"]).reindex(columns=["negatif", "netral", "positif"], fill_value=0)
        out["sentiment_by_topic"] = tab.values
    return out


def cooccurrence(data_key: str, top_n: int = 25) -> dict:
    c = CACHE[data_key + ":clean"]
    cnt = Counter(w for d in c["clean"] for w in set(d.split()))
    top = [w for w, _ in cnt.most_common(top_n)]
    ix = {w: i for i, w in enumerate(top)}
    M = np.zeros((top_n, top_n), int)
    for d in c["clean"]:
        ws = [ix[w] for w in set(d.split()) if w in ix]
        for a in ws:
            for b in ws:
                if a != b:
                    M[a, b] += 1
    return {"terms": top, "freq": [cnt[w] for w in top], "matrix": M}


def concordance(data_key: str, keyword: str, width: int = 60, n: int = 12) -> dict:
    c = get(data_key + ":corpus")
    rows = []
    for t in c["text"]:
        for m in re.finditer(re.escape(keyword), t, flags=re.I):
            a, b = max(0, m.start() - width), min(len(t), m.end() + width)
            rows.append({"left": censor(t[a:m.start()]), "kw": t[m.start():m.end()], "right": censor(t[m.end():b])})
            break
        if len(rows) >= n:
            break
    return {"keyword": keyword, "rows": rows}
