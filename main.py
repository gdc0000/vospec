import re
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import norm
from nltk.corpus import stopwords
import nltk
# Ensure stopwords are downloaded
nltk.download('stopwords')

# ---------------------- Utils: MOTS ----------------------
def _preserve_uppercase_token(token: str) -> str:
    return token if token.isupper() else token.lower()

def mots(df: pd.DataFrame, text_col: str, language: str, ngram_range: tuple):
    stopword_set = set(stopwords.words(language))

    def normalize(text):
        return re.sub(r"[\n\r\t]+", " ", str(text)).strip()

    def tokenize(text):
        tokens = re.findall(r"\w+(?:/\w+)*", text)
        return [_preserve_uppercase_token(tok) for tok in tokens]

    def extract_ngrams(tokens, n):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens)>=n else []

    def is_stopword(ngram):
        # preserve slash tokens
        if '/' in ngram:
            return False
        return any(tok in stopword_set for tok in ngram.split())

    all_tokens = []
    for raw in df[text_col]:
        tokens = tokenize(normalize(raw))
        for n in range(ngram_range[0], ngram_range[1]+1):
            for ng in extract_ngrams(tokens, n):
                if not is_stopword(ng):
                    all_tokens.append(ng)

    counter = Counter(all_tokens)
    df_tokens = pd.DataFrame(counter.items(), columns=['token','frequenza'])
    return df_tokens

# ---------------------- Utils: VOSPEC ----------------------
def vospec(df: pd.DataFrame, text_col: str, group_col: str, language: str,
           ngram_range: tuple, min_freq: int, z_threshold: float):
    stopword_set = set(stopwords.words(language))

    # reuse tokenize and extract from mots
    def normalize(text):
        return re.sub(r"[\n\r\t]+", " ", str(text)).strip()
    def tokenize(text):
        tokens = re.findall(r"\w+(?:/\w+)*", text)
        return [_preserve_uppercase_token(tok) for tok in tokens]
    def extract_ngrams(tokens, n):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens)>=n else []
    def is_stopword(ngram):
        if '/' in ngram:
            return False
        return any(tok in stopword_set for tok in ngram.split())

    # build group tokens
    group_tokens = defaultdict(list)
    for raw, grp in zip(df[text_col], df[group_col]):
        tokens = tokenize(normalize(raw))
        for n in range(ngram_range[0], ngram_range[1]+1):
            for ng in extract_ngrams(tokens, n):
                if not is_stopword(ng):
                    group_tokens[grp].append(ng)

    # compute counts
    group_counts = {g: Counter(toks) for g,toks in group_tokens.items()}
    global_counts = sum((Counter(toks) for toks in group_tokens.values()), Counter())
    total_all = sum(global_counts.values())
    totals_by_group = {g: sum(c.values()) for g,c in group_counts.items()}

    records = []
    for token, freq_global in global_counts.items():
        if freq_global < min_freq:
            continue
        for grp, counter in group_counts.items():
            freq_in = counter.get(token,0)
            total_in = totals_by_group[grp]
            freq_out = freq_global - freq_in
            total_out = total_all - total_in
            if min(freq_in, freq_out, total_in, total_out)==0:
                continue
            p1 = freq_in/total_in
            p2 = freq_out/total_out
            p = (freq_in+freq_out)/(total_in+total_out)
            se = np.sqrt(p*(1-p)*(1/total_in+1/total_out))
            if se==0: continue
            z = (p1-p2)/se
            p_value = 2*(1-norm.cdf(abs(z)))
            direction = 'over' if z>0 else 'under'
            if abs(z)>z_threshold:
                records.append({'token':token,'group':grp,'z_score':z,
                                'p_value':p_value,'direction':direction,
                                'freq_in_group':freq_in,'freq_global':freq_global})
    return pd.DataFrame(records)

# ---------------------- Streamlit App ----------------------
st.title("Lexicometric Analysis: MOTS & VOSPEC")

# File upload
uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("File loaded")

    # Sidebar options
    st.sidebar.header("Settings")
    text_col = st.sidebar.selectbox("Text column", df.columns)
    group_col = st.sidebar.selectbox("Group column (VOSPEC)", df.columns)
    langs = stopwords.fileids()
    language = st.sidebar.selectbox("Stopword language", langs, index=langs.index('italian'))
    ngram_min = st.sidebar.number_input("Min n-gram", 1, 3, 1)
    ngram_max = st.sidebar.number_input("Max n-gram", 1, 3, 2)
    min_freq = st.sidebar.number_input("Min frequency (VOSPEC)", 1, 50, 5)
    z_th = st.sidebar.slider("Z-threshold", 1.0, 3.0, 1.96)

    # Run MOTS
    if st.sidebar.button("Run MOTS"):
        mots_df = mots(df, text_col, language, (ngram_min, ngram_max))
        st.subheader("MOTS Results")
        st.dataframe(mots_df)
        # Download link
        st.download_button("Download MOTS Excel", data=
            mots_df.to_excel(index=False), file_name="mots.xlsx")

    # Run VOSPEC
    if st.sidebar.button("Run VOSPEC"):
        vospec_df = vospec(df, text_col, group_col, language,
                           (ngram_min, ngram_max), min_freq, z_th)
        st.subheader("VOSPEC Results")
        st.dataframe(vospec_df)
        st.download_button("Download VOSPEC Excel",
            data=vospec_df.to_excel(index=False), file_name="vospec.xlsx")
