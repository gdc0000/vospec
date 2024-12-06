import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from io import StringIO, BytesIO
import math
import plotly.express as px

# Ensure NLTK resources are available (do this once in your environment)
# nltk.download('punkt')
# nltk.download('stopwords')

LANGUAGES = {
    "English": "english",
    "French": "french",
    "Italian": "italian",
    "Spanish": "spanish"
}

def preprocess_text(text, lang, remove_stopwords, stemmer, ngram_range, stop_words):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Keep only alphabetic tokens
    tokens = [t for t in tokens if t.isalpha()]

    # Remove stopwords if requested
    if remove_stopwords and stop_words is not None:
        tokens = [t for t in tokens if t not in stop_words]

    # Stem tokens
    stemmed_tokens = [stemmer.stem(t) for t in tokens]

    # Generate n-grams
    final_terms = []
    for n in range(1, ngram_range + 1):
        for i in range(len(stemmed_tokens) - n + 1):
            ngram = "_".join(stemmed_tokens[i:i+n])
            final_terms.append(ngram)

    return final_terms, tokens

def find_most_frequent_original_forms(stem2original):
    stem2repr = {}
    for stem, counts in stem2original.items():
        # Pick the original form with highest frequency
        repr_word = max(counts.items(), key=lambda x: x[1])[0]
        stem2repr[stem] = repr_word
    return stem2repr

def add_footer():
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown("""
    [GitHub](https://github.com/gdc0000) | 
    [ORCID](https://orcid.org/0000-0002-1439-5790) | 
    [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """)

def load_data(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'csv':
        return pd.read_csv(uploaded_file)
    elif file_extension in ['xlsx', 'xls']:
        return pd.read_excel(uploaded_file)
    elif file_extension == 'tsv':
        return pd.read_csv(uploaded_file, sep='\t')
    elif file_extension == 'txt':
        return pd.read_csv(uploaded_file, sep='\n', header=None, names=['text'])
    else:
        raise ValueError("Unsupported file type.")

def perform_analysis(df, text_col, category_col, remove_sw, chosen_lang, ngram_range, alpha):
    # Set up stopwords and stemmer
    stop_words = set(stopwords.words(LANGUAGES[chosen_lang])) if remove_sw else None
    stemmer = SnowballStemmer(LANGUAGES[chosen_lang])

    overall_freq = {}
    category_freq = {}
    category_counts = {}
    total_terms = 0

    stem2original = {}
    categories = df[category_col].dropna().unique()

    for cat in categories:
        category_freq[cat] = {}
        category_counts[cat] = 0

    # Preprocessing and frequency counting
    for _, row in df.iterrows():
        cat = row[category_col]
        text = str(row[text_col])
        terms, original_tokens = preprocess_text(
            text, lang=chosen_lang, remove_stopwords=remove_sw,
            stemmer=stemmer, ngram_range=ngram_range, stop_words=stop_words
        )

        if ngram_range == 1:
            # Map stems to original forms for representative tokens
            for stemmed_token, orig in zip([term.split("_")[0] for term in terms], original_tokens):
                if stemmed_token not in stem2original:
                    stem2original[stemmed_token] = {}
                stem2original[stemmed_token][orig] = stem2original[stemmed_token].get(orig, 0) + 1

        for t in terms:
            overall_freq[t] = overall_freq.get(t, 0) + 1
            category_freq[cat][t] = category_freq[cat].get(t, 0) + 1
        category_counts[cat] += len(terms)
        total_terms += len(terms)

    # Exclude hapax legomena
    overall_freq = {k: v for k, v in overall_freq.items() if v > 1}
    for cat in categories:
        category_freq[cat] = {k: v for k, v in category_freq[cat].items() if k in overall_freq}

    if ngram_range == 1:
        stem2repr = find_most_frequent_original_forms(stem2original)
    else:
        stem2repr = {}

    # Statistical testing
    all_pvals = []
    all_terms = []
    all_cats = []
    all_x = []
    all_K = []
    all_n = []

    for cat in categories:
        n = category_counts[cat]
        cat_vocab = category_freq[cat]
        for t in cat_vocab:
            x = cat_vocab[t]
            K = overall_freq[t]

            # Hypergeometric test
            pval_over = hypergeom.sf(x-1, total_terms, K, n)
            pval_under = hypergeom.cdf(x, total_terms, K, n)
            pval = min(pval_over, pval_under)

            all_pvals.append(pval)
            all_terms.append(t)
            all_cats.append(cat)
            all_x.append(x)
            all_K.append(K)
            all_n.append(n)

    # Multiple testing correction
    pvals_array = np.array(all_pvals)
    reject, pvals_corrected, _, _ = multipletests(pvals_array, alpha=alpha, method='fdr_bh')

    final_data = []
    for i in range(len(all_terms)):
        t = all_terms[i]
        cat = all_cats[i]
        x = all_x[i]
        K = all_K[i]
        n = all_n[i]
        pval = pvals_corrected[i]

        epsilon = 1e-9
        term_ratio = x / (n + epsilon)
        global_ratio = K / (total_terms + epsilon)
        test_val = math.log2((term_ratio + epsilon) / (global_ratio + epsilon))

        if ngram_range == 1:
            stem = t.split("_")[0]
            term_repr = stem2repr.get(stem, t)
        else:
            term_repr = t

        final_data.append({
            "Category": cat,
            "Term": term_repr,
            "Internal Frequency": x,
            "Global Frequency": K,
            "Test-Value": round(test_val, 4),
            "P-Value": round(pval, 6),
            "Significant": "Yes" if reject[i] else "No"
        })

    result_df = pd.DataFrame(final_data)
    result_df = result_df.sort_values(by=["Category", "P-Value"], ascending=[True, True])
    return result_df, categories, total_terms

def visualize_results(result_df, categories):
    st.write("### 📊 Most Characteristic Words per Category")
    for cat in categories:
        subset = result_df[(result_df['Category'] == cat) & (result_df['Significant'] == "Yes")]
        if subset.empty:
            st.write(f"No significant characteristic words found for category **{cat}**.")
            continue
        subset = subset.reindex(subset['Test-Value'].abs().sort_values(ascending=False).index)
        top_subset = subset.head(10)

        fig = px.bar(
            top_subset,
            x="Test-Value",
            y="Term",
            orientation='h',
            title=f"Top Characteristic Words for Category: {cat}",
            labels={"Test-Value": "Test Value", "Term": "Word"},
            height=400
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

def display_results(result_df, total_terms, categories, alpha):
    st.write("### 📄 Characteristic Words Table")
    st.dataframe(result_df)
    visualize_results(result_df, categories)
    st.write("### 📈 Summary Statistics")
    st.write(f"**Total Terms in Corpus (excluding hapax):** {total_terms}")
    st.write(f"**Number of Categories:** {len(categories)}")
    st.write(f"**Significance Level (alpha):** {alpha}")

def download_results(result_df):
    st.write("### ⬇️ Download Results")
    csv_buffer = StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="characteristic_words.csv",
        mime="text/csv"
    )
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        result_df.to_excel(writer, index=False, sheet_name='Characteristic Words')
    st.download_button(
        label="📥 Download Results as Excel",
        data=excel_buffer.getvalue(),
        file_name="characteristic_words.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def main():
    st.set_page_config(page_title="Characteristic Words Detection", layout="wide")
    st.title("📊 Characteristic Words Detection in Corpus Linguistics")

    st.markdown("""
    **Introduction**

    Corpus linguistics involves analyzing large text collections to understand language use. 
    **Characteristic words** appear unusually often or rarely in specific text subsets compared to the whole corpus, helping uncover themes and distinctions.

    (Lebart, L., Salem, A., & Berry, L. (1997). *Exploring textual data*. Springer.)
    """)

    st.sidebar.header("🔧 Configuration")
    uploaded_file = st.sidebar.file_uploader("📂 Upload Your Data", type=["csv", "xlsx", "tsv", "txt"])
    
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
            st.sidebar.write("### Data Preview:")
            st.sidebar.dataframe(df.head())

            text_col = st.sidebar.selectbox("Select the text column", options=df.columns)
            category_col = st.sidebar.selectbox("Select the category column", options=df.columns)

            st.sidebar.write("### Stopword Removal")
            remove_sw = st.sidebar.checkbox("🗑️ Remove stopwords?", value=False)
            if remove_sw:
                lang_choice = st.sidebar.selectbox("🌐 Select language for stopwords", list(LANGUAGES.keys()))
            else:
                lang_choice = "English"

            st.sidebar.write("### N-gram Selection")
            ngram_option = st.sidebar.radio("Select N-grams to consider", 
                                            ["Unigrams only", "Unigrams + Bigrams", "Unigrams + Bigrams + Trigrams"])
            if ngram_option == "Unigrams only":
                ngram_range = 1
            elif ngram_option == "Unigrams + Bigrams":
                ngram_range = 2
            else:
                ngram_range = 3

            alpha = st.sidebar.number_input("📉 Significance level (alpha)", min_value=0.0001, max_value=0.5, value=0.05, step=0.01)
            
            if st.sidebar.button("🚀 Run Analysis"):
                st.header("🔍 Analysis Results")
                st.write("### Processing...")
                with st.spinner("🕒 Analyzing the corpus..."):
                    result_df, categories, total_terms = perform_analysis(
                        df,
                        text_col,
                        category_col,
                        remove_sw,
                        lang_choice,
                        ngram_range,
                        alpha
                    )
                    display_results(result_df, total_terms, categories, alpha)
                    download_results(result_df)

        except ValueError as ve:
            st.sidebar.error(f"⚠️ {ve}")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error processing the uploaded file: {e}")
    else:
        st.sidebar.info("📥 Awaiting file upload.")

if __name__ == "__main__":
    main()
    add_footer()
