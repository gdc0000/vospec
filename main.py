import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from io import StringIO
import math

# Download NLTK data if not present
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Supported languages for stopwords and stemming
LANGUAGES = {
    "English": "english",
    "French": "french",
    "Italian": "italian",
    "Spanish": "spanish"
}

def preprocess_text(text, lang, remove_stopwords, stemmer, ngram_range, stop_words):
    """
    Tokenizes, removes stopwords, stems, and generates n-grams from the input text.
    """
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Keep only alphabetic tokens
    tokens = [t for t in tokens if t.isalpha()]

    # Remove stopwords if requested
    if remove_stopwords and stop_words is not None:
        tokens = [t for t in tokens if t not in stop_words]

    # Stem tokens
    stemmed_tokens = [stemmer.stem(t) for t in tokens]

    # Generate n-grams
    # ngram_range = 1 → unigrams only
    # ngram_range = 2 → unigrams + bigrams
    # ngram_range = 3 → unigrams + bigrams + trigrams
    final_terms = []
    for n in range(1, ngram_range + 1):
        for i in range(len(stemmed_tokens) - n + 1):
            ngram = "_".join(stemmed_tokens[i:i+n])
            final_terms.append(ngram)
    return final_terms, tokens

def find_most_frequent_original_forms(stem2original):
    """
    For each stem, find the most frequent original word form.
    """
    stem2repr = {}
    for stem, counts in stem2original.items():
        # Pick the original form with highest frequency
        repr_word = max(counts.items(), key=lambda x: x[1])[0]
        stem2repr[stem] = repr_word
    return stem2repr

def add_footer():
    """
    Adds a footer with personal information and social links.
    """
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown("""
    [GitHub](https://github.com/gdc0000) | 
    [ORCID](https://orcid.org/0000-0002-1439-5790) | 
    [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """)

def main():
    st.title("Characteristic Words Detection")

    st.write("""
    This application detects characteristic words in different segments of your corpus based on a categorical grouping. 
    Upload your data, configure the preprocessing options, and run the analysis to obtain insightful results.
    """)

    st.write("### Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("#### Data Preview:")
            st.dataframe(df.head())

            # Ensure required columns exist
            if df.empty:
                st.error("The uploaded CSV is empty.")
                return

            # Let user select text and category columns
            st.write("### Configuration")
            text_col = st.selectbox("Select the text column", df.columns)
            category_col = st.selectbox("Select the category column", df.columns)

            # Choose stopword removal
            remove_sw = st.checkbox("Remove stopwords?", value=False)
            lang_choice = None
            if remove_sw:
                lang_choice = st.selectbox("Select language for stopwords", list(LANGUAGES.keys()))
                chosen_lang = LANGUAGES[lang_choice]
                try:
                    stop_words = set(stopwords.words(chosen_lang))
                except Exception as e:
                    st.error(f"Error loading stopwords for {lang_choice}: {e}")
                    stop_words = None
            else:
                stop_words = None
                chosen_lang = "english"  # Default for stemmer if no stopwords chosen

            # Choose n-gram range
            ngram_option = st.radio("N-grams to consider", 
                                    ["Unigrams only", "Unigrams + Bigrams", "Unigrams + Bigrams + Trigrams"])
            if ngram_option == "Unigrams only":
                ngram_range = 1
            elif ngram_option == "Unigrams + Bigrams":
                ngram_range = 2
            else:
                ngram_range = 3

            # Stemmer
            # If user selected a language for stopwords, we also use that language for stemming.
            # If no stopwords chosen, default to English stemmer.
            stemmer_lang = chosen_lang if chosen_lang in LANGUAGES.values() else "english"
            try:
                stemmer = SnowballStemmer(stemmer_lang)
            except ValueError as ve:
                st.error(f"Stemming not supported for language '{stemmer_lang}': {ve}")
                stemmer = SnowballStemmer("english")

            # Significance level
            alpha = st.number_input("Significance level (alpha)", min_value=0.0001, max_value=0.5, value=0.05, step=0.01)
            
            # Run Analysis Button
            if st.button("Run Analysis"):
                st.write("### Processing...")
                with st.spinner("Analyzing the corpus..."):
                    # Initialize frequency dictionaries
                    overall_freq = {}
                    category_freq = {}
                    category_counts = {}
                    total_terms = 0

                    # For unigrams only: map stem to original words
                    stem2original = {}

                    categories = df[category_col].dropna().unique()
                    for cat in categories:
                        category_freq[cat] = {}
                        category_counts[cat] = 0

                    for idx, row in df.iterrows():
                        cat = row[category_col]
                        text = str(row[text_col])
                        terms, original_tokens = preprocess_text(
                            text, lang=stemmer_lang, remove_stopwords=remove_sw,
                            stemmer=stemmer, ngram_range=ngram_range, stop_words=stop_words
                        )

                        # If unigrams only, track original forms for each stem
                        if ngram_range == 1:
                            # Map each stemmed token to original tokens to find representatives
                            for stemmed_token, orig in zip([term.split("_")[0] for term in terms], original_tokens):
                                if stemmed_token not in stem2original:
                                    stem2original[stemmed_token] = {}
                                stem2original[stemmed_token][orig] = stem2original[stemmed_token].get(orig, 0) + 1

                        for t in terms:
                            overall_freq[t] = overall_freq.get(t, 0) + 1
                            category_freq[cat][t] = category_freq[cat].get(t, 0) + 1
                        category_counts[cat] += len(terms)
                        total_terms += len(terms)

                    # If unigrams only, replace stems with most frequent original forms in final results
                    if ngram_range == 1:
                        stem2repr = find_most_frequent_original_forms(stem2original)
                    else:
                        stem2repr = {}  # Not used for n-grams

                    # Perform Hypergeometric tests
                    # M = total_terms
                    M = total_terms
                    results = []

                    # Prepare lists for multiple testing correction
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
                            # Probability of seeing at least x occurrences:
                            pval_over = hypergeom.sf(x-1, M, K, n)
                            # Probability of seeing at most x occurrences:
                            pval_under = hypergeom.cdf(x, M, K, n)
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

                    # Compute test-value = log2((x/n)/(K/M))
                    # Avoid division by zero:
                    # If x=0, (x/n) = 0, test-value = large negative
                    # If K=0 (can't happen after processing), skip.
                    final_data = []
                    for i in range(len(all_terms)):
                        t = all_terms[i]
                        cat = all_cats[i]
                        x = all_x[i]
                        K = all_K[i]
                        n = all_n[i]
                        pval = pvals_corrected[i]

                        # Compute test-value
                        # Add a small epsilon to avoid division by zero errors
                        epsilon = 1e-9
                        term_ratio = (x / (n + epsilon))
                        global_ratio = (K / (M + epsilon))
                        test_val = math.log2((term_ratio + epsilon) / (global_ratio + epsilon))

                        # Get representative form if unigrams only
                        if ngram_range == 1 and t in stem2repr:
                            term_repr = stem2repr[t]
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

                    # Sort results by category and p-value
                    result_df = result_df.sort_values(by=["Category", "P-Value"], ascending=[True, True])

                    st.write("### Results")
                    st.dataframe(result_df)

                    # Provide summary statistics
                    st.write("### Summary Statistics")
                    st.write(f"**Total Terms in Corpus:** {M}")
                    st.write(f"**Number of Categories:** {len(categories)}")
                    st.write(f"**Significance Level (alpha):** {alpha}")

                    # Download button
                    csv_buffer = StringIO()
                    result_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="characteristic_words.csv",
                        mime="text/csv"
                    )

        else:
            st.info("Awaiting CSV file to be uploaded.")

    # Run the main function
    main()

    # Add the footer at the end
    add_footer()
