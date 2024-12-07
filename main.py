import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.stats import hypergeom
import math
from io import BytesIO
import plotly.express as px

# Preprocess text: Tokenize and remove stopwords
def preprocess_text(text, stop_words):
    tokens = re.findall(r'\b\w+\b', text.lower())
    if stop_words:
        tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Apply word grouping: Replace words based on a mapping
def apply_word_grouping(text, word_group_mapping):
    if not word_group_mapping:
        return text
    for word, group in word_group_mapping.items():
        text = re.sub(rf'\b{re.escape(word)}\b', group, text, flags=re.IGNORECASE)
    return text

# Perform characteristic words analysis
def perform_analysis(df, text_col, category_col, stop_words, word_group_mapping):
    # Apply word grouping
    df[text_col] = df[text_col].apply(lambda x: apply_word_grouping(str(x), word_group_mapping))

    overall_freq = {}
    category_freq = {}
    category_counts = {}
    total_terms = 0
    categories = df[category_col].unique()

    # Calculate frequencies
    for category in categories:
        texts = df[df[category_col] == category][text_col]
        category_terms = []
        for text in texts:
            tokens = preprocess_text(text, stop_words)
            category_terms.extend(tokens)
            for token in tokens:
                overall_freq[token] = overall_freq.get(token, 0) + 1
        category_counts[category] = len(category_terms)
        category_freq[category] = pd.Series(category_terms).value_counts().to_dict()
        total_terms += category_counts[category]

    # Characteristic words computation
    all_results = []
    for category in categories:
        category_terms = category_freq[category]
        for term, freq_in_category in category_terms.items():
            global_freq = overall_freq.get(term, 0)
            category_size = category_counts[category]
            pval = hypergeom.sf(freq_in_category - 1, total_terms, global_freq, category_size)
            test_value = math.log2((freq_in_category / category_size) / (global_freq / total_terms)) if global_freq > 0 else 0
            all_results.append({
                "Category": category,
                "Word": term,
                "Frequency (Category)": freq_in_category,
                "Frequency (Global)": global_freq,
                "Test Value": round(test_value, 4),
                "P-Value": f"{pval:.4f}" if pval >= 0.0001 else "<0.0001"
            })

    return pd.DataFrame(all_results), overall_freq

# Generate downloadable Excel file
def generate_excel_file(overall_freq, results_df):
    with BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Corpus-wide frequency
            pd.DataFrame(list(overall_freq.items()), columns=["Word", "Frequency"]).to_excel(writer, sheet_name="Corpus Frequency", index=False)
            # Characteristic words
            results_df.to_excel(writer, sheet_name="Characteristic Words", index=False)
        return buffer.getvalue()

# Streamlit app
def main():
    st.set_page_config(page_title="Characteristic Words Analysis", layout="wide")
    st.title("📊 Characteristic Words Analysis in Corpus Linguistics")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload a Dataset (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        # Load dataset
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.sidebar.success("✅ File uploaded successfully!")
        st.sidebar.write("### Dataset Preview")
        st.sidebar.dataframe(df.head())

        # Select text and category columns
        text_col = st.sidebar.selectbox("Select the text column", options=df.columns)
        category_col = st.sidebar.selectbox("Select the category column", options=df.columns)

        # Word grouping
        word_group_file = st.sidebar.file_uploader("Upload Word Group Mapping (CSV/Excel)", type=["csv", "xlsx"])
        word_group_mapping = {}
        if word_group_file:
            if word_group_file.name.endswith(".csv"):
                word_group_df = pd.read_csv(word_group_file)
            else:
                word_group_df = pd.read_excel(word_group_file)
            word_group_mapping = dict(zip(word_group_df["word"], word_group_df["group"]))

        # Stopword removal
        remove_stopwords = st.sidebar.checkbox("Remove Stopwords?", value=True)
        stop_words = []
        if remove_stopwords:
            stop_words_input = st.sidebar.text_area("Enter Stopwords (comma-separated)", value="")
            if stop_words_input:
                stop_words = [word.strip() for word in stop_words_input.split(",")]

        # Run analysis
        if st.sidebar.button("Run Analysis"):
            with st.spinner("Processing..."):
                results_df, overall_freq = perform_analysis(df, text_col, category_col, stop_words, word_group_mapping)

                # Display results
                st.subheader("Corpus-Wide Word Frequencies")
                st.dataframe(pd.DataFrame(list(overall_freq.items()), columns=["Word", "Frequency"]))

                st.subheader("Characteristic Words")
                st.dataframe(results_df)

                # Download results
                excel_data = generate_excel_file(overall_freq, results_df)
                st.download_button(
                    label="Download Results as Excel",
                    data=excel_data,
                    file_name="characteristic_words_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
