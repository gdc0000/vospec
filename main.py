import streamlit as st
import pandas as pd
import re
import math
from scipy.stats import hypergeom
from io import BytesIO
import plotly.express as px

# Define supported languages and their corresponding stopword lists
LANGUAGES = {
    "English": {"stopwords": ["the", "and", "is", "in", "to", "of"]},  # Extend as needed
    "French": {"stopwords": ["le", "et", "est", "dans", "à", "de"]},
    "Italian": {"stopwords": ["il", "e", "è", "in", "a", "di"]},
    "Spanish": {"stopwords": ["el", "y", "es", "en", "a", "de"]}
}

# Preprocess text: Tokenizes and removes stopwords
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

# Perform the analysis
def perform_analysis(df, text_col, category_col, stop_words, word_group_mapping):
    # Apply word grouping
    df[text_col] = df[text_col].apply(lambda x: apply_word_grouping(str(x), word_group_mapping))
    
    overall_freq = {}
    category_freq = {}
    category_counts = {}
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

    return overall_freq, category_freq, category_counts

# Download results as Excel
def download_results(overall_freq, category_freq):
    with BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Save overall frequency table
            pd.DataFrame(list(overall_freq.items()), columns=["Word", "Frequency"]).to_excel(writer, sheet_name="Corpus Frequency", index=False)
            # Save category frequency tables
            for category, freq in category_freq.items():
                pd.DataFrame(list(freq.items()), columns=["Word", "Frequency"]).to_excel(writer, sheet_name=f"{category}_Words", index=False)
        return buffer.getvalue()

# Streamlit app
def main():
    st.set_page_config(page_title="Word Analysis", layout="wide")
    st.title("📊 Word Analysis in Corpus Linguistics")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload a Dataset (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")

        # Select columns
        text_col = st.sidebar.selectbox("Select the text column", df.columns)
        category_col = st.sidebar.selectbox("Select the category column", df.columns)

        # Word grouping
        word_group_file = st.sidebar.file_uploader("Upload Word Group Mapping (CSV/Excel)", type=["csv", "xlsx"])
        word_group_mapping = {}
        if word_group_file:
            if word_group_file.name.endswith('.csv'):
                word_group_df = pd.read_csv(word_group_file)
            else:
                word_group_df = pd.read_excel(word_group_file)
            word_group_mapping = dict(zip(word_group_df["word"], word_group_df["group"]))

        # Stopword removal
        remove_stopwords = st.sidebar.checkbox("Remove Stopwords?", value=True)
        lang_choice = st.sidebar.selectbox("Select Stopword Language", options=LANGUAGES.keys()) if remove_stopwords else None
        stop_words = LANGUAGES[lang_choice]["stopwords"] if remove_stopwords and lang_choice else None

        # Run analysis
        if st.sidebar.button("Run Analysis"):
            with st.spinner("Processing..."):
                overall_freq, category_freq, category_counts = perform_analysis(df, text_col, category_col, stop_words, word_group_mapping)

                # Display results
                st.subheader("Corpus-Wide Word Frequencies")
                st.dataframe(pd.DataFrame(list(overall_freq.items()), columns=["Word", "Frequency"]))

                for category, freq in category_freq.items():
                    st.subheader(f"Category: {category}")
                    st.dataframe(pd.DataFrame(list(freq.items()), columns=["Word", "Frequency"]))

                # Download results
                st.download_button(
                    label="Download Results as Excel",
                    data=download_results(overall_freq, category_freq),
                    file_name="word_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
