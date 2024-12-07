import streamlit as st
import pandas as pd
import numpy as np
import re
from stop_words import get_stop_words
from snowballstemmer import stemmer
from scipy.stats import hypergeom
from io import StringIO, BytesIO
import math
import plotly.express as px

# Define supported languages and their corresponding stopword lists and stemmers
LANGUAGES = {
    "English": {"stopwords": get_stop_words("english"), "stemmer": stemmer("english")},
    "French": {"stopwords": get_stop_words("french"), "stemmer": stemmer("french")},
    "Italian": {"stopwords": get_stop_words("italian"), "stemmer": stemmer("italian")},
    "Spanish": {"stopwords": get_stop_words("spanish"), "stemmer": stemmer("spanish")}
}

def preprocess_text(text, lang, remove_stopwords, stem_words, stemmer_obj, ngram_ranges, stop_words):
    """
    Tokenizes, removes stopwords, stems, and generates n-grams from the input text.
    """
    # Tokenization using regex to extract words
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stopwords if requested
    if remove_stopwords and stop_words is not None:
        tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming if requested
    if stem_words:
        stemmed_tokens = [stemmer_obj.stemWord(token) for token in tokens]
    else:
        stemmed_tokens = tokens.copy()
    
    # Generate n-grams based on selected ranges
    final_terms = []
    for n in ngram_ranges:
        if n > len(stemmed_tokens):
            continue
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

def benjamini_hochberg_correction(pvals, alpha=0.05):
    """
    Performs the Benjamini-Hochberg FDR correction on a list of p-values.
    Returns a list indicating which hypotheses are rejected.
    """
    n = len(pvals)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = np.array(pvals)[sorted_indices]
    thresholds = (np.arange(1, n+1) / n) * alpha
    below_threshold = sorted_pvals <= thresholds
    if not np.any(below_threshold):
        return np.zeros(n, dtype=bool)
    max_idx = np.max(np.where(below_threshold))
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_indices[:max_idx+1]] = True
    return rejected

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

@st.cache_data
def load_data(uploaded_file):
    """
    Loads data from the uploaded file based on its extension.
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'csv':
            return pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(uploaded_file)
        elif file_extension == 'tsv':
            return pd.read_csv(uploaded_file, sep='\t')
        elif file_extension == 'txt':
            return pd.read_csv(uploaded_file, sep='\n', header=None, names=['text'])
        else:
            raise ValueError("Unsupported file type. Please upload a CSV, XLSX, TSV, or TXT file.")
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

def perform_analysis(df, text_col, category_col, remove_sw, stem_words, chosen_lang, ngram_ranges, min_freq, alpha, custom_stopwords, progress):
    """
    Performs the characteristic words analysis and returns the result DataFrame.
    Includes progress updates.
    """
    # Update progress: Retrieval of stopwords and stemmer
    progress.progress(10)
    # Retrieve stopwords and stemmer based on selected language
    stop_words = LANGUAGES[chosen_lang]["stopwords"] if remove_sw else []
    stemmer_obj = LANGUAGES[chosen_lang]["stemmer"] if stem_words else None
    
    # Add custom stopwords if provided
    if custom_stopwords:
        stop_words.extend(custom_stopwords)

    overall_freq = {}
    category_freq = {}
    category_counts = {}
    total_terms = 0

    stem2original = {}
    categories = df[category_col].dropna().unique()

    # Initialize frequency dictionaries
    for cat in categories:
        category_freq[cat] = {}
        category_counts[cat] = 0

    # Initialize word frequency for summary stats
    word_freq = {}

    # Preprocessing and frequency counting
    # Update progress: Start processing texts
    progress.progress(25)
    for idx, row in df.iterrows():
        cat = row[category_col]
        text = str(row[text_col])
        terms, tokens = preprocess_text(
            text,
            lang=chosen_lang,
            remove_stopwords=remove_sw,
            stem_words=stem_words,
            stemmer_obj=stemmer_obj,
            ngram_ranges=ngram_ranges,
            stop_words=stop_words
        )

        # Update word frequency
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1

        # Generate selected n-grams
        selected_terms = []
        for n in ngram_ranges:
            if n > len(tokens):
                continue
            for i in range(len(tokens) - n + 1):
                if stem_words:
                    ngram = "_".join([stemmer_obj.stemWord(token) for token in tokens[i:i+n]])
                else:
                    ngram = "_".join(tokens[i:i+n])
                selected_terms.append(ngram)

        # If unigrams are included and stemming is applied, map stems to original forms
        if 1 in ngram_ranges and stem_words:
            for stemmed_token, orig in zip(
                [term.split("_")[0] for term in selected_terms if '_' not in term],
                tokens
            ):
                if stemmed_token not in stem2original:
                    stem2original[stemmed_token] = {}
                stem2original[stemmed_token][orig] = stem2original[stemmed_token].get(orig, 0) + 1

        # Update frequency counts
        for t in selected_terms:
            overall_freq[t] = overall_freq.get(t, 0) + 1
            category_freq[cat][t] = category_freq[cat].get(t, 0) + 1
        category_counts[cat] += len(selected_terms)
        total_terms += len(selected_terms)

    # Update progress: Filtering frequencies
    progress.progress(40)
    # Exclude hapax legomena (words with global frequency = 1) and apply minimum frequency
    overall_freq_filtered = {k: v for k, v in overall_freq.items() if v > 1 and v >= min_freq}
    for cat in categories:
        category_freq[cat] = {k: v for k, v in category_freq[cat].items() if k in overall_freq_filtered}

    if 1 in ngram_ranges and stem_words:
        stem2repr = find_most_frequent_original_forms(stem2original)
    else:
        stem2repr = {}

    # Statistical testing
    # Update progress: Statistical analysis
    progress.progress(60)
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
            K = overall_freq_filtered[t]

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

    # Multiple testing correction using Benjamini-Hochberg
    # Update progress: Multiple testing correction
    progress.progress(75)
    rejected = benjamini_hochberg_correction(all_pvals, alpha=alpha)

    # Compile results
    # Update progress: Compiling results
    progress.progress(85)
    final_data = []
    for i in range(len(all_terms)):
        t = all_terms[i]
        cat = all_cats[i]
        x = all_x[i]
        K = all_K[i]
        n = all_n[i]
        pval = all_pvals[i]

        # Compute test-value
        epsilon = 1e-9
        term_ratio = x / (n + epsilon)
        global_ratio = K / (total_terms + epsilon)
        test_val = math.log2((term_ratio + epsilon) / (global_ratio + epsilon))

        if 1 in ngram_ranges and stem_words:
            stem = t.split("_")[0]
            term_repr = stem2repr.get(stem, t)
        else:
            term_repr = t

        # Replace underscores with spaces for display
        term_repr_display = term_repr.replace("_", " ")

        # Format P-Value
        if pval < 0.001:
            pval_formatted = "<.001"
        else:
            pval_formatted = f"{pval:.3f}".lstrip('0') if pval >= 0.001 else "<.001"

        # Include only significant words
        if rejected[i]:
            final_data.append({
                "Category": cat,
                "Term": term_repr_display,
                "Internal Frequency": x,
                "Global Frequency": K,
                "Test Value": round(test_val, 4),
                "P-Value": pval_formatted
            })

    result_df = pd.DataFrame(final_data)
    result_df = result_df.sort_values(by=["Category", "P-Value"], ascending=[True, True])

    # Compute summary statistics
    num_categories = len(categories)
    total_tokens = sum(word_freq.values())
    total_types = len(word_freq)
    morphological_complexity = round(total_types / total_tokens, 2) if total_tokens > 0 else 0.00
    num_hapax = sum(1 for freq in word_freq.values() if freq == 1)

    # Compute per-category statistics
    category_stats = {}
    for cat in categories:
        cat_df = df[df[category_col] == cat]
        num_instances = len(cat_df)
        # Calculate tokens and types within the category
        cat_tokens = 0
        cat_types_set = set()
        for text in cat_df[text_col].dropna().astype(str):
            tokens = re.findall(r'\b\w+\b', text.lower())
            if remove_sw and stop_words:
                tokens = [token for token in tokens if token not in stop_words]
            if stem_words:
                tokens = [stemmer_obj.stemWord(token) for token in tokens]
            cat_tokens += len(tokens)
            cat_types_set.update(tokens)
        cat_types = len(cat_types_set)
        cat_morph_complexity = round(cat_types / cat_tokens, 2) if cat_tokens > 0 else 0.00
        cat_num_hapax = sum(1 for token in cat_types_set if word_freq.get(token, 0) == 1)
        category_stats[cat] = {
            "Number of Instances": num_instances,
            "Number of Tokens": cat_tokens,
            "Number of Types": cat_types,
            "Morphological Complexity": cat_morph_complexity,
            "Number of Hapax": cat_num_hapax
        }

    # Update progress: Done
    progress.progress(100)
    return result_df, categories, num_categories, total_tokens, total_types, morphological_complexity, num_hapax, category_stats

def visualize_results(result_df, categories):
    """
    Generates horizontal bar plots for the most characteristic words per category.
    """
    st.write("### 📊 Most Characteristic Words per Category")
    for cat in categories:
        subset = result_df[result_df['Category'] == cat]
        if subset.empty:
            st.write(f"No significant characteristic words found for category **{cat}**.")
            continue
        # Select top 10 based on absolute test value
        subset = subset.reindex(subset['Test Value'].abs().sort_values(ascending=False).index)
        top_subset = subset.head(10)

        fig = px.bar(
            top_subset,
            x="Test Value",
            y="Term",
            orientation='h',
            title=f"Top Characteristic Words for Category: {cat}",
            labels={"Test Value": "Test Value (log2((x/n)/(K/M)))", "Term": "Word"},
            height=400
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

def display_results(result_df, total_tokens, num_categories, total_types, morphological_complexity, num_hapax, alpha, category_stats):
    """
    Displays the summary statistics and the results table.
    """
    # Summary statistics at the top
    st.write("### 📈 Summary Statistics")
    st.markdown(f"""
    - **Number of Categories:** {num_categories}
    - **Total Number of Terms (Tokens):** {total_tokens}
    - **Total Number of Unique Words (Types):** {total_types}
    - **Morphological Complexity (Types/Token Ratio):** {morphological_complexity}
    - **Number of Hapax (Words that appear only once):** {num_hapax}
    - **Significance Level (alpha):** {alpha}
    """)

    st.write("### 📄 Characteristic Words Table")
    st.markdown("""
    **Table Explanation:**
    - **Category:** The specific group or category within your dataset.
    - **Term:** The characteristic word or n-gram.
    - **Internal Frequency:** Number of times the term appears within the category.
    - **Global Frequency:** Number of times the term appears across the entire corpus.
    - **Test Value:** Calculated as log2((x/n)/(K/M)) where:
        - *x* = Internal Frequency
        - *n* = Total terms in the category
        - *K* = Global Frequency
        - *M* = Total terms in the corpus
      This value indicates how much more (or less) the term is associated with the category compared to the corpus.
    - **P-Value:** The probability of observing the term's frequency in the category by chance. Displayed with three decimal points (e.g., .035) or as "<.001" if p < .001.
    """)

    if result_df.empty:
        st.warning("No significant characteristic words found based on the provided criteria.")
    else:
        # Group by category and display separate sortable tables
        for cat in sorted(result_df['Category'].unique()):
            st.subheader(f"Category: {cat}")
            # Display category-specific statistics
            stats = category_stats.get(cat, {})
            st.markdown(f"""
            - **Number of Instances (Documents):** {stats.get('Number of Instances', 0)}
            - **Number of Tokens:** {stats.get('Number of Tokens', 0)}
            - **Number of Types:** {stats.get('Number of Types', 0)}
            - **Morphological Complexity (Types/Token Ratio):** {stats.get('Morphological Complexity', 0.00)}
            - **Number of Hapax (Words that appear only once):** {stats.get('Number of Hapax', 0)}
            """)
            cat_df = result_df[result_df['Category'] == cat][['Term', 'Internal Frequency', 'Global Frequency', 'Test Value', 'P-Value']]
            st.dataframe(cat_df.reset_index(drop=True), use_container_width=True)

    # Visualization
    visualize_results(result_df, sorted(result_df['Category'].unique()))

def download_results(result_df):
    """
    Provides download buttons for CSV and Excel formats.
    """
    st.write("### ⬇️ Download Results")
    if result_df.empty:
        st.info("No results to download.")
        return
    # CSV download
    csv_buffer = StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="characteristic_words.csv",
        mime="text/csv"
    )
    # Excel download
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
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Characteristic Words Detection", layout="wide")
    st.title("📊 Characteristic Words Detection in Corpus Linguistics")

    st.markdown("""
    **Introduction**
    
    Corpus linguistics involves analyzing large text collections to understand language use. 
    Characteristic words appear unusually often or rarely in specific text subsets compared to the whole corpus, helping uncover themes and distinctions.
    
    *Lebart, L., Salem, A., & Berry, L. (1997). _Exploring textual data_. Springer.*
    """)

    st.markdown("""
    **Overview**
    
    This app allows you to identify characteristic words within different categories of your dataset. By analyzing the frequency of words (or n-grams) in specific categories compared to the entire corpus, you can uncover terms that are uniquely associated with each category. The app provides statistical significance testing to ensure that the identified words are not occurring by chance.
    """)

    st.sidebar.header("🔧 Configuration")
    uploaded_file = st.sidebar.file_uploader("📂 Upload Your Data", type=["csv", "xlsx", "tsv", "txt"])

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
            # Show data shape instead of preview
            num_instances, num_features = df.shape
            st.sidebar.write("### Data Preview:")
            st.sidebar.write(f"**Shape:** {num_instances} instances, {num_features} features")

            # Column Selection
            text_col = st.sidebar.selectbox("Select the text column", options=df.columns)
            category_col = st.sidebar.selectbox("Select the category column", options=df.columns)

            # Word Exclusion Section
            st.sidebar.write("### Word Exclusion")
            # Custom Stopword List Options
            custom_stopword_option = st.sidebar.radio(
                "Choose how to provide custom stopwords:",
                options=["None", "Type custom stopwords", "Upload custom stopwords file"],
                index=0
            )
            
            custom_stopwords = []
            
            if custom_stopword_option == "Type custom stopwords":
                separator = st.sidebar.text_input("📝 Enter a custom separator (e.g., comma, space):", value=",")
                custom_stopword_input = st.sidebar.text_input("✍️ Enter custom stopwords separated by your chosen separator:")
                if custom_stopword_input:
                    custom_stopwords = [word.strip() for word in custom_stopword_input.split(separator) if word.strip()]
            
            elif custom_stopword_option == "Upload custom stopwords file":
                uploaded_stopword_file = st.sidebar.file_uploader("📂 Upload a custom stopwords file (.txt):", type=["txt"])
                separator = st.sidebar.text_input("📝 Enter the separator used in your file (e.g., comma, space):", value=",")
                if uploaded_stopword_file and separator:
                    stopword_content = uploaded_stopword_file.read().decode('utf-8')
                    custom_stopwords = [word.strip() for word in stopword_content.split(separator) if word.strip()]

            # Stopword Removal Section
            st.sidebar.write("### Stopword Removal")
            remove_sw = st.sidebar.checkbox("🗑️ Remove stopwords?", value=False)
            if remove_sw:
                lang_choice = st.sidebar.selectbox("🌐 Select language for stopwords", list(LANGUAGES.keys()))
            else:
                lang_choice = "English"  # Default language if stopword removal is not selected

            # Stemming Option
            st.sidebar.write("### Stemming")
            stem_words = st.sidebar.checkbox("🪓 Apply stemming?", value=True)

            # N-gram Selection with Multiple Choices
            st.sidebar.write("### N-gram Selection")
            ngram_options = st.sidebar.multiselect(
                "Select N-grams to consider",
                options=["Unigrams", "Bigrams", "Trigrams"],
                default=["Unigrams"]
            )
            ngram_mapping = {"Unigrams": 1, "Bigrams": 2, "Trigrams": 3}
            ngram_ranges = sorted([ngram_mapping[ngram] for ngram in ngram_options])

            if not ngram_ranges:
                st.sidebar.error("⚠️ Please select at least one N-gram option.")

            # Minimum Frequency Selection
            st.sidebar.write("### Minimum Frequency")
            min_freq = st.sidebar.number_input(
                "🔢 Set the minimum frequency for words to be considered in the analysis:",
                min_value=1,
                max_value=1000,
                value=1,
                step=1
            )

            # Significance Level
            st.sidebar.write("### Significance Level")
            alpha = st.sidebar.number_input("📉 Significance level (alpha)", min_value=0.0001, max_value=0.5, value=0.05, step=0.01)

            # Run Analysis Button
            if st.sidebar.button("🚀 Run Analysis"):
                if not ngram_ranges:
                    st.error("⚠️ Please select at least one N-gram option to proceed.")
                else:
                    st.header("🔍 Analysis Results")
                    st.write("### Processing...")

                    # Create a progress bar
                    progress = st.progress(0)

                    with st.spinner("🕒 Analyzing the corpus..."):
                        result_df, categories, num_categories, total_tokens, total_types, morphological_complexity, num_hapax, category_stats = perform_analysis(
                            df,
                            text_col,
                            category_col,
                            remove_sw,
                            stem_words,
                            lang_choice,
                            ngram_ranges,
                            min_freq,
                            alpha,
                            custom_stopwords,
                            progress
                        )
                        display_results(result_df, total_tokens, num_categories, total_types, morphological_complexity, num_hapax, alpha, category_stats)
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
