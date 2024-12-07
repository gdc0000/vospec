import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from scipy.stats import hypergeom
from io import BytesIO
import plotly.express as px

# 1. Set page configuration as the very first Streamlit command
st.set_page_config(page_title="Characteristic Words Detection", layout="wide")

# 2. Define necessary functions and decorators below
@st.cache_resource
def initialize_nltk():
    """
    Downloads necessary NLTK data and initializes the WordNetLemmatizer.
    """
    download('stopwords')
    download('wordnet')
    return WordNetLemmatizer()

lemmatizer = initialize_nltk()

# Define supported languages and their corresponding stopword lists
LANGUAGES = {
    "English": {"stopwords": stopwords.words("english")},
    "French": {"stopwords": stopwords.words("french")},
    "Italian": {"stopwords": stopwords.words("italian")},
    "Spanish": {"stopwords": stopwords.words("spanish")}
}

def preprocess_text(text, lang, remove_stopwords, lemmatize, lemmatizer_obj, ngram_ranges, stop_words):
    """
    Tokenizes, removes stopwords (if requested), lemmatizes (if enabled),
    and generates n-grams from the input text.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())

    # Remove stopwords if requested
    if remove_stopwords and stop_words:
        tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize if requested
    if lemmatize and lemmatizer_obj is not None:
        tokens = [lemmatizer_obj.lemmatize(token) for token in tokens]

    # Generate n-grams based on selected ranges
    final_terms = []
    for n in ngram_ranges:
        if n > len(tokens):
            continue
        for i in range(len(tokens) - n + 1):
            if n == 1:
                term = tokens[i]
            else:
                term = "_".join(tokens[i:i+n])
            final_terms.append(term)

    return final_terms, tokens

def benjamini_hochberg_correction(pvals, alpha=0.05):
    """
    Performs the Benjamini-Hochberg FDR correction on a list of p-values.
    Returns a boolean array indicating which hypotheses are rejected.
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

def apply_replacements(text, replacements):
    """
    Applies string replacements to the text based on the replacements dictionary.
    """
    for old, new in replacements.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return text

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

def perform_analysis(df, text_col, category_col, selected_categories, remove_sw, lemmatize, lemmatizer_obj, ngram_ranges, min_freq, alpha, custom_stopwords, replacements, progress, chosen_lang):
    """
    Performs the characteristic words analysis and returns the result DataFrame and frequency inventory.
    Updates the progress bar to indicate analysis steps.
    """
    # Apply Replacements
    progress.progress(5)
    df[text_col] = df[text_col].astype(str).apply(lambda x: apply_replacements(x, replacements))
    
    # Retrieve stopwords based on selected language
    progress.progress(10)
    stop_words = LANGUAGES[chosen_lang]["stopwords"] if remove_sw else []
    
    if custom_stopwords:
        stop_words.extend(custom_stopwords)

    overall_freq = {}
    category_freq = {}
    category_counts = {}
    total_terms = 0
    categories = selected_categories

    for cat in categories:
        category_freq[cat] = {}
        category_counts[cat] = 0

    word_freq = {}
    
    progress.progress(20)
    for idx, row in df.iterrows():
        cat = row[category_col]
        if cat not in categories:
            continue
        text = str(row[text_col])
        terms, tokens = preprocess_text(
            text,
            lang=chosen_lang,
            remove_stopwords=remove_sw,
            lemmatize=lemmatize,
            lemmatizer_obj=lemmatizer_obj,
            ngram_ranges=ngram_ranges,
            stop_words=stop_words
        )

        # Update word frequency
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1

        # Update frequency counts
        for t in terms:
            overall_freq[t] = overall_freq.get(t, 0) + 1
            category_freq[cat][t] = category_freq[cat].get(t, 0) + 1
        category_counts[cat] += len(terms)
        total_terms += len(terms)

    # Frequency Inventory for the Entire Corpus
    progress.progress(35)
    corpus_terms, corpus_tokens = preprocess_text(
        ' '.join(df[text_col].astype(str)),
        lang=chosen_lang,
        remove_stopwords=remove_sw,
        lemmatize=lemmatize,
        lemmatizer_obj=lemmatizer_obj,
        ngram_ranges=[1],
        stop_words=stop_words
    )
    corpus_freq = {}
    for term in corpus_terms:
        corpus_freq[term] = corpus_freq.get(term, 0) + 1
    frequency_inventory = pd.DataFrame(list(corpus_freq.items()), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

    progress.progress(50)
    # Filtering frequencies for characteristic words
    overall_freq_filtered = {k: v for k, v in overall_freq.items() if v >= min_freq}
    for cat in categories:
        category_freq[cat] = {k: v for k, v in category_freq[cat].items() if k in overall_freq_filtered}

    progress.progress(65)
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
            K = corpus_freq.get(t, 0)
            if K == 0:
                continue  # Avoid division by zero
            pval_over = hypergeom.sf(x-1, len(corpus_terms), K, n)
            pval_under = hypergeom.cdf(x, len(corpus_terms), K, n)
            pval = min(pval_over, pval_under)

            all_pvals.append(pval)
            all_terms.append(t)
            all_cats.append(cat)
            all_x.append(x)
            all_K.append(K)
            all_n.append(n)

    progress.progress(80)
    rejected = benjamini_hochberg_correction(all_pvals, alpha=alpha)

    progress.progress(90)
    final_data = []
    for i in range(len(all_terms)):
        t = all_terms[i]
        cat = all_cats[i]
        x = all_x[i]
        K = all_K[i]
        n = all_n[i]
        pval = all_pvals[i]

        epsilon = 1e-9
        term_ratio = x / (n + epsilon)
        global_ratio = K / (len(corpus_terms) + epsilon)
        test_val = math.log2((term_ratio + epsilon) / (global_ratio + epsilon))

        term_repr_display = t.replace("_", " ")

        if pval < 0.001:
            pval_formatted = "<.001"
        else:
            pval_formatted = f"{pval:.3f}".lstrip('0') if pval >= 0.001 else "<.001"

        if rejected[i]:
            final_data.append({
                "Category": cat,
                "Term": term_repr_display,
                "Internal Frequency": x,
                "Global Frequency": K,
                "Test Value": round(test_val, 4),
                "P-Value": pval_formatted
            })

    characteristic_words_df = pd.DataFrame(final_data).sort_values(by=["Category", "P-Value"])

    # Complete the progress bar
    progress.progress(100)
    return characteristic_words_df, frequency_inventory

def visualize_and_display(category, cat_df, category_stats, alpha, top_n=10):
    """
    Displays the characteristic words table and its corresponding bar chart for a given category.
    """
    st.subheader(f"Category: {category}")
    stats = category_stats.get(category, {})
    st.markdown(f"""
    - **Number of Instances (Documents):** {stats.get('Number of Instances', 0)}
    - **Number of Tokens:** {stats.get('Number of Tokens', 0)}
    - **Number of Types:** {stats.get('Number of Types', 0)}
    - **Morphological Complexity (Types/Token Ratio):** {stats.get('Morphological Complexity', 0.00)}
    - **Number of Hapax (Words that appear only once):** {stats.get('Number of Hapax', 0)}
    """)

    display_table = cat_df[['Term', 'Internal Frequency', 'Global Frequency', 'Test Value', 'P-Value']]
    st.dataframe(display_table.reset_index(drop=True), use_container_width=True)

    if display_table.empty:
        st.write(f"No significant characteristic words found for category **{category}**.")
        return

    positive_subset = display_table[display_table['Test Value'] > 0].sort_values('Test Value', ascending=False).head(top_n)
    negative_subset = display_table[display_table['Test Value'] < 0].sort_values('Test Value').head(top_n)
    combined_subset = pd.concat([positive_subset, negative_subset])

    if combined_subset.empty:
        st.write(f"No characteristic words with both positive and negative test values for category **{category}**.")
        return

    combined_subset['Type'] = combined_subset['Test Value'].apply(lambda x: 'Overrepresented' if x > 0 else 'Underrepresented')

    fig = px.bar(
        combined_subset,
        x="Test Value",
        y="Term",
        color="Type",
        orientation='h',
        title=f"Top Characteristic Words for Category: {category}",
        labels={"Test Value": "Test Value", "Term": "Word"},
        height=600,
        color_discrete_map={"Overrepresented": "blue", "Underrepresented": "red"}
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder='total ascending',
            showgrid=False
        ),
        legend_title_text='Word Representation',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
    )

    fig.add_shape(
        dict(
            type='line',
            x0=0,
            y0=-0.5,
            x1=0,
            y1=len(combined_subset) - 0.5,
            line=dict(color='black', dash='dash')
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def display_results(characteristic_words_df, frequency_inventory, categories, num_categories, total_tokens, total_types, morphological_complexity, num_hapax, alpha, category_stats):
    """
    Displays the summary statistics and the results tables along with corresponding bar charts.
    """
    st.write("### 📈 Summary Statistics")
    st.markdown(f"""
    - **Number of Categories Analyzed:** {num_categories}
    - **Total Number of Terms (Tokens):** {total_tokens}
    - **Total Number of Unique Words (Types):** {total_types}
    - **Morphological Complexity (Types/Token Ratio):** {morphological_complexity}
    - **Number of Hapax (Words that appear only once):** {num_hapax}
    - **Significance Level (alpha):** {alpha}
    """)

    st.write("### 📄 Characteristic Words Tables and Visualizations")
    st.markdown("""
    **Table Explanation:**
    - **Category:** The specific group or category within your dataset.
    - **Term:** The characteristic word or n-gram.
    - **Internal Frequency:** Number of times the term appears within the category.
    - **Global Frequency:** Number of times the term appears across the entire corpus.
    - **Test Value:** 
      $$\\log_2\\left(\\frac{x/n}{K/M}\\right)$$
      where:
        - *x* = Internal Frequency
        - *n* = Total terms in the category
        - *K* = Global Frequency
        - *M* = Total terms in the corpus
      This value indicates how much more (or less) the term is associated with the category compared to the corpus.
    - **P-Value:** The probability of observing the term's frequency in the category by chance. Displayed with three decimal points (e.g., .035) or as "<.001" if p < .001.
    """)

    if characteristic_words_df.empty:
        st.warning("No significant characteristic words found based on the provided criteria.")
    else:
        for cat in sorted(categories):
            cat_df = characteristic_words_df[characteristic_words_df['Category'] == cat]
            visualize_and_display(cat, cat_df, category_stats, alpha, top_n=10)

    st.write("### 📝 Frequency Inventory of the Entire Corpus")
    st.dataframe(frequency_inventory.reset_index(drop=True), use_container_width=True)

def download_results(frequency_inventory, characteristic_words_df):
    """
    Provides download option for Excel format with two sheets:
    1. Frequency Inventory
    2. Characteristic Words
    """
    st.write("### ⬇️ Download Results")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        frequency_inventory.to_excel(writer, index=False, sheet_name='Frequency Inventory')
        characteristic_words_df.to_excel(writer, index=False, sheet_name='Characteristic Words')
    st.download_button(
        label="📥 Download Results as Excel",
        data=excel_buffer.getvalue(),
        file_name="characteristic_words_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("📊 Characteristic Words Detection in Corpus Linguistics")

    st.markdown("""
    **Introduction**
    
    Corpus linguistics involves analyzing large text collections to understand language use. 
    Characteristic words appear unusually often or rarely in specific text subsets compared to the whole corpus, helping uncover themes and distinctions.
    
    *Lebart, L., Salem, A., & Berry, L. (1997). _Exploring textual data_. Springer.*
    """)

    st.markdown("""
    **Overview**
    
    This app identifies characteristic words within selected categories of your dataset. By analyzing the frequency of words (or n-grams) in specific categories compared to the entire corpus, it uncovers terms uniquely associated with each category.
    Statistical significance testing ensures identified words are not occurring by chance.
    """)

    st.sidebar.header("🔧 Configuration")
    uploaded_file = st.sidebar.file_uploader("📂 Upload Your Data", type=["csv", "xlsx", "tsv", "txt"])

    # Initialize session_state variables if not already present
    for var, default in [
        ('result_df', pd.DataFrame()),
        ('frequency_inventory', pd.DataFrame()),
        ('categories', []),
        ('num_categories', 0),
        ('total_tokens', 0),
        ('total_types', 0),
        ('morphological_complexity', 0.00),
        ('num_hapax', 0),
        ('category_stats', {}),
        ('replacements', []),
        ('alpha', 0.05)
    ]:
        if var not in st.session_state:
            st.session_state[var] = default

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
            
            st.sidebar.write("### Available Columns:")
            st.sidebar.write(df.columns.tolist())

            st.sidebar.write("### Data Preview:")
            st.sidebar.dataframe(df.head())

            # Verify if the DataFrame has any columns
            if df.empty:
                st.sidebar.error("⚠️ The uploaded file is empty.")
            else:
                # User selects the text and category columns
                text_col = st.sidebar.selectbox("Select the text column", options=df.columns)
                category_col = st.sidebar.selectbox("Select the category column", options=df.columns)

                # Check if the selected columns are valid
                if text_col not in df.columns:
                    st.sidebar.error(f"⚠️ The text column '{text_col}' does not exist in the data.")
                if category_col not in df.columns:
                    st.sidebar.error(f"⚠️ The category column '{category_col}' does not exist in the data.")

                st.sidebar.write("### Word Replacements")
                if st.sidebar.button("➕ Add Replacement"):
                    st.session_state['replacements'].append({'old': '', 'new': ''})

                # Display existing replacements
                for idx, repl in enumerate(st.session_state.get('replacements', [])):
                    with st.sidebar.expander(f"Replacement {idx+1}", expanded=True):
                        repl['old'] = st.text_input(f"Original Phrase {idx+1}", value=repl['old'], key=f"repl_old_{idx}")
                        repl['new'] = st.text_input(f"Replacement {idx+1}", value=repl['new'], key=f"repl_new_{idx}")
                        remove_repl = st.checkbox(f"🗑️ Remove Replacement {idx+1}", key=f"remove_repl_{idx}")
                        if remove_repl:
                            st.session_state['replacements'].pop(idx)
                            st.experimental_rerun()

                # Convert replacements list to dictionary
                replacements = {repl['old']: repl['new'] for repl in st.session_state.get('replacements', []) if repl['old'] and repl['new']}

                st.sidebar.write("### Stopword Removal")
                remove_sw = st.sidebar.checkbox("🗑️ Remove stopwords?", value=False)
                if remove_sw:
                    lang_choice = st.sidebar.selectbox("🌐 Select language for stopwords", list(LANGUAGES.keys()))
                else:
                    lang_choice = "English"

                st.sidebar.write("### Custom Stopwords")
                custom_stopword_option = st.sidebar.radio(
                    "Choose how to provide custom stopwords:",
                    options=["None", "Type custom stopwords"],
                    index=0
                )
                
                custom_stopwords = []
                if custom_stopword_option == "Type custom stopwords":
                    custom_stopword_input = st.sidebar.text_input("✍️ Enter custom stopwords separated by commas:")
                    if custom_stopword_input:
                        custom_stopwords = [word.strip().lower() for word in custom_stopword_input.split(',') if word.strip()]

                st.sidebar.write("### Lemmatization")
                lemmatize = st.sidebar.checkbox("📝 Apply lemmatization?", value=True)

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

                st.sidebar.write("### Minimum Frequency")
                min_freq = st.sidebar.number_input(
                    "🔢 Minimum frequency:",
                    min_value=1,
                    max_value=1000,
                    value=1,
                    step=1
                )

                st.sidebar.write("### Significance Level")
                alpha = st.sidebar.number_input("📉 Significance level (alpha)", min_value=0.0001, max_value=0.5, value=0.05, step=0.01)

                st.sidebar.write("### Categories to Analyze")
                all_categories = df[category_col].dropna().unique().tolist()
                selected_categories = st.sidebar.multiselect(
                    "Select categories to analyze",
                    options=all_categories,
                    default=[]
                )

                if st.sidebar.button("🚀 Run Analysis"):
                    # Validate selections
                    if not ngram_ranges:
                        st.error("⚠️ Please select at least one N-gram option to proceed.")
                    elif not text_col or not category_col:
                        st.error("⚠️ Please select both text and category columns.")
                    elif not selected_categories:
                        st.error("⚠️ Please select at least one category to analyze.")
                    else:
                        st.header("🔍 Analysis Results")
                        st.write("### Processing...")
                        progress = st.progress(0)

                        with st.spinner("🕒 Analyzing the corpus..."):
                            try:
                                characteristic_words_df, frequency_inventory = perform_analysis(
                                    df,
                                    text_col,
                                    category_col,
                                    selected_categories,
                                    remove_sw,
                                    lemmatize,
                                    lemmatizer,
                                    ngram_ranges,
                                    min_freq,
                                    alpha,
                                    custom_stopwords,
                                    replacements,
                                    progress,
                                    lang_choice
                                )
                                # Store results in session_state
                                st.session_state['result_df'] = characteristic_words_df
                                st.session_state['frequency_inventory'] = frequency_inventory
                                st.session_state['categories'] = selected_categories
                                st.session_state['num_categories'] = len(selected_categories)
                                st.session_state['total_tokens'] = frequency_inventory['Frequency'].sum()
                                st.session_state['total_types'] = frequency_inventory['Word'].nunique()
                                st.session_state['morphological_complexity'] = round(st.session_state['total_types'] / st.session_state['total_tokens'], 2) if st.session_state['total_tokens'] > 0 else 0.00
                                st.session_state['num_hapax'] = (frequency_inventory['Frequency'] == 1).sum()
                                # Compute category_stats
                                category_stats = {}
                                for cat in selected_categories:
                                    cat_df = df[df[category_col] == cat]
                                    num_instances = len(cat_df)
                                    cat_tokens = 0
                                    cat_types_set = set()
                                    for text in cat_df[text_col].dropna().astype(str):
                                        tokens = re.findall(r'\b\w+\b', text.lower())
                                        terms, tokens_processed = preprocess_text(
                                            ' '.join(tokens),
                                            lang=lang_choice,
                                            remove_stopwords=remove_sw,
                                            lemmatize=lemmatize,
                                            lemmatizer_obj=lemmatizer_obj,
                                            ngram_ranges=[1],
                                            stop_words=stop_words
                                        )
                                        cat_tokens += len(tokens_processed)
                                        cat_types_set.update(tokens_processed)
                                    cat_types = len(cat_types_set)
                                    cat_morph_complexity = round(cat_types / cat_tokens, 2) if cat_tokens > 0 else 0.00
                                    # Handle cases where the term might not be found in frequency_inventory
                                    cat_num_hapax = 0
                                    for token in cat_types_set:
                                        freq_row = frequency_inventory.loc[frequency_inventory['Word'] == token, 'Frequency']
                                        if not freq_row.empty and freq_row.values[0] == 1:
                                            cat_num_hapax += 1
                                    category_stats[cat] = {
                                        "Number of Instances": num_instances,
                                        "Number of Tokens": cat_tokens,
                                        "Number of Types": cat_types,
                                        "Morphological Complexity": cat_morph_complexity,
                                        "Number of Hapax": cat_num_hapax
                                    }
                                st.session_state['category_stats'] = category_stats
                                st.session_state['alpha'] = alpha
                                # Display results
                                display_results(
                                    characteristic_words_df,
                                    frequency_inventory,
                                    selected_categories,
                                    len(selected_categories),
                                    st.session_state['total_tokens'],
                                    st.session_state['total_types'],
                                    st.session_state['morphological_complexity'],
                                    st.session_state['num_hapax'],
                                    alpha,
                                    category_stats
                                )
                            except KeyError as ke:
                                st.error(f"⚠️ Missing expected column: {ke}")
                            except Exception as e:
                                st.error(f"⚠️ An error occurred during analysis: {e}")

    # 3. Run the main function and add the footer
    if __name__ == "__main__":
        main()
        add_footer()
