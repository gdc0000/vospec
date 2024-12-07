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
import zipfile

# Define supported languages and their corresponding stopword lists and stemmers
LANGUAGES = {
    "English": {"stopwords": get_stop_words("english"), "stemmer": stemmer("english")},
    "French": {"stopwords": get_stop_words("french"), "stemmer": stemmer("french")},
    "Italian": {"stopwords": get_stop_words("italian"), "stemmer": stemmer("italian")},
    "Spanish": {"stopwords": get_stop_words("spanish"), "stemmer": stemmer("spanish")}
}

def preprocess_text(text, lang, remove_stopwords, stem_words, stemmer_obj, ngram_ranges, stop_words, group_names=None):
    """
    Tokenizes, removes stopwords (if requested), stems (if enabled and token not in group_names),
    and generates n-grams from the input text.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())

    # Remove stopwords if requested
    if remove_stopwords and stop_words:
        tokens = [token for token in tokens if token not in stop_words]

    # Stemming if requested
    if stem_words and stemmer_obj is not None:
        tokens = [stemmer_obj.stemWord(token) if not group_names or token not in group_names else token for token in tokens]

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

def find_most_frequent_original_forms(stem2original):
    """
    For each stem, find the most frequent original word form.
    """
    stem2repr = {}
    for stem, counts in stem2original.items():
        repr_word = max(counts.items(), key=lambda x: x[1])[0]
        stem2repr[stem] = repr_word
    return stem2repr

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

def apply_word_grouping(text, word_group_mapping):
    """
    Replace grouped words with their group name.
    Grouped words are not stemmed and are treated as single tokens.
    """
    if not word_group_mapping:
        return text

    all_grouped_words = [word for words in word_group_mapping.values() for word in words]
    all_grouped_words_sorted = sorted(all_grouped_words, key=lambda x: len(x), reverse=True)
    all_grouped_words_escaped = [re.escape(word) for word in all_grouped_words_sorted]

    pattern = re.compile(r'\b(?:' + '|'.join(all_grouped_words_escaped) + r')\b', re.IGNORECASE)
    word_to_group = {}
    for group, words in word_group_mapping.items():
        for word in words:
            word_to_group[word.lower()] = group.lower()

    def replace_match(match):
        matched_word = match.group(0).lower()
        return word_to_group.get(matched_word, matched_word)

    return pattern.sub(replace_match, text)

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

def perform_analysis(df, text_col, category_col, remove_sw, stem_words, chosen_lang, ngram_ranges, min_freq, alpha, custom_stopwords, word_group_mapping, progress):
    """
    Performs the characteristic words analysis and returns the result DataFrame.
    Updates the progress bar to indicate analysis steps.
    """
    # Apply Word Grouping
    progress.progress(5)
    df[text_col] = df[text_col].astype(str).apply(lambda x: apply_word_grouping(x, word_group_mapping))
    
    # Retrieve stopwords and stemmer based on selected language
    progress.progress(10)
    stop_words = LANGUAGES[chosen_lang]["stopwords"] if remove_sw else []
    stemmer_obj = LANGUAGES[chosen_lang]["stemmer"] if stem_words else None

    if custom_stopwords:
        stop_words.extend(custom_stopwords)

    overall_freq = {}
    category_freq = {}
    category_counts = {}
    total_terms = 0
    stem2original = {}
    categories = df[category_col].dropna().unique()

    for cat in categories:
        category_freq[cat] = {}
        category_counts[cat] = 0

    word_freq = {}
    group_names = set(word_group_mapping.keys())

    progress.progress(20)
    for _, row in df.iterrows():
        cat = row[category_col]
        text = str(row[text_col])
        terms, tokens = preprocess_text(
            text,
            lang=chosen_lang,
            remove_stopwords=remove_sw,
            stem_words=stem_words,
            stemmer_obj=stemmer_obj,
            ngram_ranges=ngram_ranges,
            stop_words=stop_words,
            group_names=group_names
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
                if n == 1:
                    token = tokens[i]
                    if token in group_names:
                        ngram = token
                    else:
                        ngram = stemmer_obj.stemWord(token) if (stem_words and stemmer_obj) else token
                else:
                    if stem_words and stemmer_obj:
                        ngram = "_".join([stemmer_obj.stemWord(tok) if tok not in group_names else tok for tok in tokens[i:i+n]])
                    else:
                        ngram = "_".join(tokens[i:i+n])
                selected_terms.append(ngram)

        if 1 in ngram_ranges and stem_words and stemmer_obj:
            for stemmed_token, orig in zip([term.split("_")[0] for term in selected_terms if '_' not in term], tokens):
                if stemmed_token not in stem2original:
                    stem2original[stemmed_token] = {}
                stem2original[stemmed_token][orig] = stem2original[stemmed_token].get(orig, 0) + 1

        for t in selected_terms:
            overall_freq[t] = overall_freq.get(t, 0) + 1
            category_freq[cat][t] = category_freq[cat].get(t, 0) + 1
        category_counts[cat] += len(selected_terms)
        total_terms += len(selected_terms)

    progress.progress(35)
    overall_freq_filtered = {k: v for k, v in overall_freq.items() if v > 1 and v >= min_freq}
    for cat in categories:
        category_freq[cat] = {k: v for k, v in category_freq[cat].items() if k in overall_freq_filtered}

    if 1 in ngram_ranges and stem_words and stem2original:
        stem2repr = find_most_frequent_original_forms(stem2original)
    else:
        stem2repr = {}

    progress.progress(50)
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
            pval_over = hypergeom.sf(x-1, total_terms, K, n)
            pval_under = hypergeom.cdf(x, total_terms, K, n)
            pval = min(pval_over, pval_under)

            all_pvals.append(pval)
            all_terms.append(t)
            all_cats.append(cat)
            all_x.append(x)
            all_K.append(K)
            all_n.append(n)

    progress.progress(65)
    rejected = benjamini_hochberg_correction(all_pvals, alpha=alpha)

    progress.progress(80)
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
        global_ratio = K / (total_terms + epsilon)
        test_val = math.log2((term_ratio + epsilon) / (global_ratio + epsilon))

        if 1 in ngram_ranges and stem_words and stem2repr:
            stem = t.split("_")[0]
            term_repr = stem2repr.get(stem, t)
        else:
            term_repr = t

        term_repr_display = term_repr.replace("_", " ")

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

    result_df = pd.DataFrame(final_data).sort_values(by=["Category", "P-Value"])

    num_categories = len(categories)
    total_tokens = sum(word_freq.values())
    total_types = len(word_freq)
    morphological_complexity = round(total_types / total_tokens, 2) if total_tokens > 0 else 0.00
    num_hapax = sum(1 for freq in word_freq.values() if freq == 1)

    category_stats = {}
    for cat in categories:
        cat_df = df[df[category_col] == cat]
        num_instances = len(cat_df)
        cat_tokens = 0
        cat_types_set = set()
        for text in cat_df[text_col].dropna().astype(str):
            tokens = re.findall(r'\b\w+\b', text.lower())
            tokens = apply_word_grouping(' '.join(tokens), word_group_mapping).split()
            if remove_sw and stop_words:
                tokens = [token for token in tokens if token not in stop_words]
            if stem_words and stemmer_obj is not None:
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

    # Complete the progress bar
    progress.progress(100)
    return result_df, categories, num_categories, total_tokens, total_types, morphological_complexity, num_hapax, category_stats

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

def display_results(result_df, categories, num_categories, total_tokens, total_types, morphological_complexity, num_hapax, alpha, category_stats):
    """
    Displays the summary statistics and the results table along with corresponding bar charts.
    """
    st.write("### 📈 Summary Statistics")
    st.markdown(f"""
    - **Number of Categories:** {num_categories}
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

    if result_df.empty:
        st.warning("No significant characteristic words found based on the provided criteria.")
    else:
        for cat in sorted(categories):
            cat_df = result_df[result_df['Category'] == cat]
            visualize_and_display(cat, cat_df, category_stats, alpha, top_n=10)

def download_results():
    """
    Provides download options for CSV, Excel, or both formats.
    Retrieves the result_df from session_state.
    """
    st.write("### ⬇️ Download Results")
    if 'result_df' not in st.session_state or st.session_state['result_df'].empty:
        st.info("No results to download.")
        return
    result_df = st.session_state['result_df']
    download_options = st.multiselect(
        "Select download format(s):",
        options=["CSV", "Excel"],
        default=["CSV"]
    )
    if download_options:
        download_files = {}
        if "CSV" in download_options:
            csv_buffer = StringIO()
            result_df.to_csv(csv_buffer, index=False)
            download_files["characteristic_words.csv"] = csv_buffer.getvalue().encode('utf-8')
        if "Excel" in download_options:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                result_df.to_excel(writer, index=False, sheet_name='Characteristic Words')
            download_files["characteristic_words.xlsx"] = excel_buffer.getvalue()
        
        if len(download_files) > 1:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zipf:
                for filename, data in download_files.items():
                    zipf.writestr(filename, data)
            st.download_button(
                label="📥 Download Selected Formats as ZIP",
                data=zip_buffer.getvalue(),
                file_name="characteristic_words.zip",
                mime="application/zip"
            )
        else:
            for filename, data in download_files.items():
                mime_type = "text/csv" if filename.endswith('.csv') else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                st.download_button(
                    label=f"📥 Download Results as {filename.split('.')[-1].upper()}",
                    data=data,
                    file_name=filename,
                    mime=mime_type
                )
    else:
        st.info("Select at least one format to download the results.")

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
    
    This app identifies characteristic words within categories of your dataset. By analyzing the frequency of words (or n-grams) in specific categories compared to the entire corpus, it uncovers terms uniquely associated with each category.
    Statistical significance testing ensures identified words are not occurring by chance.
    """)

    st.sidebar.header("🔧 Configuration")
    uploaded_file = st.sidebar.file_uploader("📂 Upload Your Data", type=["csv", "xlsx", "tsv", "txt"])

    # Initialize session_state variables if not already present
    for var, default in [
        ('result_df', pd.DataFrame()),
        ('categories', []),
        ('num_categories', 0),
        ('total_tokens', 0),
        ('total_types', 0),
        ('morphological_complexity', 0.00),
        ('num_hapax', 0),
        ('category_stats', {}),
        ('word_groups', [])
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

            text_col = st.sidebar.selectbox("Select the text column", options=df.columns)
            category_col = st.sidebar.selectbox("Select the category column", options=df.columns)

            st.sidebar.write("### Word Grouping")
            if st.sidebar.button("➕ Add Word Group"):
                st.session_state['word_groups'].append({'name': '', 'words': []})

            for idx, group in enumerate(st.session_state['word_groups']):
                with st.sidebar.expander(f"Group {idx+1}", expanded=True):
                    group['name'] = st.text_input(f"Group {idx+1} Name", value=group['name'], key=f"group_name_{idx}")
                    group_words_input = st.text_input(f"Group {idx+1} Words (separated by commas)", value=", ".join(group['words']), key=f"group_words_{idx}")
                    if group_words_input:
                        group['words'] = [word.strip().lower() for word in group_words_input.split(',') if word.strip()]
                    else:
                        group['words'] = []
                    remove_group = st.checkbox(f"🗑️ Remove Group {idx+1}", key=f"remove_group_{idx}")
                    if remove_group:
                        st.session_state['word_groups'].pop(idx)
                        st.experimental_set_query_params(refresh='true')
                        st.stop()

            word_group_mapping = {}
            for group in st.session_state['word_groups']:
                group_name = group['name'].strip().lower()
                if group_name and '_' not in group_name:
                    word_group_mapping[group_name] = group['words']
                elif group_name and '_' in group_name:
                    st.sidebar.error(f"Group name '{group['name']}' should not contain underscores '_'. Please rename it.")

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

            st.sidebar.write("### Stemming")
            stem_words = st.sidebar.checkbox("🪓 Apply stemming?", value=True)

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

            if st.sidebar.button("🚀 Run Analysis"):
                if not ngram_ranges:
                    st.error("⚠️ Please select at least one N-gram option to proceed.")
                elif not text_col or not category_col:
                    st.error("⚠️ Please select both text and category columns.")
                else:
                    if category_col not in df.columns:
                        st.error(f"⚠️ The selected category column '{category_col}' does not exist.")
                        return
                    st.header("🔍 Analysis Results")
                    st.write("### Processing...")
                    progress = st.progress(0)

                    with st.spinner("🕒 Analyzing the corpus..."):
                        result = perform_analysis(
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
                            word_group_mapping,
                            progress
                        )
                        if len(result) == 8:
                            (
                                result_df,
                                categories,
                                num_categories,
                                total_tokens,
                                total_types,
                                morphological_complexity,
                                num_hapax,
                                category_stats
                            ) = result
                            # Store results in session_state
                            st.session_state['result_df'] = result_df
                            st.session_state['categories'] = categories
                            st.session_state['num_categories'] = num_categories
                            st.session_state['total_tokens'] = total_tokens
                            st.session_state['total_types'] = total_types
                            st.session_state['morphological_complexity'] = morphological_complexity
                            st.session_state['num_hapax'] = num_hapax
                            st.session_state['category_stats'] = category_stats
                            # Display results
                            display_results(
                                result_df,
                                categories,
                                num_categories,
                                total_tokens,
                                total_types,
                                morphological_complexity,
                                num_hapax,
                                alpha,
                                category_stats
                            )
                        else:
                            st.error("Error during analysis.")

        except ValueError as ve:
            st.sidebar.error(f"⚠️ {ve}")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error processing the uploaded file: {e}")
    else:
        st.sidebar.info("📥 Awaiting file upload.")

    if 'result_df' in st.session_state and not st.session_state['result_df'].empty:
        download_results()

if __name__ == "__main__":
    main()
    add_footer()
