import streamlit as st
import pandas as pd
import numpy as np
import re
from stop_words import get_stop_words
from snowballstemmer import stemmer # Corrected: this is the module
from scipy.stats import hypergeom
from io import StringIO, BytesIO
import math
import plotly.express as px
import zipfile
import collections # Added for Counter

# Initialize stemmers and stopwords once
_SNOWBALL_STEMMERS_INSTANCES = {
    lang: stemmer(lang)() for lang in ["english", "french", "italian", "spanish"]
}

LANGUAGES = {
    "English": {"stopwords_set": set(get_stop_words("english")), "stemmer_obj": _SNOWBALL_STEMMERS_INSTANCES["english"]},
    "French": {"stopwords_set": set(get_stop_words("french")), "stemmer_obj": _SNOWBALL_STEMMERS_INSTANCES["french"]},
    "Italian": {"stopwords_set": set(get_stop_words("italian")), "stemmer_obj": _SNOWBALL_STEMMERS_INSTANCES["italian"]},
    "Spanish": {"stopwords_set": set(get_stop_words("spanish")), "stemmer_obj": _SNOWBALL_STEMMERS_INSTANCES["spanish"]}
}

def _compile_group_pattern_and_map(word_group_mapping_lower):
    """Pre-compiles regex pattern and creates word-to-group map."""
    if not word_group_mapping_lower:
        return None, {}

    all_grouped_words_flat = [
        word for words_list in word_group_mapping_lower.values() for word in words_list
    ]
    if not all_grouped_words_flat:
        return None, {}

    # Sort by length, longest first, to handle overlapping matches correctly
    all_grouped_words_flat_sorted = sorted(list(set(all_grouped_words_flat)), key=len, reverse=True)
    all_grouped_words_escaped = [re.escape(word) for word in all_grouped_words_flat_sorted]

    pattern = re.compile(r'\b(?:' + '|'.join(all_grouped_words_escaped) + r')\b', re.IGNORECASE)
    
    word_to_group = {}
    for group_name, words_in_group in word_group_mapping_lower.items():
        for word_val in words_in_group:
            word_to_group[word_val.lower()] = group_name # group_name is already lower
    return pattern, word_to_group

def _apply_grouping_with_compiled_pattern(text, compiled_pattern, word_to_group_map):
    """Applies word grouping using a pre-compiled pattern and map."""
    if not compiled_pattern:
        return text
    
    def replace_func(match):
        matched_word_lower = match.group(0).lower()
        return word_to_group_map.get(matched_word_lower, matched_word_lower) # Fallback to original if somehow not in map
    
    return compiled_pattern.sub(replace_func, text)


def preprocess_text_optimized(text_after_grouping, lang_stemmer_obj, stem_words, ngram_ranges, current_stopwords_set, group_names_set):
    """
    Optimized text preprocessing: tokenizes, removes stopwords, stems (respecting group names),
    and generates n-grams.
    Returns:
        analysis_ngrams: List of n-grams for frequency analysis (stemmed if configured).
        unstemmed_unigrams_for_mapping: List of unstemmed unigrams (post-stopword, post-grouping) for stem2original.
        base_unigrams_for_corpus_stats: List of unigrams (stemmed if configured, like analysis_ngrams) for corpus/category stats.
    """
    # Text is already grouped and lowercased before this function if it comes from apply_word_grouping
    # However, apply_word_grouping might not lowercase everything if a group name itself has capitals.
    # For safety, ensure lowercasing here or ensure input `text_after_grouping` is consistently lowercase.
    # Assuming `text_after_grouping` is from `df[text_col].astype(str).apply(...)` where `...` handles lowercasing of groups.
    raw_tokens = re.findall(r'\b\w+\b', text_after_grouping.lower())

    # Remove stopwords
    if current_stopwords_set:
        unstemmed_unigrams_for_mapping = [token for token in raw_tokens if token not in current_stopwords_set]
    else:
        unstemmed_unigrams_for_mapping = list(raw_tokens)

    # Stemming (conditionally, respecting group_names)
    # These tokens will form the basis of n-grams for analysis and unigram stats.
    if stem_words and lang_stemmer_obj:
        base_unigrams_for_corpus_stats = [
            lang_stemmer_obj.stemWord(token) if token not in group_names_set else token
            for token in unstemmed_unigrams_for_mapping
        ]
    else:
        base_unigrams_for_corpus_stats = list(unstemmed_unigrams_for_mapping)

    # Generate n-grams for analysis from base_unigrams_for_corpus_stats
    analysis_ngrams = []
    for n in ngram_ranges:
        if n == 1: # Unigrams are simply the base_unigrams_for_corpus_stats
            analysis_ngrams.extend(base_unigrams_for_corpus_stats)
        elif n > len(base_unigrams_for_corpus_stats):
            continue
        else: # For n > 1
            for i in range(len(base_unigrams_for_corpus_stats) - n + 1):
                ngram = "_".join(base_unigrams_for_corpus_stats[i:i+n])
                analysis_ngrams.append(ngram)
    
    # Ensure uniqueness if multiple ngram ranges include unigrams (e.g. [1,2] means unigrams are added once from n=1)
    if 1 in ngram_ranges and len(ngram_ranges) > 1:
         # Rebuild analysis_ngrams to avoid duplicate unigrams if n=1 was specified along with others
        temp_analysis_ngrams = []
        if 1 in ngram_ranges:
            temp_analysis_ngrams.extend(base_unigrams_for_corpus_stats)

        for n_val in ngram_ranges:
            if n_val == 1:
                continue # Unigrams already added
            if n_val > len(base_unigrams_for_corpus_stats):
                continue
            for i in range(len(base_unigrams_for_corpus_stats) - n_val + 1):
                ngram = "_".join(base_unigrams_for_corpus_stats[i:i+n_val])
                temp_analysis_ngrams.append(ngram)
        analysis_ngrams = temp_analysis_ngrams
    elif not ngram_ranges: # Should not happen due to UI validation
        analysis_ngrams = []


    return analysis_ngrams, unstemmed_unigrams_for_mapping, base_unigrams_for_corpus_stats


def find_most_frequent_original_forms(stem2original_counters):
    """For each stem, find the most frequent original word form using collections.Counter."""
    stem2repr = {}
    for stem, counts_counter in stem2original_counters.items():
        if counts_counter: # Ensure counter is not empty
            repr_word, _ = counts_counter.most_common(1)[0]
            stem2repr[stem] = repr_word
        else:
            stem2repr[stem] = stem # Fallback to stem itself if no original forms recorded (should not happen)
    return stem2repr

def benjamini_hochberg_correction(pvals, alpha=0.05):
    """
    Performs the Benjamini-Hochberg FDR correction on a list of p-values.
    Returns a boolean array indicating which hypotheses are rejected.
    """
    pvals_array = np.array(pvals)
    n = len(pvals_array)
    if n == 0:
        return np.array([], dtype=bool)
        
    sorted_indices = np.argsort(pvals_array)
    sorted_pvals = pvals_array[sorted_indices]
    
    thresholds = (np.arange(1, n + 1) / n) * alpha
    
    below_threshold = sorted_pvals <= thresholds
    
    if not np.any(below_threshold):
        return np.zeros(n, dtype=bool)
        
    max_idx_below = np.max(np.where(below_threshold)[0]) # Get the index within the sorted array
    
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_indices[:max_idx_below + 1]] = True
    return rejected

# apply_word_grouping is now a thin wrapper or integrated
# The main logic is in _compile_group_pattern_and_map and _apply_grouping_with_compiled_pattern

def add_footer():
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown("""
    [GitHub](https://github.com/gdc0000) | 
    [ORCID](https://orcid.org/0000-0002-1439-5790) | 
    [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """)

@st.cache_data
def load_data(uploaded_file):
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

def perform_analysis(df, text_col, category_col, remove_sw, stem_words, chosen_lang, ngram_ranges, min_freq, alpha, custom_stopwords_list, word_group_mapping_lower, progress):
    """
    Performs the characteristic words analysis.
    `word_group_mapping_lower` should have lowercase keys and lists of lowercase words.
    `custom_stopwords_list` should be a list of lowercase words.
    """
    progress.progress(5)
    
    # Compile pattern for word grouping (once)
    group_pattern, word_to_group_map = _compile_group_pattern_and_map(word_group_mapping_lower)
    
    # Apply Word Grouping to the text column
    df[text_col] = df[text_col].astype(str).apply(
        _apply_grouping_with_compiled_pattern,
        args=(group_pattern, word_to_group_map)
    )
    group_names_set = set(word_group_mapping_lower.keys())

    progress.progress(10)
    lang_details = LANGUAGES[chosen_lang]
    lang_stemmer_obj = lang_details["stemmer_obj"]
    
    current_stopwords_set = set()
    if remove_sw:
        current_stopwords_set.update(lang_details["stopwords_set"])
    if custom_stopwords_list: # Already lowercase from main
        current_stopwords_set.update(custom_stopwords_list)

    overall_freq = collections.Counter()
    categories = df[category_col].dropna().unique()
    category_freq = {cat: collections.Counter() for cat in categories}
    category_term_counts = {cat: 0 for cat in categories} # N_c: total terms in category c
    
    corpus_word_freq_for_stats = collections.Counter() # For corpus-level type/token, hapax based on unigrams
    stem2original = {} # Maps stem to Counter of original forms

    total_terms_corpus = 0 # M: total terms in corpus

    progress.progress(20)
    for _, row in df.iterrows(): # Consider df.itertuples() for minor speedup if needed
        cat = row[category_col]
        text = str(row[text_col]) # Text is already grouped

        analysis_ngrams, unstemmed_unigrams, base_unigrams_stats = preprocess_text_optimized(
            text, lang_stemmer_obj, stem_words, ngram_ranges, current_stopwords_set, group_names_set
        )

        overall_freq.update(analysis_ngrams)
        if cat in category_freq: # Handle potential NaN categories if not dropped earlier
            category_freq[cat].update(analysis_ngrams)
            category_term_counts[cat] += len(analysis_ngrams)
        
        total_terms_corpus += len(analysis_ngrams)
        corpus_word_freq_for_stats.update(base_unigrams_stats) # For corpus stats

        if stem_words and lang_stemmer_obj:
            for unstemmed_u, stemmed_u in zip(unstemmed_unigrams, base_unigrams_stats):
                if unstemmed_u not in group_names_set: # Only map if it's not a group name (which isn't stemmed)
                    if stemmed_u not in stem2original:
                        stem2original[stemmed_u] = collections.Counter()
                    stem2original[stemmed_u][unstemmed_u] += 1
    
    progress.progress(35)
    # Filter terms by minimum frequency.
    # Original: v > 1 and v >= min_freq. Simplified to v >= max(2, min_freq)
    # Here, using just min_freq as the threshold. Adjust if v>1 is a hard rule.
    effective_min_freq = max(2, min_freq) if min_freq == 1 else min_freq # Example: if min_freq=1, still filter hapax.
    # Or, simply: effective_min_freq = min_freq (if min_freq is the absolute floor)
    # Let's use: v >= min_freq. If you want to exclude hapax always, use max(min_freq, 2)
    overall_freq_filtered = {
        term: count for term, count in overall_freq.items() if count >= min_freq
    }
    
    for cat in categories:
        category_freq[cat] = {
            term: count for term, count in category_freq[cat].items() if term in overall_freq_filtered
        }

    stem2repr = {}
    if stem_words and stem2original: # Check if stem_words was true for stem2original to be populated
        stem2repr = find_most_frequent_original_forms(stem2original)

    progress.progress(50)
    all_pvals, all_terms, all_cats, all_x, all_K, all_n_cat_terms = [], [], [], [], [], []

    for cat in categories:
        n_c = category_term_counts[cat] # Total terms in this category (n in hypergeom)
        cat_vocab_freqs = category_freq[cat] # Frequencies of terms in this category

        for term, x in cat_vocab_freqs.items(): # x: term count in category (k in hypergeom)
            if term not in overall_freq_filtered: # Should not happen if filtered above
                continue
            
            K_corpus = overall_freq_filtered[term] # Total count of term in corpus (K in hypergeom)
            
            # M: total_terms_corpus (N in hypergeom)
            # n: n_c (n in hypergeom)
            # K: K_corpus (K in hypergeom)
            # x: x (k in hypergeom)
            # sf(k-1, N, K, n) = P(X >= k)
            pval_over = hypergeom.sf(x - 1, total_terms_corpus, K_corpus, n_c)
            # cdf(k, N, K, n) = P(X <= k)
            pval_under = hypergeom.cdf(x, total_terms_corpus, K_corpus, n_c)
            pval = min(pval_over, pval_under)

            all_pvals.append(pval)
            all_terms.append(term)
            all_cats.append(cat)
            all_x.append(x)
            all_K.append(K_corpus)
            all_n_cat_terms.append(n_c)

    progress.progress(65)
    rejected = benjamini_hochberg_correction(all_pvals, alpha=alpha)

    progress.progress(80)
    final_data = []
    epsilon = 1e-9 # For numerical stability in ratios

    for i in range(len(all_terms)):
        if rejected[i]: # Only include statistically significant terms
            term_raw = all_terms[i] # This is the (potentially stemmed) term/ngram
            cat = all_cats[i]
            x = all_x[i]
            K_corpus = all_K[i]
            n_c = all_n_cat_terms[i]
            pval = all_pvals[i]

            term_ratio_cat = x / (n_c + epsilon)
            term_ratio_corpus = K_corpus / (total_terms_corpus + epsilon)
            test_val = math.log2((term_ratio_cat + epsilon) / (term_ratio_corpus + epsilon))

            # Represent term (de-stem first part if n-gram and stemming was on)
            term_display = term_raw
            if stem_words and stem2repr:
                parts = term_raw.split("_")
                first_part_stemmed = parts[0]
                first_part_original = stem2repr.get(first_part_stemmed, first_part_stemmed)
                parts[0] = first_part_original
                term_display = " ".join(parts) # Display with spaces
            else:
                term_display = term_raw.replace("_", " ")


            pval_formatted = f"{pval:.3f}".lstrip('0') if pval >= 0.001 else "<.001"
            if pval_formatted == ".000": pval_formatted = "<.001" # Handle exact .000 rounding

            final_data.append({
                "Category": cat,
                "Term": term_display,
                "Internal Frequency": x,
                "Global Frequency": K_corpus,
                "Test Value": round(test_val, 4),
                "P-Value": pval_formatted
            })

    result_df = pd.DataFrame(final_data).sort_values(by=["Category", "Test Value"], ascending=[True, False])


    # Corpus-level statistics (based on processed unigrams)
    num_total_tokens_corpus_unigrams = sum(corpus_word_freq_for_stats.values())
    num_total_types_corpus_unigrams = len(corpus_word_freq_for_stats)
    morph_complexity_corpus = (
        round(num_total_types_corpus_unigrams / num_total_tokens_corpus_unigrams, 2)
        if num_total_tokens_corpus_unigrams > 0 else 0.00
    )
    num_hapax_corpus = sum(1 for freq in corpus_word_freq_for_stats.values() if freq == 1)

    # Category-level statistics
    category_stats_results = {}
    # We need to re-process text for category stats if we want doc counts etc.
    # Or, better, get this from the main loop if possible.
    # For now, re-iterating for cat stats for simplicity, but using already grouped text.
    
    # Pre-calculate number of instances per category
    category_instance_counts = df[category_col].value_counts().to_dict()

    for cat_val in categories:
        num_instances = category_instance_counts.get(cat_val, 0)
        
        # For tokens, types, hapax within category, we need to iterate texts of that category
        cat_texts_series = df.loc[df[category_col] == cat_val, text_col]
        
        cat_total_tokens_unigram = 0
        cat_types_set_unigram = set()

        for text_content in cat_texts_series.dropna().astype(str):
            # Minimal processing: tokenize, apply stopwords, stem (consistent with corpus_word_freq_for_stats)
            # We use the same base unigrams as for corpus_word_freq_for_stats
            _, _, cat_base_unigrams = preprocess_text_optimized(
                text_content, lang_stemmer_obj, stem_words, [1], # Only unigrams for these stats
                current_stopwords_set, group_names_set
            )
            cat_total_tokens_unigram += len(cat_base_unigrams)
            cat_types_set_unigram.update(cat_base_unigrams)

        cat_total_types_unigram = len(cat_types_set_unigram)
        cat_morph_complexity = (
            round(cat_total_types_unigram / cat_total_tokens_unigram, 2) 
            if cat_total_tokens_unigram > 0 else 0.00
        )
        # Hapax: types in this category that are hapax in the *corpus* (based on consistent token processing)
        cat_num_hapax = sum(1 for token_type in cat_types_set_unigram if corpus_word_freq_for_stats.get(token_type, 0) == 1)

        category_stats_results[cat_val] = {
            "Number of Instances": num_instances,
            "Number of Tokens": cat_total_tokens_unigram, # Unigram tokens after processing
            "Number of Types": cat_total_types_unigram,   # Unigram types after processing
            "Morphological Complexity": cat_morph_complexity,
            "Number of Hapax": cat_num_hapax 
        }
    
    progress.progress(100)
    return (result_df, categories, len(categories), 
            num_total_tokens_corpus_unigrams, num_total_types_corpus_unigrams, 
            morph_complexity_corpus, num_hapax_corpus, category_stats_results)


def visualize_and_display(category, cat_df, category_stats_dict, alpha, top_n=10):
    st.subheader(f"Category: {category}")
    stats = category_stats_dict.get(category, {})
    st.markdown(f"""
    - **Number of Instances (Documents):** {stats.get('Number of Instances', 0)}
    - **Number of Tokens (Processed Unigrams):** {stats.get('Number of Tokens', 0)}
    - **Number of Types (Processed Unigrams):** {stats.get('Number of Types', 0)}
    - **Morphological Complexity (Types/Token Ratio):** {stats.get('Morphological Complexity', 0.00)}
    - **Number of Hapax (Unique category words that are hapax in corpus):** {stats.get('Number of Hapax', 0)}
    """)

    display_table = cat_df[['Term', 'Internal Frequency', 'Global Frequency', 'Test Value', 'P-Value']]
    st.dataframe(display_table.reset_index(drop=True), use_container_width=True)

    if display_table.empty:
        st.write(f"No significant characteristic words found for category **{category}**.")
        return

    positive_subset = display_table[display_table['Test Value'] > 0].sort_values('Test Value', ascending=False).head(top_n)
    negative_subset = display_table[display_table['Test Value'] < 0].sort_values('Test Value').head(top_n)
    
    # Ensure at least one row for plotting if only one type exists
    if positive_subset.empty and negative_subset.empty:
        st.write(f"No characteristic words with positive or negative test values for category **{category}** to plot.")
        return
    
    combined_subset = pd.concat([positive_subset, negative_subset])
    if combined_subset.empty: # Should be caught by above, but as a safeguard
        st.write(f"No characteristic words to plot for category **{category}**.")
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
        height=max(400, len(combined_subset) * 30 + 100), # Dynamic height
        color_discrete_map={"Overrepresented": "blue", "Underrepresented": "red"}
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder='total ascending', # Sorts y-axis by the x value
            showgrid=False
        ),
        legend_title_text='Word Representation',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
    )

    fig.add_shape( # Vertical line at x=0
        type='line', x0=0, y0=-0.5, x1=0, y1=len(combined_subset) - 0.5,
        line=dict(color='black', dash='dash')
    )
    st.plotly_chart(fig, use_container_width=True)


def display_results(result_df, categories, num_categories, total_tokens, total_types, morphological_complexity, num_hapax, alpha, category_stats):
    st.write("### 📈 Corpus Summary Statistics (based on processed unigrams)")
    st.markdown(f"""
    - **Number of Categories:** {num_categories}
    - **Total Number of Processed Unigram Tokens:** {total_tokens}
    - **Total Number of Unique Processed Unigram Types:** {total_types}
    - **Morphological Complexity (Types/Token Ratio):** {morphological_complexity}
    - **Number of Hapax (Processed unigrams appearing once in corpus):** {num_hapax}
    - **Significance Level (alpha for B-H correction):** {alpha}
    """)

    st.write("### 📄 Characteristic Words Tables and Visualizations")
    st.markdown("""
    **Table Explanation:**
    - **Category:** The specific group or category within your dataset.
    - **Term:** The characteristic word or n-gram.
    - **Internal Frequency:** Number of times the term appears within the category.
    - **Global Frequency:** Number of times the term appears across the entire corpus.
    - **Test Value:** 
      $$\\log_2\\left(\\frac{x/n_c}{K/M}\\right)$$
      where:
        - *x* = Internal Frequency of the term in the category.
        - *n<sub>c</sub>* = Total terms (sum of analysis n-gram frequencies) in the category.
        - *K* = Global Frequency of the term in the corpus.
        - *M* = Total terms (sum of analysis n-gram frequencies) in the corpus.
      This value indicates how much more (or less) the term is associated with the category compared to the corpus.
    - **P-Value:** The Benjamini-Hochberg corrected p-value from a hypergeometric test. Displayed with three decimal points (e.g., .035) or as "<.001" if p < .001.
    """)

    if result_df.empty:
        st.warning("No significant characteristic words found based on the provided criteria.")
    else:
        for cat in sorted(categories): # Ensure consistent order
            cat_df = result_df[result_df['Category'] == cat]
            visualize_and_display(cat, cat_df, category_stats, alpha, top_n=10)


def download_results():
    st.write("### ⬇️ Download Results")
    if 'result_df' not in st.session_state or st.session_state['result_df'].empty:
        st.info("No results to download.")
        return
    
    result_df_to_download = st.session_state['result_df']
    download_options = st.multiselect(
        "Select download format(s):",
        options=["CSV", "Excel"],
        default=["CSV"]
    )

    if not download_options:
        st.info("Select at least one format to download the results.")
        return

    download_files = {}
    if "CSV" in download_options:
        csv_buffer = StringIO()
        result_df_to_download.to_csv(csv_buffer, index=False, encoding='utf-8')
        download_files["characteristic_words.csv"] = (csv_buffer.getvalue().encode('utf-8'), "text/csv")
    
    if "Excel" in download_options:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            result_df_to_download.to_excel(writer, index=False, sheet_name='Characteristic Words')
        download_files["characteristic_words.xlsx"] = (excel_buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    if len(download_files) > 1:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf: # Use 'w' for new zip
            for filename, (data_bytes, _) in download_files.items():
                zipf.writestr(filename, data_bytes)
        st.download_button(
            label="📥 Download Selected Formats as ZIP",
            data=zip_buffer.getvalue(),
            file_name="characteristic_words.zip",
            mime="application/zip"
        )
    elif len(download_files) == 1:
        filename, (data_bytes, mime_type) = list(download_files.items())[0]
        st.download_button(
            label=f"📥 Download Results as {filename.split('.')[-1].upper()}",
            data=data_bytes,
            file_name=filename,
            mime=mime_type
        )


def main():
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
    Statistical significance testing (hypergeometric test with Benjamini-Hochberg correction) ensures identified words are not occurring by chance.
    """)

    st.sidebar.header("🔧 Configuration")
    uploaded_file = st.sidebar.file_uploader("📂 Upload Your Data", type=["csv", "xlsx", "tsv", "txt"])

    # Initialize session_state variables
    default_session_state = {
        'result_df': pd.DataFrame(), 'categories': [], 'num_categories': 0,
        'total_tokens': 0, 'total_types': 0, 'morphological_complexity': 0.00,
        'num_hapax': 0, 'category_stats': {}, 'word_groups': []
    }
    for var, default_val in default_session_state.items():
        if var not in st.session_state:
            st.session_state[var] = default_val

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
            
            st.sidebar.write("### Available Columns:")
            st.sidebar.write(df.columns.tolist())

            st.sidebar.write("### Data Preview:")
            st.sidebar.dataframe(df.head())

            text_col = st.sidebar.selectbox("Select the text column", options=df.columns, index=0 if df.columns.n_items > 0 else None)
            category_col = st.sidebar.selectbox("Select the category column", options=df.columns, index=1 if df.columns.n_items > 1 else (0 if df.columns.n_items > 0 else None) )

            st.sidebar.write("### Word Grouping")
            if st.sidebar.button("➕ Add Word Group"):
                st.session_state['word_groups'].append({'name': '', 'words': []})
                st.experimental_rerun() # Rerun to update UI for new group

            word_group_mapping_input = {} # Store user input, will be processed to lowercase
            groups_to_remove_indices = []

            for idx, group_data in enumerate(st.session_state['word_groups']):
                with st.sidebar.expander(f"Group {idx+1}", expanded=True):
                    group_name_input = st.text_input(f"Group {idx+1} Name", value=group_data['name'], key=f"group_name_{idx}")
                    group_words_str_input = st.text_input(f"Group {idx+1} Words (comma-separated)", value=", ".join(group_data['words']), key=f"group_words_{idx}")
                    
                    st.session_state['word_groups'][idx]['name'] = group_name_input
                    st.session_state['word_groups'][idx]['words'] = [w.strip() for w in group_words_str_input.split(',') if w.strip()]

                    if st.checkbox(f"🗑️ Remove Group {idx+1}", key=f"remove_group_{idx}"):
                        groups_to_remove_indices.append(idx)
            
            if groups_to_remove_indices:
                for i in sorted(groups_to_remove_indices, reverse=True):
                    st.session_state['word_groups'].pop(i)
                st.experimental_rerun()


            # Process word_group_mapping_input into lowercase for analysis
            word_group_mapping_lower = {}
            valid_groups = True
            for group_data in st.session_state['word_groups']:
                g_name = group_data['name'].strip()
                g_words = [w.lower() for w in group_data['words'] if w] # Ensure words are lowercase
                if g_name:
                    if '_' in g_name:
                        st.sidebar.error(f"Group name '{g_name}' cannot contain underscores ('_'). Please rename.")
                        valid_groups = False
                    elif not re.fullmatch(r'\w+', g_name): # Ensure group name itself is like a single token
                        st.sidebar.error(f"Group name '{g_name}' should be a single 'word' (alphanumeric characters and underscores, but underscores are disallowed by previous check).")
                        valid_groups = False
                    else:
                         word_group_mapping_lower[g_name.lower()] = g_words


            st.sidebar.write("### Stopword Removal")
            remove_sw_option = st.sidebar.checkbox("🗑️ Remove stopwords?", value=False)
            lang_choice_for_sw = "English" # Default
            if remove_sw_option:
                lang_choice_for_sw = st.sidebar.selectbox("🌐 Select language for stopwords", list(LANGUAGES.keys()))

            st.sidebar.write("### Custom Stopwords")
            custom_stopword_input_str = st.sidebar.text_area("✍️ Enter custom stopwords (one per line or comma-separated):")
            
            custom_stopwords_processed = []
            if custom_stopword_input_str:
                # Split by comma or newline, strip whitespace, lowercase, and filter empty strings
                raw_custom_stopwords = re.split(r'[,\n]', custom_stopword_input_str)
                custom_stopwords_processed = [word.strip().lower() for word in raw_custom_stopwords if word.strip()]


            st.sidebar.write("### Stemming")
            stem_words_option = st.sidebar.checkbox("🪓 Apply stemming?", value=True)

            st.sidebar.write("### N-gram Selection")
            ngram_options_selected = st.sidebar.multiselect(
                "Select N-grams to consider (e.g., Unigrams for single words, Bigrams for two-word phrases)",
                options=["Unigrams", "Bigrams", "Trigrams"],
                default=["Unigrams"]
            )
            ngram_map = {"Unigrams": 1, "Bigrams": 2, "Trigrams": 3}
            ngram_ranges_selected = sorted(list(set(ngram_map[ng] for ng in ngram_options_selected))) # Ensure unique and sorted

            if not ngram_ranges_selected:
                st.sidebar.error("⚠️ Please select at least one N-gram option.")
                # Do not proceed with analysis if no n-grams selected
            
            st.sidebar.write("### Minimum Global Frequency")
            min_freq_val = st.sidebar.number_input(
                "🔢 Minimum global frequency for a term to be considered:",
                min_value=1, max_value=1000, value=2, step=1,
                help="Terms appearing less than this many times in the entire corpus will be ignored."
            )

            st.sidebar.write("### Significance Level")
            alpha_val = st.sidebar.slider(
                "📉 Significance level (alpha) for Benjamini-Hochberg:", 
                min_value=0.001, max_value=0.25, value=0.05, step=0.001, format="%.3f"
            )

            if st.sidebar.button("🚀 Run Analysis"):
                if not ngram_ranges_selected:
                    st.error("⚠️ Please select at least one N-gram option to proceed.")
                elif not text_col or not category_col:
                    st.error("⚠️ Please select both text and category columns.")
                elif not valid_groups:
                    st.error("⚠️ Please correct errors in word group names.")
                elif text_col not in df.columns or category_col not in df.columns:
                     st.error(f"⚠️ One or both selected columns ('{text_col}', '{category_col}') do not exist in the uploaded file.")
                else:
                    st.header("🔍 Analysis Results")
                    st.write("### Processing...")
                    analysis_progress_bar = st.progress(0)

                    with st.spinner("🕒 Analyzing the corpus... this may take a while for large datasets."):
                        analysis_results = perform_analysis(
                            df.copy(), # Pass a copy to avoid modifying original loaded df
                            text_col,
                            category_col,
                            remove_sw_option,
                            stem_words_option,
                            lang_choice_for_sw, # This is the language for built-in stopwords and stemmer
                            ngram_ranges_selected,
                            min_freq_val,
                            alpha_val,
                            custom_stopwords_processed,
                            word_group_mapping_lower, # Pass the processed lowercase mapping
                            analysis_progress_bar
                        )
                        
                        if len(analysis_results) == 8:
                            (res_df, cats, num_cats, tot_toks, tot_types, morph_comp, n_hapax, cat_stats) = analysis_results
                            
                            st.session_state['result_df'] = res_df
                            st.session_state['categories'] = cats
                            st.session_state['num_categories'] = num_cats
                            st.session_state['total_tokens'] = tot_toks
                            st.session_state['total_types'] = tot_types
                            st.session_state['morphological_complexity'] = morph_comp
                            st.session_state['num_hapax'] = n_hapax
                            st.session_state['category_stats'] = cat_stats
                            
                            display_results(res_df, cats, num_cats, tot_toks, tot_types, morph_comp, n_hapax, alpha_val, cat_stats)
                        else:
                            st.error("⚠️ An error occurred during analysis. Please check your parameters and data.")
                            # Optionally log `analysis_results` or the exception if it were caught and returned

        except ValueError as ve:
            st.sidebar.error(f"⚠️ Value Error: {ve}")
        except Exception as e:
            st.sidebar.error(f"⚠️ An unexpected error occurred: {e}")
            st.exception(e) # Show full traceback for debugging
    else:
        st.sidebar.info("📥 Awaiting file upload.")

    # Download section, always visible if results exist
    if not st.session_state['result_df'].empty:
        download_results()

if __name__ == "__main__":
    main()
    add_footer()
