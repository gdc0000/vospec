# Characteristic Words Detection in Corpus Linguistics

**Characteristic Words Detection** is a Streamlit application designed for corpus linguistics research. It identifies characteristic words (or n-grams) within categories of your dataset by comparing the frequency of words in each category with their global frequency in the corpus. The app applies statistical testing (hypergeometric tests with Benjamini–Hochberg correction) to determine significant over- or under-representation of terms.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This app processes text data by:
- **Preprocessing:** Tokenizing text, removing stopwords (customizable by language), and applying stemming.
- **N-gram Generation:** Supports unigrams, bigrams, and trigrams.
- **Word Grouping:** Optionally groups words together under a common label.
- **Statistical Analysis:** Computes internal and global frequencies, applies hypergeometric tests, and corrects p-values using the Benjamini–Hochberg procedure.
- **Visualization & Results:** Displays summary statistics, characteristic words tables, and interactive bar charts (via Plotly). Results can be downloaded in CSV and/or Excel formats.

The application is ideal for linguists, social scientists, and researchers analyzing large text corpora to identify thematic or stylistic markers.

---

## Features

- **Customizable Text Preprocessing:**  
  - Tokenization and stopword removal (supports English, French, Italian, and Spanish).
  - Optional stemming using Snowball stemmers.
  - N-gram selection (unigrams, bigrams, trigrams).
  
- **Word Grouping:**  
  - Replace sets of words with a group name for consistent treatment.
  
- **Statistical Testing:**  
  - Uses hypergeometric distribution to assess term significance.
  - Applies Benjamini–Hochberg correction for multiple comparisons.
  
- **Visualization:**  
  - Generates interactive bar charts showing overrepresented and underrepresented terms.
  
- **Downloadable Results:**  
  - Export the results in CSV, Excel, or ZIP format.

- **Session Persistence:**  
  - Utilizes Streamlit's session state to store analysis results and configuration.

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/characteristic-words-detection.git
   cd characteristic-words-detection
   ```

2. **(Optional) Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   The required packages are listed in the [`requirements.txt`](./requirements.txt) file. Install them using:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Run the Streamlit App**

   Launch the application by running:

   ```bash
   streamlit run main.py
   ```

2. **Upload Your Data**

   - Upload a CSV, Excel, TSV, or TXT file containing your corpus.
   - Use the sidebar to preview your data and select the text and category columns.

3. **Configure Analysis Settings**

   - **Word Grouping:** Add word groups and provide a name and comma-separated words.
   - **Stopword Removal:** Enable stopword removal and select a language. Optionally, add custom stopwords.
   - **Stemming:** Choose whether to apply stemming.
   - **N-gram Selection:** Select which n-grams to consider (unigrams, bigrams, trigrams).
   - **Minimum Frequency & Significance Level:** Set the minimum frequency threshold and significance level (alpha).

4. **Run Analysis**

   Click the **"Run Analysis"** button. A progress bar will update as the corpus is processed. The app displays:
   - Summary statistics (number of tokens, types, morphological complexity, etc.)
   - A table of significant characteristic words with their internal and global frequencies, test values, and p-values.
   - Interactive visualizations for each category.

5. **Download Results**

   Once the analysis is complete, select your preferred download format(s) (CSV and/or Excel) and download the results.

---

## File Structure

```
.
├── main.py           # Main Streamlit application code for characteristic words detection
├── requirements.txt  # Required Python packages and versions
└── README.md         # This file
```

---

## Requirements

The app requires the following packages (with minimum versions):

- streamlit >= 1.40.2
- pandas >= 2.1.1
- numpy >= 1.25.3
- stop-words >= 0.2.5
- snowballstemmer >= 2.1.0
- scipy >= 1.10.1
- plotly >= 5.17.0
- openpyxl >= 3.1.2

---

## Contributing

Contributions are welcome! If you have suggestions, bug fixes, or improvements:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Open a pull request with a detailed description of your modifications.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Contact

**Gabriele Di Cicco, PhD in Social Psychology**  
[GitHub](https://github.com/gdc0000) | [ORCID](https://orcid.org/0000-0002-1439-5790) | [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)

---

Happy Analyzing!
