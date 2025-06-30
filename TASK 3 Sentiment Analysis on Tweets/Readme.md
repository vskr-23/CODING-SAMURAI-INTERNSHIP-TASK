## Sentiment Analysis: VADER vs RoBERTa Comparison

A comprehensive sentiment analysis project comparing traditional lexicon-based approaches (VADER) with modern transformer-based models (RoBERTa) on Amazon product reviews.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Models Compared](#models-compared)
- [Visualizations](#visualizations)
- [Results](#results)
- [Contributing](#contributing)

## 🔍 Overview

This project performs a comprehensive comparison between two different sentiment analysis approaches:
1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)** - A lexicon-based sentiment analysis tool
2. **RoBERTa** - A transformer-based model fine-tuned for sentiment classification

The analysis is performed on Amazon product reviews to understand how different models perform on real-world customer feedback data.

## ✨ Features

- **Data Preprocessing**: Text cleaning and preparation for sentiment analysis
- **Exploratory Data Analysis**: Distribution analysis of review scores
- **Text Processing**: Tokenization using NLTK's TreebankWordTokenizer
- **NLP Pipeline**: Part-of-speech tagging and named entity recognition using spaCy
- **Dual Model Comparison**: Side-by-side comparison of VADER and RoBERTa sentiment scores
- **Comprehensive Visualizations**: Multiple chart types showing model performance
- **Statistical Analysis**: Correlation analysis between different sentiment metrics

## 📊 Dataset

- **Source**: Amazon product reviews dataset (`Reviews.csv`)
- **Sample Size**: 500 reviews (subset for analysis)
- **Features**: Review text, star ratings (1-5), and metadata
- **Format**: CSV file with columns including 'Text' and 'Score'

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Required Libraries
```bash
# Core data science libraries
pip install numpy pandas matplotlib seaborn

# NLP libraries
pip install nltk spacy

# Transformer models
pip install transformers torch scipy

# Progress bars
pip install tqdm

# Download required NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### Alternative Installation
```bash
pip install -r requirements.txt
```

## 📖 Usage

1. **Clone the repository**:
   ```bash
   git clone [your-repository-url]
   cd sentiment-analysis-comparison
   ```

2. **Ensure you have the dataset**:
   - Place `Reviews.csv` in the project root directory
   - The dataset should contain at minimum 'Text' and 'Score' columns

3. **Run the analysis**:
   ```bash
   jupyter notebook sentiment_analysis.ipynb
   ```

4. **Execute all cells** to perform the complete analysis pipeline

## 🔬 Methodology

### 1. Data Exploration
- Load and inspect the Amazon reviews dataset
- Analyze the distribution of review scores (1-5 stars)
- Visualize review score frequencies

### 2. Text Preprocessing
- Extract sample reviews for detailed analysis
- Perform tokenization using NLTK's TreebankWordTokenizer
- Apply spaCy for advanced NLP processing (POS tagging, NER)

### 3. Sentiment Analysis
- **VADER Analysis**: Apply VADER sentiment analyzer to extract compound, positive, neutral, and negative scores
- **RoBERTa Analysis**: Use pre-trained `cardiffnlp/twitter-roberta-base-sentiment` model for sentiment classification
- Process all reviews through both models

### 4. Comparative Analysis
- Merge results from both models
- Create visualizations comparing sentiment scores
- Analyze correlations between different sentiment metrics

## 🤖 Models Compared

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Type**: Lexicon-based sentiment analysis
- **Outputs**: Compound score, positive, neutral, negative scores
- **Advantages**: Fast, interpretable, works well with social media text
- **Use Case**: Quick sentiment analysis, real-time applications

### RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment`
- **Type**: Transformer-based neural network
- **Outputs**: Negative, neutral, positive probability scores
- **Advantages**: Context-aware, handles complex language patterns
- **Use Case**: High-accuracy sentiment analysis, complex text understanding

## 📈 Visualizations

The project generates several key visualizations:

1. **Review Score Distribution**: Bar chart showing frequency of 1-5 star reviews
2. **VADER Compound Scores**: Sentiment scores grouped by star ratings
3. **Individual Sentiment Components**: Separate analysis of positive, neutral, and negative scores
4. **Model Comparison**: Pair plots comparing VADER and RoBERTa outputs
5. **Correlation Analysis**: Heatmaps showing relationships between different sentiment metrics

## 📊 Results

The analysis provides insights into:
- How well each model correlates with actual star ratings
- Strengths and weaknesses of lexicon-based vs. transformer-based approaches
- Distribution patterns of sentiment scores across different rating levels
- Comparative performance metrics between VADER and RoBERTa

## 🛠️ Dependencies

### Core Libraries
- `numpy`: Numerical computing
- `pandas`: Data manipulation and analysis
- `matplotlib`: Basic plotting
- `seaborn`: Statistical data visualization

### NLP Libraries
- `nltk`: Natural Language Toolkit for tokenization and VADER
- `spacy`: Advanced NLP processing
- `transformers`: Hugging Face transformers for RoBERTa
- `torch`: PyTorch for transformer model execution
- `scipy`: Scientific computing for softmax function

### Utility Libraries
- `tqdm`: Progress bars for long-running operations

## 🔧 Configuration

### Model Settings
- **VADER**: Uses default NLTK VADER configuration
- **RoBERTa**: Uses `cardiffnlp/twitter-roberta-base-sentiment` with default tokenizer settings

### Analysis Parameters
- **Sample Size**: 500 reviews (configurable)
- **Visualization Style**: 'ggplot' matplotlib style
- **Color Palettes**: 'viridis' and 'tab10' for different chart types

## 📈 Performance Considerations

- **VADER**: Fast processing, suitable for real-time analysis
- **RoBERTa**: Slower but more accurate, requires GPU for large datasets
- **Memory Usage**: RoBERTa model requires ~500MB memory
- **Processing Time**: Approximately 2-5 minutes for 500 reviews (CPU)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Notes

- Ensure sufficient memory when processing large datasets with RoBERTa
- The notebook includes error handling for failed sentiment analysis attempts
- Results are cached in DataFrame format for easy manipulation and export

