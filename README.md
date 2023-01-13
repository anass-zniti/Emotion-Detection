# Sentiment Analysis and Emotion Detection from Tweets

## Project Overview
This document details two related projects focusing on:
1. **Sentiment Analysis**: Classifying tweets as positive, negative, or neutral.
2. **Emotion Detection**: Identifying specific emotions (e.g., Joy, Sadness, Anger) in tweets using advanced machine learning techniques.

Both projects leverage datasets like the **Huggingface Emotion Dataset**, **NRC Emotion Intensity Lexicon**, and **Sentiment140**, and explore various machine learning architectures to achieve their goals.

---

## Sentiment Analysis

### Dataset Used
**Sentiment140 Dataset**:
- Contains tweets labeled as Positive, Negative, or Neutral.

### Methodology
- **Model**: Logistic Regression.
- **Preprocessing**:
  - Tokenization, normalization, and stopword removal.
  - Features extracted using `CountVectorizer` with n-grams.
- **Metrics**: Accuracy, Precision, Recall, F1-score.

### Results
- **Accuracy**: **78%**.
- Successfully applied to real-time sentiment classification of tweets.

---

## Emotion Detection

### Datasets Used
1. **Huggingface Emotion Dataset**:
   - Labeled tweets for six emotions: Joy, Sadness, Anger, Fear, Love, and Surprise.
2. **NRC Emotion Intensity Lexicon**:
   - Provides lexicon-based scores for emotion intensity.

### Methodology
- **Models**:
  - **CNN**: Extracts local features from text using convolutional layers.
  - **LSTM**: Captures long-term dependencies in sequential text data.
  - **CNN-LSTM Hybrid**: Combines local feature extraction (CNN) with sequential modeling (LSTM).
- **Pretrained Word Embeddings**:
  - Used **GloVe** embeddings (300-dimensional vectors).
- **Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrices.

### Results
- **LSTM Model** achieved the highest performance:
  - **Validation Accuracy**: **89%**.
  - **Best Emotion Prediction**: Sadness (F1-score: **0.94**).
- **CNN Model**: ~85% accuracy.
- **CNN-LSTM Hybrid**: ~86% accuracy.

---

## Combined Key Results

- **Real-Time Predictions**:
  - Both models demonstrated robust performance when applied to live tweet streams.
- **Emotion Detection Models** outperformed Sentiment Analysis in classification complexity and accuracy.

---

## Project Structure

### Sentiment Analysis
- Data Preprocessing: Cleaning, tokenization, and balancing datasets.
- Model Training: Logistic regression implementation.
- Real-Time Sentiment Analysis: Integration with live Twitter data streams.

### Emotion Detection
- Data Preprocessing: Tokenization, stopword removal, and dataset augmentation.
- Model Training:
  - CNN, LSTM, and CNN-LSTM hybrid architectures.
- Evaluation: Detailed metrics analysis and comparison.

---

## Conclusion
These projects highlight the effectiveness of machine learning models in:
- Classifying sentiment polarity in tweets.
- Identifying nuanced emotions in text data.

The **LSTM model** proved to be the most effective overall, achieving high accuracy and robustness across diverse datasets and real-world scenarios.

---

## Usage

### Sentiment Analysis
1. Preprocess datasets.
2. Run scripts in `Sentiment_analysis.ipynb`.
3. Connect to Twitter API for live sentiment analysis.

### Emotion Detection
1. Preprocess datasets and create embeddings.
2. Train models using `Emotion_Detection_CNN_LSTM_HYBRID.ipynb`.
3. Apply trained models to real-time tweet streams.

---

## Future Directions

- Incorporating multilingual datasets to enhance generalizability.
- Experimenting with transformer-based models (e.g., **BERT**, **GPT**) for improved performance.

---

## Repositories

1. **Sentiment Analysis**: [GitHub Repository](https://github.com/anass-zniti/Sentiment-analysis.git)
2. **Emotion Detection**: [GitHub Repository](https://github.com/anass-zniti/Emotion-Detection.git)

---

## Acknowledgments
Supervised by **Dr. Kate MacFarlane**, Faculty of Technology, University of Sunderland.
