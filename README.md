# Sentiment Analysis on Text Reviews
📘 Overview

This project focuses on analyzing customer sentiments from text reviews using Natural Language Processing (NLP) and deep learning techniques. The primary goal is to classify reviews into extremly positive,positive,neutral, negative and extremly negative sentiments to understand public opinion during lockdown time

🎯 Objectives

Perform text preprocessing and cleaning for effective model training.

Represent text using pretrained GloVe embeddings for better contextual understanding.

Train a Bidirectional LSTM model to capture both past and future dependencies in textual data as it helps getting better context of text and is also used for name entity recognition


🧩 Dataset Overview

The dataset consists of customer reviews with corresponding sentiment labels.

Sentiments distribution - 

Positive              11422 
Negative               9917
Neutral                7713
Extremely Positive     6624
Extremely Negative     5481

⚙️ Data Preprocessing

Key steps included:

Text Cleaning – Removed stopwords, punctuation, numbers, and special characters.

Tokenization – Converted each review into sequences of tokens using Keras Tokenizer.

Padding – Standardized input length by padding sequences.

Embedding Preparation – Loaded pretrained GloVe (Global Vectors for Word Representation) embeddings to map words to dense vector representations.


🧠 Model Development

Architecture: Bidirectional Long Short-Term Memory (Bi-LSTM)

Embedding Layer: Initialized with pretrained GloVe vectors.

Regularization: Dropout layers to reduce overfitting.

Output Layer: Softmax activation for multi-class classification.

Optimizer: Adam

Loss Function: Categorical Crossentropy

📈 Evaluation Metrics

Model performance was measured using:

Accuracy

Precision, Recall, and F1-Score

Confusion Matrix

🚀 Results

The Bidirectional LSTM model achieved strong performance, effectively understanding contextual polarity in sentences.It achived following metrics  accuracy: 0.6571 - loss: 0.8371 - val_accuracy: 0.7053 - val_loss: 0.8093 

🛠️ Tools and Libraries

Python

Pandas, NumPy

Matplotlib, Seaborn

TensorFlow / Keras

NLTK, GloVe
