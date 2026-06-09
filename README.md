# Spam Message Detection using RNN

## Overview

This project was built to practice and understand the working of Recurrent Neural Networks (RNNs) for Natural Language Processing (NLP).

To study how different text representations affect RNN performance, two approaches were implemented and compared:

1. TF-IDF Vectorization + RNN
2. Word Embeddings + RNN

The project demonstrates how the choice of text representation can significantly impact model performance.

---

## Dataset

Dataset: SMS Spam Collection Dataset

Source: Kaggle

Classes:

* Ham (Normal Message)
* Spam (Unwanted Promotional Message)

---

## Preprocessing

The following preprocessing steps were performed:

* Lowercasing
* Tokenization
* Stemming
* Vocabulary Creation
* Sequence Padding (for Word Embedding approach)

---

## Approach 1: TF-IDF + RNN

Pipeline:

Text → Preprocessing → TF-IDF Vectorization → RNN → Prediction

### Accuracy

**98.45%**

### Observation

The TF-IDF representation performed extremely well on this dataset. Since spam messages often contain distinctive keywords such as promotional terms, prizes, rewards, and offers, TF-IDF was able to capture these patterns effectively.

---

## Approach 2: Word Embeddings + RNN

Pipeline:

Text → Word Indexing → Embedding Layer → RNN → Prediction

### Accuracy

**85.88%**

### Observation

This approach allows the RNN to process messages sequentially and learn word representations through embeddings. Although the accuracy was lower than the TF-IDF approach, it provided valuable hands-on experience with sequence modeling, embeddings, hidden states, and RNN architecture.

---

## Performance Comparison

| Approach              | Accuracy |
| --------------------- | -------- |
| TF-IDF + RNN          | 98.45%   |
| Word Embeddings + RNN | 85.88%   |

---

## Key Learnings

* Text preprocessing for NLP tasks
* Tokenization and stemming
* TF-IDF vectorization
* Vocabulary creation and word indexing
* Sequence padding
* Word Embeddings
* RNN architecture and hidden states
* Binary text classification using PyTorch
* Impact of different text representations on model performance

---

## Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* NLTK
* Scikit-learn

---

## Conclusion

For this dataset, TF-IDF combined with an RNN achieved significantly higher accuracy than Word Embeddings with an RNN. This highlights the importance of choosing an appropriate text representation and shows that a more complex representation does not always guarantee better performance.

This project was completed primarily to gain practical experience with RNNs, embeddings, sequence processing, and NLP workflows in PyTorch.

