# Sentiment Analysis on Twitter Using Deep Learning and Transformers

Final year project (B.Tech CSE, Data Science), Mohan Babu University, Tirupati.

This project compares CNN, LSTM and BERT models for classifying tweets into four sentiment classes (Positive, Negative, Neutral, Irrelevant) and proposes a hybrid **BERT + BiLSTM + CNN** model that performs best.

The work was also written up as an IEEE conference paper, *"A Review of Deep Learning and Transformer-Based Architectures for Social Media Sentiment Analysis"*, accepted at ICCMC 2026.

## Results

Dataset: 74,681 tweets, 4 classes.

| Model | Validation accuracy |
|---|---|
| BERT (fine-tuned) | 88.55% |
| LSTM | 89.5% |
| CNN | 90.1% |
| **BERT + BiLSTM + CNN (hybrid)** | **92.02%** (weighted F1: 92.01%) |

## Approach

1. **Preprocessing:** duplicate and null removal, stopword removal, tokenization and lemmatization (NLTK).
2. **Split:** stratified 80/20 train-validation split.
3. **Models:** CNN, LSTM, fine-tuned BERT, and the hybrid model that feeds BERT embeddings into parallel BiLSTM and CNN branches.
4. **Training:** only the last 4 BERT layers are fine-tuned, with dropout, gradient clipping and a learning-rate scheduler to reduce overfitting.
5. **Evaluation:** accuracy, precision, recall, F1-score and confusion matrix.
6. **Inference:** a prediction function returns the sentiment label with a confidence score.

## Repository structure
.
├── CNN.ipynb             # CNN baseline
├── LSTM.ipynb            # LSTM baseline
├── BERT.ipynb            # Fine-tuned BERT baseline
├── BERT_Enhanced.ipynb   # Hybrid BERT + BiLSTM + CNN model
├── twitter_training.csv
├── twitter_validation.csv
├── requirements.txt
└── README.md
## How to run

The notebooks were developed in Google Colab.

1. Open a notebook in [Google Colab](https://colab.research.google.com) (File > Upload notebook) or in Jupyter.
2. Upload `twitter_training.csv` into the same working directory as the notebook. The notebooks read it by file name.
3. Install the dependencies if needed: `pip install -r requirements.txt`
4. Run all cells. Use a GPU runtime for the BERT-based notebooks.

Running a notebook also generates `tweet_clean.csv` and the trained model weights, which are not stored in this repo.

## Dataset

Twitter entity sentiment dataset (`twitter_training.csv`, `twitter_validation.csv`) with tweets labelled Positive, Negative, Neutral or Irrelevant. All credit for the data goes to its original authors.

## Known limitations

The model can misclassify sarcasm and neutral, news-style tweets. Future work: multilingual support and better sarcasm handling.

## Tech stack

Python, PyTorch, Hugging Face Transformers, TensorFlow/Keras, NLTK, scikit-learn, Google Colab.

## Team and guide

- Guide: Dr. M. Sirish Kumar
- Team: Akula Nagaraju Gari Adarsh, Chappidi Charishma, Marapareddy Aswini, Bayanaboyana Vasu Yadav
