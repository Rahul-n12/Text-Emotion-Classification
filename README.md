# Emotion Classification from Text Comments

This project is a comprehensive pipeline to classify emotions (Anger, Joy, Fear) from textual comments using various machine learning and deep learning models. The dataset contains 5,937 labeled examples with three classes.

---

## 📊 Dataset Overview

- **Total Records:** 5,937
- **Features:** 
  - Comment (text)
  - Emotion (anger, joy, fear)
- **Target:** Emotion
- **Distribution:**
  - Anger: 2000
  - Joy: 2000
  - Fear: 1937

---

## 🧹 Data Preprocessing

- Removed stopwords
- Lemmatized and stemmed words
- Tokenized the text
- Added `Preprocessed_Comment` feature

---

## 📈 Visualizations

- **Word Clouds** were generated for each emotion to show frequent words.

---

## 🔁 Label Encoding

Mapped the `Emotion` column to `Emotion_num`:
- Anger = 0
- Fear = 1
- Joy = 2

---

## 🔀 Train-Test Split

- **TF-IDF Vectorization** for ML models
- **Tokenizer + Padding** for LSTM models
- **Train/Test Split Ratio:** 70/30 for ML, 80/20 for DL

---

## 🤖 Models Used

### ✅ Traditional ML Models:
| Model                | Accuracy |
|---------------------|----------|
| MultinomialNB        | 89%      |
| Logistic Regression  | 92%      |
| Random Forest        | **94%**  |
| XGBoost              | 93%      |

### ✅ Deep Learning Models:
| Model           | Accuracy |
|----------------|----------|
| LSTM            | 83%      |
| BiLSTM          | 88%      |
| BiLSTM (Tuned)  | **93.5%**|

---

## 🔍 Model Evaluation

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

---

## 🔧 Hyperparameter Tuning

Used **Keras Tuner (Random Search)** for BiLSTM:
- Units: 32 to 128
- Learning Rate: 0.01, 0.001, 0.0001

Best accuracy after tuning: **93.5%**

---

## ✅ Conclusion

- **Random Forest** performed best among traditional models.
- **BiLSTM (tuned)** came very close with 93.5% accuracy.

---

## 🚀 Future Scope

- Use more layers in LSTM/BiLSTM
- Apply attention mechanisms
- Try transformer-based models (BERT, RoBERTa)

---

## 🗂️ Dependencies

```bash
pip install numpy pandas scikit-learn seaborn matplotlib nltk xgboost tensorflow keras keras-tuner wordcloud
```

---
