# ✨ Sentiment Analysis

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![NLP](https://img.shields.io/badge/NLP-Scikit--learn-brightgreen.svg)
![Pandas](https://img.shields.io/badge/Data-Pandas-yellow.svg)
![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-orange)
![IDE-VSCode](https://img.shields.io/badge/IDE-VS%20Code-007ACC?logo=visualstudiocode&logoColor=white)
![Platform-Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## Overview
This project demonstrates a **machine learning pipeline for sentiment analysis** using real-world Twitter data related to airline customer feedback. It is designed as a professional AI portfolio piece to showcase proficiency in natural language processing (NLP), data preprocessing, model training, and evaluation.

The ultimate goal is to develop a scalable sentiment analysis tool that can be extended to platforms such as **WhatsApp messages**, **live chat systems**, or **customer support tools**.

---

## 🎯 Project Goals
- Build a production-ready sentiment classifier using classical ML algorithms.
- Demonstrate applied knowledge of NLP and text preprocessing techniques.
- Use clean, real-world data from **Kaggle** to train and evaluate the model.
- Offer a visual explanation of model performance using classification metrics and confusion matrices.
- Prepare for future expansion with deep learning models or integration into web interfaces.

---

## 📁 Dataset
- Source: [Kaggle - Airline Tweets Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Number of tweets: ~15,000
- Sentiment labels: `positive`, `neutral`, `negative`

---

## ⚙️ Workflow
1. **Data Loading & Cleaning**  
   Load CSV data, remove noise, lowercase text, strip special characters.

2. **Text Vectorization**  
   Apply `CountVectorizer` to convert text into numerical features.

3. **Model Training**  
   Train a `Multinomial Naive Bayes` model using `scikit-learn`.

4. **Model Evaluation**  
   Generate a classification report and confusion matrix to visualize model accuracy and class-wise performance.

---

## 📈 Model Performance
- **Overall Accuracy**: 78%
- Strong performance on the `negative` class (Precision: 0.78, Recall: 0.96)
- Moderate performance on the `positive` class (Precision: 0.82, Recall: 0.55)
- Lower performance on the `neutral` class (Precision: 0.72, Recall: 0.35) – affected by class imbalance

**Classification Report Sample:**
```
          precision    recall  f1-score   support

negative       0.78      0.96      0.86      1889
 neutral       0.72      0.35      0.47       580
positive       0.82      0.55      0.66       459

accuracy                           0.78      2928

macro avg 0.77 0.62 0.66 2928
weighted avg 0.77 0.78 0.75 2928
```
---

## 🔍 Example Prediction
```python
model.predict(vectorizer.transform(["I love this airline!"]))
# Output: ['positive']
```

## 🧠 Future Plans
- Replace `CountVectorizer` with `TF-IDF` or deep embeddings like `Word2Vec` or `BERT`
- Try stronger models such as Logistic `Regression`, `SVM`, or deep learning classifiers
- Address class imbalance using `SMOTE` or class weighting
- Deploy the model as a simple web app using `Streamlit` or `Gradio`
- Extend to multilingual sentiment analysis (Arabic/English)
- Create a real-time API for processing WhatsApp or chat messages

## 🚀 Setup & Run
1. Install requirements
```
  pip install -r requirements.txt
```

2. Run the notebook
Open `Sentiment_Analysis_Project.ipynb` in Jupyter Notebook or VS Code.

## 📦 File Structure
```
📁 Sentiment_Analysis
│
├── Sentiment_Analysis_Project.ipynb       # Main notebook
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
│
└── 📁 Data_Explorer
    ├── Tweets.csv                         # Raw dataset
    └── database.sqlite                    # SQLite version of dataset
```
---

## 📌 Author
### Omar Khamis
AI & Robotics Enthusiast | Python Developer
- 💼 [LinkedIn](https://www.linkedin.com/in/omar-khamis-dev)
 | 💻 [GitHub](https://github.com/omar-khamis-dev)
- 📧 Email: omar.khamis.dev@gmail.com

---

## 📜 License
This project is licensed under the MIT License – feel free to use and adapt with attribution.
