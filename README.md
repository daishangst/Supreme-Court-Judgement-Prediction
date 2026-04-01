# Supreme-Court-Judgement-Prediction

A machine learning and deep learning pipeline designed to forecast the winning party and legal issue areas for U.S. Supreme Court cases based on the textual facts of the case.

## ⚖️ Project Overview
* **Accomplished:** A predictive modeling framework to forecast winning parties and categorize issue areas for Supreme Court cases.
* **Measured by:** Analyzing **3,303 legal cases** (including landmark decisions like *Roe v. Wade*) and evaluating performance using weighted precision, recall, and F1-score.
* **By doing:** Cleaning complex HTML-formatted legal facts, addressing severe class imbalance through targeted oversampling techniques, and developing a comparative analysis between **Linear SVM (LinearSVC)** and a **Bidirectional LSTM neural network**.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (`LinearSVC`, oversampling techniques)
* **Deep Learning:** TensorFlow / Keras (Bidirectional LSTM, Word Embeddings)
* **Data Wrangling:** Pandas, NumPy
* **Text Processing:** Regular Expressions (Regex) for HTML stripping, Tokenization
* **Environment:** Google Colab / Jupyter Notebook

## 🧠 Model Architectures
This project compares a traditional, high-performing machine learning model against a deep sequence model:
1. **Linear Support Vector Classification (LinearSVC):** Leveraged for high-dimensional text data to establish a strong baseline.
2. **Bidirectional LSTM:** A recurrent neural network architecture featuring an **Embedding layer of 50,000 input dimensions**. Bidirectional processing helps capture the complex, contextual legal language from the case facts.

## 📂 Dataset
The model processes a structured dataset containing:
* **Facts:** The written background and legal context of the case (scraped/cleaned text).
* **Target Labels:** The ultimate decision/winning party and the categorized legal issue area.

## ⚙️ How to Run This Project
1. Clone this repository or download the `.ipynb` file to run in Google Colab.
2. Make sure you have the required libraries installed:
   ```bash
   pip install tensorflow scikit-learn pandas numpy
