# 🚨 Disaster Tweet Classifier

A deep learning-based web app that classifies whether a tweet refers to a real disaster or not. This project leverages a fine-tuned BERT model (`TFBertForSequenceClassification`) with `BertTokenizer`, and provides an interactive UI using Streamlit.


## 🧠 Model Details

- **Tokenizer**: `BertTokenizer` (pretrained)
- **Model**: `TFBertForSequenceClassification` (fine-tuned)
- **Framework**: TensorFlow
- **Output**: Binary classification — `0 = Not Disaster`, `1 = Disaster`

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Hugging Face Transformers  
- TensorFlow  
- Pandas / NumPy  

---

## 🚀 How to Run the Project

### 1. Clone the Repository
git clone [https://github.com/your-username/disaster-tweet-classifier](https://github.com/Narsireddy-Konapalli/Disaster-Tweet-Classifier).git

cd disaster-tweet-classifier

### 2. Install Dependencies
pip install -r requirements.txt


***3.Run the Streamlit App***

streamlit run app.py


