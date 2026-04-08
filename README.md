# Cyberbullying Detection using Ensemble Learning

## Project Overview
This project focuses on detecting **cyberbullying in tweets** using **Machine Learning and Ensemble Learning techniques**.

The system processes raw tweet text, cleans it, converts it into numerical features using **TF-IDF**, and applies multiple models like:
- Logistic Regression  
- Linear Support Vector Classifier (SVC)  
- Voting Classifier (Hard & Soft)  
- Stacking Classifier  

Finally, it predicts whether a tweet belongs to a specific type of cyberbullying.

---

##  Objectives
- Detect cyberbullying in text data (tweets)
- Compare multiple ML models
- Improve performance using ensemble techniques
- Provide real-time prediction using user input

---

##  Dataset
- Dataset used: `cyberbullying_tweets.csv`
- Key columns:
  - `tweet_text` → Input text
  - `cyberbullying_type` → Target label (class)

---

##  Technologies Used
- Python   
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

##  Workflow

### 1. Data Loading
- Load dataset using Pandas
- Explore structure (`head`, `info`, `shape`)

### 2. Data Preprocessing
- Convert text to lowercase
- Remove special characters
- Remove extra spaces

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
