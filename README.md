# 🎯 VoiceFlow AI Analytics

Built with **Streamlit, LangChain, Llama 3.3 (Groq), and RAG**, the **Customer Feedback Intelligence Platform** uses AI.

This app looks at customer reviews, does sentiment analysis, gives you insights, has an AI chat assistant for deep analytics, and makes friendly, professional, or apologetic emails to customers based on reviews.
---

- ✅ Automatic Sentiment Analysis
- ✅ Chat with an AI Analyst (RAG Powered)
- ✅ Dashboard for the Voice of the Customer
- ✅ Finding Categories
- ✅ Email Response Maker
- ✅ Data Explorer and CSV Export

---

## 🧠 Tech Stack
- Streamlit
- LangChain
- Groq (Llama-3.3-70B)
- ChromaDB
- FlashRank
- Embeddings from HuggingFace
- TextBlob- Plotly

---

## 📂 How the Project Is Set Up

voiceflow_analysis/ │ ├── app.py
├──.env
├── requirements.txt │ ├── data/ │ └── reviews.csv
│ └── README.md

⚠️ The app only works when you upload CSV files by hand using the sidebar file uploader. To see sample test data, go to `data/reviews.csv`

---

⚙️ Setting Up the Environment

Make a file called `.env` in the root directory:

GROQ_API_KEY=your_api_key_here
---

## 📦 Setting Up

Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git cd YOUR_REPO

Make a virtual environment:

python -m venv venv

Turn on the environment:

Mac/Linux: source venv/bin/activate

Windows: venv\Scripts\activate

Install the things you need:

pip install -r requirements.txt
---

## ▶️ Run on your own computer

run app.py in streamlit
---

## 📊 Format of the Dataset

Put the dataset here:

data/reviews.csv

Minimum requirement:

Your CSV file needs to have a column that looks like this:

review text comment feedback  

The app does this on its own:
- Finds out how people feel
- Makes a rating
- Gives a category

---

## AI Pipeline

1. CSV to Documents  
2. Embeddings from HuggingFace  
3. Kept in ChromaDB  
4. FlashRank reordering  
5. Llama-3.3 gives us new ideas  
---

What are the most common complaints from customers?  
What groups have the most problems?  
Which changes should we make first?
---

## 🧾 Notes on Deployment

Check to see if your demo dataset is at:

data/reviews.csv

And your app loads it like this:

elif os.path.exists("data/reviews.csv"):

---

Priyanshu Sinha  
---

Thanks for referring the project.
