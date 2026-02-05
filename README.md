# 🎯 VoiceFlow AI Analytics

This Customer Feedback Intelligence Platform leverages AI, constructed with Streamlit, LangChain, Llama 3.3 (Groq), and RAG.

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
- TextBlob
- Plotly


---
## 📐 Building-Architecture

The system works in a Retrieval-Augmented Generation (RAG) way:

1. **Input Data**- The user uploads a CSV file containing customer reviews.

2. **Preprocessing**- it has the ability to figure out how people feel.
   - Automatic grouping (Delivery, Pricing, Technical Issues, etc.).

3. **Putting in and saving**- HuggingFace's "all-MiniLM-L6-v2" embeddings.
   - Kept in the Chroma Vector Database.

4. **Retrieval Pipeline**- Using the Lang Chain retriever for top-level semantic retrieval.
   - FlashRank reranker makes context more relevant.

5. **LLM Response Generation**- Groq Llama-3.3-70B makes:
     - What customers think- Answers from AI analytics
     - Replies to professional emails (friendly, professional, or apologetic).


---
- Model for embedding: `all-MiniLM-L6-v2`
- Chroma: Vector DB
- Retriever Top-K: 15
- Reranker: "ms-marco-MiniLM-L-12-v2"
- LLM: "llama-3.3-70b-versatile"
- Temp: 0.3


---

## 🗂️ Data Schema

The system needs a CSV file with comments from customers.

### Required Column (found automatically): `review_text`- "read"
- "text"
- "say"
- "response"
- "sentiment" can be good, bad, or neutral (thanks to TextBlob)
- "rating" is a number from 1 to 5 that comes from the sentiment score.
- The term "category" could refer to several areas, such as delivery, product quality, technical problems, customer service, and pricing, among others.---
---
## 🧱 Indexing

- Each row of the dataset makes up a LangChain "Document."
- The **HuggingFace `all-MiniLM-L6-v2`** library creates embeddings.
- The written materials are stored in a collection called a "Chroma Vector Database."

1. User query → semantic search ("Top-K = 15")
2. FlashRank reranker makes things more useful
3. The best reviews that have been changed are sent to **Llama-3.3-70B** to make the final response.
---

## 📂 How the Project Is Set Up

voiceflow_analysis/
│
├── app.py
├── .env
├── requirements.txt
├── data/
│   └── reviews.csv
└── README.md


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

streamlit run app.py

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
## 💬 Example Queries

What are the most common complaints from customers?  
What groups have the most problems?  
Which changes should we make first?
---
## 📝 Remarks

- Instead of OpenAI APIs, the project uses **Groq Llama-3.3** for quick inference to cut down on lag and cost.
- TextBlob is used to tag sentiment locally so that no extra API calls are needed.
- FlashRank reranking is used to make retrieval more accurate than just looking at raw vector similarity.
- The dataset can be uploaded by hand through the sidebar to make deployment easier and save space.
- There were limits on free-tier hosting; caching is used to cut down on the number of times the same embedding is generated.
- The vector database is rebuilt on the fly from the uploaded CSV to keep the deployment stateless.

## 🧾 Notes on Deployment

Check to see if your demo dataset is at:

data/reviews.csv

---

Priyanshu Sinha
📄 Resume: https://drive.google.com/file/d/1EZBAGbStVipudz4ZSpxZMK0v6qDhUHXf/view?usp=sharing
---

Thanks for referring the project.
