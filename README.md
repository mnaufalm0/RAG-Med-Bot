RAG-Med-Bot
A health assistant that answers questions about diseases and symptoms — and actually tells you where it got the information from.
I built this because most LLM demos just hallucinate confidently. Wanted every answer to be traceable back to a real source, so the model can only respond based on what it retrieves from the knowledge base. No made-up information.

How it works
You ask a question → it finds the most relevant disease records from a local vector store → passes those to the LLM as context → returns a structured answer with sources cited.
The knowledge base is built from a Kaggle disease dataset — 41 diseases, 132 symptoms, treatments and precautions.

Stack

LangChain — RAG pipeline and chain orchestration
FAISS — local vector store, no external DB needed
sentence-transformers — all-MiniLM-L6-v2 for embeddings
Groq — llama-3.1-8b-instant, llama-3.3-70b-versatile, mixtral-saba-24b
Streamlit — Basic web UI

Project structure
RAG-Med-Bot/
├── app.py            # streamlit web UI
├── bot.py            # terminal chatbot
├── pipeline.py       # RAG logic
├── build_index.py    # builds the FAISS index from CSVs       
└── requirements.txt

Setup
bash:    git clone https://github.com/mnaufalm0/RAG-Med-Bot
cd RAG-Med-Bot
pip install -r requirements.txt

Create a .env file in the project root:
GROQ_API_KEY=your_key_here

Build the index
bash:  python build_index.py

# web UI
streamlit run app.py

# or terminal
python bot.py

