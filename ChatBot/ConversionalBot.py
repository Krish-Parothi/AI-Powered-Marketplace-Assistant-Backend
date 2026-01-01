# app.py
# Single-file, database-grounded, multi-turn explanation chatbot using LangChain + Groq

import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI(title="Artisan Explanation Chatbot")

# -------------------------------------------------
# VECTOR DB
# -------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vectorstore = Chroma(
    persist_directory="./artisan_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -------------------------------------------------
# LLM (GROQ)
# -------------------------------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)

# -------------------------------------------------
# PROMPT (EXPLANATION ENGINE)
# -------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an explanation engine.
Use only retrieved facts.
No persuasion. No marketing.

Explain using:
- time investment
- material quality
- cultural value
- market comparison

If data is missing, explicitly say so.
"""
    ),
    (
        "human",
        "Buyer question: {question}\n\nContext:\n{context}"
    )
])

# -------------------------------------------------
# MEMORY (STRUCTURED)
# -------------------------------------------------
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=600
)

# -------------------------------------------------
# QA CHAIN
# -------------------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt,
        "output_parser": StrOutputParser()
    },
    return_source_documents=False
)

# -------------------------------------------------
# REQUEST SCHEMA
# -------------------------------------------------
class ChatRequest(BaseModel):
    question: str

# -------------------------------------------------
# ROUTES
# -------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    result = qa_chain.invoke({
        "query": req.question
    })

    memory.save_context(
        {"input": req.question},
        {"output": result["result"]}
    )

    return {
        "answer": result["result"]
    }

@app.post("/reset-memory")
def reset_memory():
    memory.clear()
    return {"status": "memory cleared"}

# -------------------------------------------------
# RUN
# -------------------------------------------------
# uvicorn app:app --reload
