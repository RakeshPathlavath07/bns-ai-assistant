# app.py

import streamlit as st
import os
from dotenv import load_dotenv
import re
import tiktoken

from duckduckgo_search import DDGS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

load_dotenv()

# --- POLISH: Set the page configuration for the entire app ---
st.set_page_config(
    page_title="⚖️ AI Legal Assistant - Bharatiya Nyaya Sanhita (BNS)",
    page_icon="⚖️",
    layout="wide"
)

# --- CONFIG ---
CHROMA_DIR = "database_chroma_db"
COLLECTION_NAME = "bns_sections_definitive"
BI_ENCODER_MODEL = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- CACHED RESOURCES ---
@st.cache_resource
def load_llm():
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("OPENROUTER_API_KEY is not set.")
        st.stop()
    return ChatOpenAI(model_name="openai/gpt-3.5-turbo", openai_api_base="https://openrouter.ai/api/v1", openai_api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0, max_tokens=1024)

@st.cache_resource
def load_vector_store():
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name=BI_ENCODER_MODEL)
        return Chroma(persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME, embedding_function=embedding_function)
    except Exception as e:
        st.error(f"Failed to load the vector database. Error: {e}")
        st.stop()

@st.cache_resource
def load_reranker():
    return CrossEncoder(CROSS_ENCODER_MODEL)

@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

@st.cache_data
def find_best_link(section_id):
    if not section_id or section_id == "Unknown": return None
    section_number = section_id.split(' ')[-1]
    search_query = f"Bharatiya Nyaya Sanhita Section {section_number} site:indiankanoon.org OR site:devgan.in"
    try:
        with DDGS() as searcher:
            search_results = list(searcher.text(search_query, max_results=1))
        return search_results[0]["href"] if search_results else None
    except Exception: return None

# --- RETRIEVAL & RE-RANKING LOGIC ---
def retrieve_and_rerank(question: str, vector_store, reranker, llm, num_candidates=15, top_n=3):
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(search_kwargs={"k": num_candidates}), llm=llm)
    candidate_docs = multi_query_retriever.invoke(question)
    
    if not candidate_docs: return []
    
    unique_docs = list({doc.metadata.get('section', doc.page_content): doc for doc in candidate_docs}.values())
    pairs = [[question, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(pairs)
    
    scored_docs = zip(scores, unique_docs)
    sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    
    return [doc for score, doc in sorted_docs][:top_n]

# --- MAIN APP LOGIC ---
st.title("⚖️ AI Assistant for Bharatiya Nyaya Sanhita (BNS)")
st.subheader("Your Intelligent Guide to India's New Penal Code")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Seal_of_the_Supreme_Court_of_India.svg/1200px-Seal_of_the_Supreme_Court_of_India.svg.png", width=100)
    st.header("About")
    st.info("This app is an AI-powered search and summarization tool for the Bharatiya Nyaya Sanhita, 2023. It uses a Retrieve-and-Re-rank strategy for high accuracy.")
    st.header("Disclaimer")
    st.warning(
        "**This is an AI-generated tool and not a substitute for professional legal advice.** "
        "The information provided is for informational purposes only. Always consult with a qualified legal professional for any legal concerns."
    )

llm = load_llm()
vector_store = load_vector_store()
reranker = load_reranker()
tokenizer = get_tokenizer()

def format_context_with_token_limit(docs: list[Document], max_tokens=12000):
    formatted_context = ""
    total_tokens = 0
    for doc in docs:
        doc_string = f"--- \nSource: {doc.metadata.get('section', 'Unknown')}\nContent: {doc.page_content}\n ---\n\n"
        
        # =====================================================================
        #  THE DEFINITIVE FIX: Use len() to get the number of tokens.
        # =====================================================================
        doc_tokens = len(tokenizer.encode(doc_string))
        
        if total_tokens + doc_tokens > max_tokens:
            break
        formatted_context += doc_string
        total_tokens += doc_tokens
    return formatted_context

template = """
You are an expert legal assistant for India's Bharatiya Nyaya Sanhita (BNS).
Your task is to provide a clear, precise, and comprehensive answer based *only* on the legal context provided below.
**Instructions:**
1. Synthesize the information from all provided sources into a coherent answer.
2. You **MUST** cite the specific section number for each piece of information you use. Use the format `[Source: Section XX]`.
3. If the context contains different punishments for different circumstances, explain these nuances clearly.
4. Structure your answer logically. Start with a direct answer, then provide the details.
5. If the provided context is empty or does not contain a relevant answer, you MUST state: "Based on the provided legal context, I cannot answer this question."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = (prompt | llm | StrOutputParser())

question = st.text_input("Ask your legal question about the BNS:", placeholder="e.g., What is the punishment for murder?")

if st.button("Search", type="primary") and question.strip():
    with st.spinner("Step 1: Retrieving candidate sections..."):
        final_docs = retrieve_and_rerank(question, vector_store, reranker, llm)

    if not final_docs:
        st.warning("No relevant sections were found in the Bharatiya Nyaya Sanhita for your query.")
    else:
        with st.spinner("Step 2: Building context and generating final answer..."):
            context = format_context_with_token_limit(final_docs)
            
            if not context.strip():
                 st.error("Could not build a valid context from the retrieved documents.")
            else:
                answer = chain.invoke({"context": context, "question": question})
                st.markdown("### 📝 Synthesized Summary with Citations")
                st.success(answer)

                st.markdown("---")
                st.header("✅ Top Retrieved Sections for Verification")
                for doc in final_docs:
                    section_id = doc.metadata.get('section', 'Unknown')
                    with st.expander(f"**{section_id}** (Top Match)"):
                        st.write(doc.page_content)
                        with st.spinner(f"Searching for a web link for {section_id}..."):
                            link = find_best_link(section_id)
                        if link:
                            st.markdown(f"**Read more online:** [{section_id} on external site]({link})")