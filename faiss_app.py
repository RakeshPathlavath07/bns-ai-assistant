# app.py

import streamlit as st
import os
from dotenv import load_dotenv
import re
import tiktoken

# --- THE DEFINITIVE FIX: Switch back to the ddgs import ---
from ddgs.ddgs import DDGS

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

load_dotenv()

st.set_page_config(page_title="BNS AI Legal Assistant", page_icon="⚖️", layout="wide")

# --- CONFIG ---
FAISS_INDEX_DIR = "faiss_index"
BI_ENCODER_MODEL = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- CACHED RESOURCES ---
@st.cache_resource
def load_llm():
    api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
    if not api_key: st.error("OPENROUTER_API_KEY is not set."); st.stop()
    return ChatOpenAI(model_name="meta-llama/llama-4-maverick:free", openai_api_base="https://openrouter.ai/api/v1", openai_api_key=api_key, temperature=0.3, max_tokens=1024)

@st.cache_resource
def load_vector_store():
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name=BI_ENCODER_MODEL)
        return FAISS.load_local(FAISS_INDEX_DIR, embedding_function, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load the FAISS database. Did you run the database builder? Error: {e}")
        st.stop()

@st.cache_resource
def load_reranker():
    return CrossEncoder(CROSS_ENCODER_MODEL)

@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

# --- DEFINITIVE, ROBUST WEB SEARCH FUNCTION ---
@st.cache_data
def find_best_link(section_id):
    if not section_id or section_id == "Unknown": return None
    section_number = section_id.split(' ')[-1]
    search_query = f"Bharatiya Nyaya Sanhita Section {section_number} site:indiankanoon.org OR site:devgan.in"
    print(f"Executing web search for: {search_query}")
    try:
        # --- THE DEFINITIVE FIX: Use the 'with' statement for the ddgs library ---
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(search_query, max_results=1))
        
        print(f"Search results for {section_id}: {results}")
        
        if results:
            return results[0].get('href')
        return None
    except Exception as e:
        print(f"Web search for {section_id} failed with an error: {e}")
        return None

# (The rest of your code remains exactly the same)
# ... (retrieve_and_rerank, UI, main chain, etc.)
def retrieve_and_rerank(question: str, vector_store, reranker, llm, num_candidates=15, top_n=3):
    query_with_instruction = "Represent the law document for retrieval: " + question
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(search_kwargs={"k": num_candidates}), llm=llm)
    candidate_docs = multi_query_retriever.invoke(query_with_instruction)
    if not candidate_docs: return []
    unique_docs = list({doc.metadata.get('section', doc.page_content): doc for doc in candidate_docs}.values())
    pairs = [[question, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(pairs)
    scored_docs = zip(scores, unique_docs)
    sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    return [doc for score, doc in sorted_docs][:top_n]

st.title("⚖️ AI Assistant for Bharatiya Nyaya Sanhita (BNS)")
st.subheader("Your Intelligent Guide to India's New Penal Code")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Seal_of_the_Supreme_Court_of_India.svg/1200px-Seal_of_the_Supreme_Court_of_India.svg.png", width=100)
    st.header("About")
    st.info("This app uses a Retrieve-and-Re-rank strategy with a FAISS database for maximum reliability.")
    st.header("Disclaimer")
    st.warning("**This is an AI-generated tool and not a substitute for professional legal advice.**")

llm = load_llm()
vector_store = load_vector_store()
reranker = load_reranker()
tokenizer = get_tokenizer()

def format_context_with_token_limit(docs: list[Document], max_tokens=12000):
    formatted_context = ""
    total_tokens = 0
    for doc in docs:
        doc_string = f"--- \nSource: {doc.metadata.get('section', 'Unknown')}\nContent: {doc.page_content}\n ---\n\n"
        doc_tokens = len(tokenizer.encode(doc_string))
        if total_tokens + doc_tokens > max_tokens: break
        formatted_context += doc_string
        total_tokens += doc_tokens
    return formatted_context

template = "You are an expert legal assistant... Answer based *only* on the context... **Cite your sources** `[Source: Section XX]`... CONTEXT: {context} QUESTION: {question} ANSWER:"
prompt = ChatPromptTemplate.from_template(template)
chain = (prompt | llm | StrOutputParser())

question = st.text_input("Ask your legal question about the BNS:", placeholder="e.g., What is the punishment for murder?")

if st.button("Search", type="primary") and question.strip():
    with st.spinner("Step 1: Retrieving and re-ranking sections..."):
        final_docs = retrieve_and_rerank(question, vector_store, reranker, llm)

    if not final_docs:
        st.warning("No relevant sections were found for your query.")
    else:
        with st.spinner("Step 2: Building context and generating final answer..."):
            context = format_context_with_token_limit(final_docs)
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
                    else:
                        st.info("No reliable web link could be found for this section.")