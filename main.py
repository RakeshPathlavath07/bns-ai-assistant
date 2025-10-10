# comparative_app.py (The Definitive, Final, and Polished Version)

import streamlit as st
import os
from dotenv import load_dotenv
import re
import tiktoken
import json

from ddgs.ddgs import DDGS

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

st.set_page_config(page_title="BNS AI Legal Assistant", page_icon="⚖️", layout="wide")

# --- CACHED RESOURCES ---
@st.cache_resource
def load_llm():
    api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
    if not api_key: st.error("OPENROUTER_API_KEY is not set."); st.stop()
    model_id = "mistralai/mistral-7b-instruct:free"
    return ChatOpenAI(model_name=model_id, openai_api_base="https://openrouter.ai/api/v1", openai_api_key=api_key, temperature=0.1, max_tokens=2048)

@st.cache_resource
def load_vector_store():
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        return FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load the FAISS database. Did you run the database builder? Error: {e}")
        st.stop()

# --- HELPER FUNCTIONS ---
@st.cache_data
def find_best_link(section_id):
    if not section_id or section_id == "Unknown": return None
    clean_section_id = re.match(r"(Section \d+[A-Z]?)", section_id).group(1) if re.match(r"(Section \d+[A-Z]?)", section_id) else section_id
    section_number = clean_section_id.split(' ')[-1]
    search_query = f"Bharatiya Nyaya Sanhita Section {section_number} site:indiankanoon.org OR site:devgan.in"
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(search_query, max_results=1))
        if results: return results[0].get('href')
    except Exception: return None

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('section', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)

# --- UI SETUP ---
st.title("⚖️ AI Assistant for Bharatiya Nyaya Sanhita (BNS)")
st.subheader("Your Intelligent Guide to India's New Penal Code")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Seal_of_the_Supreme_Court_of_India.svg/1200px-Seal_of_the_Supreme_Court_of_India.svg.png", width=100)
    st.header("About")
    st.info("This app combines a conversational chatbot with a powerful comparative case analysis tool.")
    st.header("Disclaimer")
    st.warning("**This is an AI-generated tool and not a substitute for professional legal advice.**")

llm = load_llm()
vector_store = load_vector_store()

# ==========================================
# 1. WORKING, STRICT CHATBOT WITH LINKS
# ==========================================
st.header("💬 Chat with the BNS Assistant")
st.info("Ask a question (e.g., 'what is section 113') or a follow-up (e.g., 'what is the punishment for it?').")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", input_key="question")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # This will render the full saved content, including the expanders
        st.markdown(message["content"], unsafe_allow_html=True)

chatbot_retriever = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(search_kwargs={"k": 7}), llm=llm)
condense_question_template = "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\n\nFollow Up Input: {question}\n\nStandalone question:"
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(condense_question_template)
answer_template = """You are an expert legal assistant. Your answers must be based *only* on the provided context from the Bharatiya Nyaya Sanhita (BNS). If the context doesn't contain the answer, state that you cannot answer based on the provided BNS sections. For every piece of information you derive from the context, you MUST cite it using the exact format `[Source: Section XX]` or `[Source: Section XX(Y)]`.
Context:
{context}
Question: {question}
Helpful Answer:"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_template)

def create_standalone_question(inputs):
    chat_history_messages = st.session_state.memory.chat_memory.messages
    return (CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()).invoke({"question": inputs["question"], "chat_history": chat_history_messages})

final_answer_chain = ({"context": chatbot_retriever | format_docs, "question": RunnablePassthrough()} | ANSWER_PROMPT | llm | StrOutputParser())

if prompt := st.chat_input("Ask about the BNS..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching BNS and thinking..."):
            standalone_question = create_standalone_question({"question": prompt})
            response = final_answer_chain.invoke(standalone_question)
            st.session_state.memory.save_context({"question": prompt}, {"output": response})
            
            # --- THE UI POLISH IS BACK ---
            # Create the full response content with expanders
            full_response_content = response
            cited_sections = sorted(list(set(re.findall(r'\[Source: (Section \d+[A-Z]?\s?(?:\(\w+\))?)\]', response))))
            if cited_sections:
                full_response_content += "\n\n**Referenced Sections:**"
                retrieved_docs_for_chat = chatbot_retriever.invoke(standalone_question)
                doc_map_chat = {doc.metadata.get('section'): doc for doc in retrieved_docs_for_chat}
                
                # We will build these expanders as part of the string to save in history
                for section_id in cited_sections:
                    base_section_id = re.match(r"(Section \d+[A-Z]?)", section_id).group(1)
                    if base_section_id in doc_map_chat:
                        expander_content = doc_map_chat[base_section_id].page_content.replace('\n', '<br>')
                        link_html = ""
                        if link := find_best_link(section_id):
                            link_html = f'<br><a href="{link}" target="_blank">Read more on external site</a>'
                        # Using HTML details tag for expanders inside markdown
                        full_response_content += f"<details><summary><b>{section_id}</b></summary>{expander_content}{link_html}</details>"
                    else:
                        full_response_content += f"<details><summary><b>{section_id}</b></summary>Full text not in top results.</details>"

            st.markdown(full_response_content, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response_content})

# ==========================================
# 2. Comparative Legal Analysis Tool
# ==========================================
st.markdown("---")
st.header("⚖️ Comparative Legal Analysis Tool")
st.info("Provide a case summary and a real judgment to compare the AI's analysis with a real-world outcome.")

if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
    st.session_state.case_summary_input = ""
    st.session_state.judgment_summary_input = ""

def trigger_analysis():
    st.session_state.run_analysis = True
    st.session_state.case_summary_input = st.session_state.case_area
    st.session_state.judgment_summary_input = st.session_state.judgment_area

with st.expander("Expand to use Comparative Analysis", expanded=True):
    st.text_area("📝 Enter Case Summary", key='case_area', value=st.session_state.case_summary_input)
    st.text_area("🧑‍⚖️ Enter Real Judgment Summary", key='judgment_area', value=st.session_state.judgment_summary_input)
    st.button("Perform Comparative Analysis", type="secondary", on_click=trigger_analysis)

if st.session_state.run_analysis:
    case_summary = st.session_state.case_summary_input
    real_judgment_summary = st.session_state.judgment_summary_input

    if not case_summary.strip() or not real_judgment_summary.strip():
        st.warning("Please provide both a Case Summary and a Real Judgment Summary.")
    else:
        with st.spinner("Running reliable analysis..."):
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            model_analysis_docs = retriever.invoke(case_summary)
            
            model_bns_interpretation = "The AI could not find sufficiently relevant BNS sections."
            model_cited_sections = []
            if model_analysis_docs:
                model_context = format_docs(model_analysis_docs)
                analysis_prompt = ChatPromptTemplate.from_template("Analyze the case based *only* on the provided BNS sections. Provide a concise interpretation and **MANDATORY:** Cite info using `[Source: Section XX]` or `[Source: Section XX(Y)]`.\n\nCase: {case_summary}\n\nContext: {context}\n\nAnalysis:")
                model_bns_interpretation = (analysis_prompt | llm | StrOutputParser()).invoke({"context": model_context, "case_summary": case_summary})
                model_cited_sections = sorted(list(set(re.findall(r'\[Source: (Section \d+[A-Z]?\s?(?:\(\w+\))?)\]', model_bns_interpretation))))

            extraction_template = """You are a data extraction assistant. Read the judgment summary and extract the BNS sections mentioned and the final outcome. Provide the output ONLY in this exact format, with each item on a new line:
SECTIONS: [Section XXX, Section YYY]
OUTCOME: [The full outcome description]

Judgment Summary: {judgment_summary}
"""
            extraction_prompt = ChatPromptTemplate.from_template(extraction_template)
            extracted_text = (extraction_prompt | llm | StrOutputParser()).invoke({"judgment_summary": real_judgment_summary})
            judge_sections = re.findall(r"Section \d+[A-Z]?", extracted_text)
            outcome_match = re.search(r"OUTCOME: \[(.*)\]", extracted_text)
            judge_outcome = outcome_match.group(1).strip() if outcome_match else "Could not determine outcome."
            
            comparison_prompt = ChatPromptTemplate.from_template("Briefly explain discrepancies between the AI's analysis and the real judgment. Consider provocation, judicial discretion, or evidence. Be neutral.\n\nAI Analysis (Cited {ai_sections}): {ai_interpretation}\n\nReal Judgment (Cited {judge_sections}): {judge_outcome}\n\nDiscrepancy Analysis:")
            discrepancy_analysis = (comparison_prompt | llm | StrOutputParser()).invoke({"ai_sections": ", ".join(model_cited_sections) or 'None', "ai_interpretation": model_bns_interpretation, "judge_sections": ", ".join(judge_sections) or 'None', "judge_outcome": judge_outcome})

            st.markdown("---")
            st.header("📊 Final Comparative Report")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🤖 AI's BNS-based Analysis")
                st.markdown("**AI Suggested Sections:**")
                doc_map = {doc.metadata.get('section'): doc for doc in model_analysis_docs}
                if model_cited_sections:
                    for section_id in model_cited_sections:
                        with st.expander(f"**{section_id}**"):
                            base_section_id = re.match(r"(Section \d+[A-Z]?)", section_id).group(1)
                            if base_section_id in doc_map:
                                st.write(doc_map[base_section_id].page_content)
                                if link := find_best_link(section_id): st.markdown(f"**Read more:** [{section_id}]({link})")
                            else:
                                if link := find_best_link(section_id): st.info(f"Content not in top results. You can [read {section_id} on external site]({link}).")
                else: st.info("The AI did not cite any specific BNS sections in its analysis.")
                with st.expander("Show full AI interpretation text", expanded=False): st.info(model_bns_interpretation)
            
            with col2:
                st.subheader("🧑‍⚖️ Real Judgment Details")
                st.markdown("**Judge Cited Sections:**")
                if judge_sections:
                    for section_id in judge_sections:
                        if link := find_best_link(section_id): st.markdown(f"🔗 **[{section_id} (Read on external site)]({link})**")
                        else: st.markdown(f"**{section_id}** (Link unavailable)")
                else: st.info("No sections found in the judgment summary.")
                st.markdown(f"**Outcome:**\n> {judge_outcome}")
            
            st.markdown("---")
            st.subheader("📝 Discrepancy Analysis")
            st.success(discrepancy_analysis)

    st.session_state.run_analysis = False