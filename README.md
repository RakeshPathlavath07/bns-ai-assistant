⚖️ BNS AI Legal Assistant
An AI-powered assistant for India’s Bharatiya Nyaya Sanhita (BNS), with a unique Comparative Case Analysis Engine
💡 Overview

This project is an advanced, full-stack AI legal assistant built specifically for the Bharatiya Nyaya Sanhita (BNS 2023). It provides two major capabilities:

1️⃣ Conversational Legal Chatbot

Ask any question about the BNS in simple English.
The system retrieves relevant legal sections from a FAISS vector database created from the official BNS PDF and generates accurate, grounded answers with proper citations.

2️⃣ Comparative Case Analysis Tool (Unique Feature)

You provide:

a case summary

a real judgment summary

The AI:
✔ Retrieves relevant BNS sections
✔ Performs its own interpretation
✔ Extracts judge-cited sections & final outcome
✔ Compares both
✔ Generates a Discrepancy Analysis explaining the differences

Perfect for law students, researchers, and legal analysis training.

🚀 Problem & Solution
The Problem

Legal research is slow and requires expert knowledge. Understanding how the BNS applies to real-life cases is even more challenging.

The Solution

A powerful AI tool that:

interprets BNS sections instantly

provides grounded legal answers

compares AI-based legal reasoning with real-world judgments

explains judicial discretion and differences

🧠 Key Features
✅ 1. Conversational AI Chatbot

Natural language input

Understands follow-up questions using memory

Converts follow-up queries into meaningful standalone questions

Retrieves relevant sections using FAISS

Answers strictly based on the BNS legal context

Uses citations like:
[Source: Section XX]

Provides expandable legal content + external source links

✅ 2. Comparative Case Analysis Engine

A custom 3-step AI workflow:

Step 1 — AI’s BNS-Based Interpretation

Takes case summary

Retrieves relevant BNS sections

Produces structured legal analysis with citations

Step 2 — Judgment Data Extraction

From the real judgment summary, the AI extracts:

SECTIONS: [...]
OUTCOME: [...]

Step 3 — Final Discrepancy Analysis

Compares:

AI interpretation

Real judge interpretation

Explains differences due to:

judicial discretion

evidence

provocation

interpretation variations

✅ 3. Intelligent External Legal Links

For each cited section, the system fetches the most relevant link from:

IndianKanoon.org

Devgan.in

✅ 4. Clean Streamlit UI

Chat-style conversation

Memory-enabled

Expandable legal references

Two-column comparative report

Warnings when information is outside BNS context

🏗️ Architecture & Tech Stack
Frontend

Streamlit

AI Orchestration

LangChain

ConversationBufferMemory

Runnable workflows

MultiQueryRetriever

LLM

Mistral 7B Instruct via OpenRouter API

Vector Search

FAISS vector database

Embeddings: BAAI/bge-large-en-v1.5

Data Processing

Custom section extraction using Regex

PDF parsing with PyMuPDF (from builder script)

DDGS for external link search

📁 Project Structure
│── comparative_app.py               # Main application
│── build_database_faiss.py          # Builds FAISS index from BNS PDF
│── Bharatiya_Nyaya_Sanhita_2023.pdf
│── faiss_index/                     # Vector DB storage
│── requirements.txt
│── runtime.txt
│── .gitattributes
│── .gitignore

⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/yourusername/bns-ai-assistant
cd bns-ai-assistant

2. Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Add API Key

Create:

.streamlit/secrets.toml


Add:

OPENROUTER_API_KEY = "your_key_here"

5. Run the App
streamlit run comparative_app.py

🎯 Usage
🟦 Chatbot Mode

Ask questions like:

"What is Section 113?"

"What is the punishment for it?"

Features:

citation-linked answers

expandable sections

links to IndianKanoon/Devgan

memory-based follow-up

🟩 Comparative Case Analysis Mode

Input:

case summary

real judgment summary

Outputs:

AI interpretation

cited sections

judge-cited sections

final judgment outcome

discrepancy analysis

expandable legal references

🗺️ Future Enhancements

Add case law databases

Multi-language support (Hindi first)

Persistent conversation history

Cross-encoder re-ranking

Mobile UI optimization

📜 License

This project is licensed under the MIT License.
