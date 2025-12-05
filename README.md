# ⚖️ BNS AI Legal Assistant

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-green)
![LLM](https://img.shields.io/badge/Mistral-7B-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An AI-powered, full-stack legal assistant built for India’s Bharatiya Nyaya Sanhita (BNS 2023).**

This system goes far beyond a simple Q&A chatbot. It combines **Retrieval-Augmented Generation (RAG)**, deep legal text understanding, and a unique **Comparative Case Analysis** engine that allows users to analyze a case, compare it with a real judgment, and understand legal discrepancies.

The core mission: **Make the BNS easier to understand, explore, and apply — for students, researchers, and legal professionals.**

---

## 🚀 The Problem & Our Solution

The **Bharatiya Nyaya Sanhita (BNS)** is a dense legal document. Understanding it traditionally requires manual reading, section-by-section searching, cross-referencing, and years of interpretation experience. For students and interns, this process is slow and overwhelming.

**This project solves that problem.**

*   **Problem:** Slow, manual, and expertise-heavy legal research.
*   **Solution:** A resilient AI assistant that:
    *   Answers questions strictly using BNS context.
    *   Retrieves the most relevant legal sections instantly.
    *   Handles follow-up questions with **conversational memory**.
    *   **Compares AI interpretation with real judgments** to highlight discrepancies (e.g., judicial discretion, evidence gaps).

---

## 🛠️ Key Features

### 1. 💬 Conversational BNS Chatbot
A memory-enabled assistant that understands natural language.
*   **Contextual Understanding:** Handles follow-up queries and converts them into standalone legal questions.
*   **RAG Technology:** Retrieves relevant sections from a **FAISS vector database**.
*   **Strict Citations:** Generates answers with specific source references (e.g., `[Source: Section XX]`).
*   **Expandable Text:** Displays full legal text and links to external references (**IndianKanoon** & **Devgan**).

### 2. ⚖️ Comparative Case Analysis Engine
A unique feature that bridges the gap between theory and practice.
1.  **AI Analysis:** The system performs its own BNS-based analysis using semantic retrieval.
2.  **Judgment Extraction:** Extracts structured data (`SECTIONS`, `OUTCOME`) from real judgment text.
3.  **Discrepancy Report:** Compares the AI's logic with the Judge's ruling, explaining differences due to **judicial discretion**, **mitigating circumstances**, or **intent**.

### 3. 🧠 Stateful Conversational Memory
The assistant remembers the context of your conversation.
*   *User:* "What is Section 113?"
*   *User:* "What is the punishment for it?"
*   *Result:* The AI knows "it" refers to Section 113.

### 4. 🔗 External Legal Link Integration
For every cited section, the system uses **DuckDuckGo Search** to fetch authoritative reference links from **IndianKanoon** and **Devgan** for further reading.

---

## 🏗️ Architecture & Tech Stack

This project is built for modularity, accuracy, and scalability.

| Component | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Orchestration** | LangChain |
| **LLM** | Mistral-7B (via OpenRouter API) |
| **Vector Store** | FAISS |
| **Embeddings** | SentenceTransformers (BAAI/bge-large-en-v1.5) |
| **Search** | DuckDuckGo Search (DDGS) |
| **Data Pipeline** | Custom Python PDF Parser |

---

## ⚙️ Installation & Usage

Follow these steps to run the project locally.

### 1. Prerequisites
*   Python 3.9+
*   Virtual environment tool (`venv` or `conda`)

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/bns-ai-assistant.git
cd bns-ai-assistant
```
### 3. Set Up the Environment
Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```

### 4.Configure API Keys
This project uses OpenRouter to access Mistral-7B.
Create a secrets file for Streamlit:
```bash
mkdir .streamlit
touch .streamlit/secrets.toml  # Or create manually in Windows
```

Add your API key to .streamlit/secrets.toml:
```bash
OPENROUTER_API_KEY = "your_openrouter_key_here"
```

### 5.Run the Application
```bash
streamlit run comparative_app.py
```
The app will launch automatically in your default browser.

---

##  🗺️ Future Roadmap
This project is designed to grow into a complete legal intelligence system.

*   [ ] ** IPC ↔ BNS Cross-Referencing: Mapping old laws to the new Sanhita.

*   [ ] ** Case Law Database: Direct integration with a larger database of judgments.

*   [ ] ** Multilingual Support: Hindi and regional language query support.

*   [ ] ** Session Management: Save and share analysis reports.

*   [ ] ** Advanced Re-ranking: Improve retrieval accuracy using cross-encoders.

---

##  📜 License
This project is licensed under the MIT License. 

