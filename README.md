# 🕵️ Financial Analytics & Compliance Agent

> An autonomous Tier-1 AI financial analyst built with LangGraph that intelligently routes, self-corrects, and answers complex enterprise queries across structured SQLite databases and unstructured RAG compliance documents.

## 🎯 What It Does

This agent autonomously bridges the gap between raw transactional data and regulatory policy. It takes a natural language question, decides whether to query a SQL database (for numbers/trends) or a Vector Store (for policies/rules), executes the necessary tools, and synthesizes a grounded, accurate final answer.

**Key Capabilities:**
*   **Intelligent Routing:** Uses an LLM Router to classify intent as `SQL`, `RAG`, or `Both`.
*   **Self-Correction:** Detects SQL syntax errors or irrelevant RAG retrievals and autonomously retries with refined prompts.
*   **Multi-Source Synthesis:** Combines structured transaction data with unstructured anti-fraud policies to answer hybrid questions.
*   **Automated Evaluation:** Includes a custom LLM-as-a-Judge harness to score agent performance on Faithfulness and Correctness.

## 🏗️ Architecture

The agent is built using a stateful LangGraph workflow:

1.  **Router Node:** Analyzes the user's question and decides which data source to use.
2.  **Tool Nodes:** 
    *   `SQL Tool`: Generates and executes read-only SQLite queries against normalized bank transaction tables.
    *   `RAG Tool`: Retrieves relevant chunks from anti-fraud policy PDFs stored in ChromaDB.
3.  **Generator Node:** Synthesizes the tool outputs into a coherent, markdown-formatted response.
4.  **Evaluation Harness:** Automatically scores the agent's routing accuracy, faithfulness to context, and overall correctness.

## 🛠️ Tech Stack

*   **Agent Framework:** [LangGraph](https://langchain-ai.github.io/langgraph/) (Stateful, self-correcting workflows)
*   **LLM:** Qwen 2.5 7B (via [Ollama](https://ollama.com/), fully local & open-source)
*   **Embeddings:** `nomic-embed-text` (via Ollama)
*   **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Local persistence)
*   **Relational Database:** [SQLite](https://www.sqlite.org/) (Normalized schema: Customers, Accounts, Transactions)
*   **Document Processing:** PyPDF, LangChain Text Splitters
*   **Evaluation:** Custom LLM-as-a-Judge harness (Faithfulness + Correctness scoring)

## 📊 Evaluation Results

The agent was evaluated against a golden dataset of complex financial queries:

| Metric | Score | Details |
|--------|-------|---------|
| **Routing Accuracy** | **100%** (3/3) | Perfectly classified SQL, RAG, and Hybrid intents |
| **Avg. Faithfulness** | **3.0/5.0** | Grounded in context; occasional minor hallucinations caught by eval harness |
| **Avg. Correctness** | **4.67/5.0** | High-quality answers that directly address user prompts |

*(Note: The evaluation harness successfully identified areas for prompt refinement, demonstrating its value in production monitoring.)*

## 📂 Project Structure

```text
financial-analytics-and-compliance-agent/
├── agent/                # Core LangGraph agent logic
│   ├── generator.py      # Final answer synthesis node
│   ├── graph.py          # LangGraph stateful workflow definition
│   ├── router.py         # Intent classification node
│   ├── state.py          # TypedDict state schema
│   └── tools.py          # SQL and RAG execution utilities
├── data/                 # Local directory for raw transaction CSVs and PDFs
│   ├── policies/         # Anti-fraud PDF documents
│   └── README.md         # Instructions for data sources
├── evaluation/           # Automated testing framework
│   └── eval_harness.py   # LLM-as-a-Judge evaluation pipeline
├── ingestion/            # Data loading and normalization scripts
│   ├── sqlite_setup.py   # CSV to relational DB conversion
│   ├── vector_store.py   # PDF chunking and embedding
│   └── data_loader.py    # Master ingestion orchestrator
├── .env.example          # Template configuration for environment secrets
├── .gitignore            # Excludes local data, caches, and venv files
├── LICENSE               # MIT Open Source License
├── config.py             # Centralized configuration management
├── models.py             # Provider-agnostic LLM/Embedding initialization
├── requirements.txt      # Project library dependency manifest
└── README.md             # Project documentation
```

## 🚀 Quick Start

### Prerequisites
*   Python 3.10+
*   [Ollama](https://ollama.com/) installed and running

### 1. Clone and Install
```bash
git clone https://github.com/kunaltibe/financial-analytics-and-compliance-agent.git
cd financial-analytics-and-compliance-agent
pip install -r requirements.txt
```

### 2. Pull Local Foundation Models
Ensure Ollama is active in the background, then pull the execution models:
```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 3. Initialize Environment Secrets
Generate your environment configuration file from the template:
```bash
cp .env.example .env
```
*Open the newly created `.env` file and append any preferred cloud provider keys if testing external model routing parameters.*

### 4. Execute Data Ingestion Pipelines
Place your source transaction records and target policy documents into the `data/` directory, then process them:
```bash
python -m ingestion.data_loader
```

### 5. Launch the Triage Agent
Interact with the autonomous agent using standard queries:
```bash
python -m agent.graph
```

### 6. Run Automated Testing & Evaluation
Execute the custom testing framework to compile system capability scores:
```bash
python -m evaluation.eval_harness
```

---

## 🎓 Production-Grade Engineering Patterns

*   **Provider-Agnostic Design:** The `models.py` layer allows swapping between local Ollama, OpenAI, or Azure with a single config change.
*   **Self-Correction Loops:** The agent detects tool failures (e.g., bad SQL) and retries, mimicking human analyst behavior.
*   **LLM-as-a-Judge:** Building a custom evaluation harness revealed subtle hallucinations that manual testing missed, highlighting the importance of automated MLOps pipelines.
*   **Data Governance:** Sensitive raw data (CSVs, PDFs) is kept out of Git, with clear instructions for local ingestion, mirroring enterprise security standards.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for comprehensive terms.