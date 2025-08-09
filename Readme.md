# ADGM-Compliant Corporate Agent: An Intelligent Legal Assistant

This project is an intelligent legal assistant designed to automate and streamline the compliance review process for business incorporation within the Abu Dhabi Global Market (ADGM) jurisdiction. The agent provides a robust, multi-layered analysis of legal documents to ensure they meet the stringent requirements of the ADGM.

## Project Overview

The Corporate Agent is engineered to serve as a first line of defense against common compliance errors. It ingests user-uploaded `.docx` files and performs a comprehensive analysis covering document completeness, regulatory adherence, and legal best practices. By leveraging a hybrid approach of a deterministic rule engine and a sophisticated RAG-powered citation system, the agent provides precise, actionable feedback, significantly reducing the time and effort required for manual legal reviews.

---

## Key Capabilities

* **Automated Checklist Verification:** The system intelligently identifies the user's intended legal process (e.g., Company Incorporation) by analyzing the content of uploaded documents. It then cross-references the submission against official ADGM checklists to instantly flag any missing mandatory documents.

* **Multi-Point Red Flag Detection:** A sophisticated rule-based engine scans documents for a wide range of potential compliance issues, including:
    * **Jurisdictional Errors:** Pinpoints incorrect jurisdiction references (e.g., UAE Federal Courts instead of ADGM).
    * **Missing Clauses:** Verifies the presence of all mandatory clauses as stipulated by ADGM templates and regulations.
    * **Formatting Deficiencies:** Detects the absence of critical sections, such as signatory blocks.

* **RAG-Powered Inline Commenting:** For every identified issue, the agent generates a reviewed `.docx` file with an appendix detailing each finding, its severity, a suggested fix, and a relevant legal reference retrieved directly from the ADGM knowledge base.

* **Structured Reporting:** Produces a structured `JSON` report summarizing the complete analysis, including details on missing documents and all issues found, suitable for logging or downstream processing.

---

## RAG and Technical Architecture

The core of this agent's intelligence is its **Retrieval-Augmented Generation (RAG)** architecture, which ensures all compliance checks are grounded in actual ADGM legal documents.

* **Knowledge Base:** The system ingests and indexes official ADGM documents (like `Data Sources.pdf` and regulatory checklists) into a local vector store using `FAISS`.
* **Embeddings:** Legal texts are converted into vector embeddings using the `all-MiniLM-L6-v2` model from `sentence-transformers`.
* **Retrieval:** When a compliance issue is detected, the system formulates a semantic query to the vector store to retrieve the most relevant legal snippet. This allows for dynamic, context-aware legal citations rather than static, hardcoded rules.
* **LLM-Ready:** The architecture is designed for extensibility and includes a module (`rag_system.py`) to integrate a powerful local LLM like **Llama 3.1 (8b)** for more advanced semantic analysis, such as detecting ambiguous language.

* **Tech Stack:**
    * **Backend:** Python
    * **Web Framework:** Streamlit
    * **Document Processing:** `python-docx`, `PyMuPDF`
    * **Vector Embeddings & Search:** `sentence-transformers`, `faiss-cpu`

---

## Getting Started

### Prerequisites

* Python 3.10+
* Git

### Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and Activate Virtual Environment (Windows):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run the Application

#### **Method 1: Quick Launch (Windows)**

Simply double-click the **`start_app.bat`** file. This will handle the environment activation and launch the application automatically.

#### **Method 2: Manual Launch**

Ensure your virtual environment is activated, then run the following command from the project's root directory:
```bash
streamlit run app.py
