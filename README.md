# Agentic AI PDF RAG Chatbot (Groq + Local HuggingFace Embeddings)

A strict, production-ready Retrieval-Augmented Generation (RAG) chatbot built using **LangGraph** for workflow orchestration, **Pinecone** for vector search, **Groq API** (`llama-3.3-70b-versatile`) for ultra-fast inference, and a **Streamlit** user interface.

Strict grounding is enforced: if the retrieved context does not contain the answer, the LLM will fall back and refuse to hallucinate.

---

## Features

- **Ingestion Pipeline:** Reads the Agentic AI PDF, chunks it semantically based on sentence embedding differences, generates local embeddings, and uploads vectors to Pinecone.
- **LangGraph Orchestration:** A state graph controls the flow of context retrieval, similarity confidence threshold routing, and response generation.
- **Strict Grounding:** The assistant will refuse to answer using outside knowledge if the context is weak (similarity score < 0.55).
- **Dual Interfaces:** Contains both a **FastAPI REST API** backend and a native **Streamlit Chat UI** frontend.
- **Full Traceability:** The response returns:
  - The final answer
  - The retrieved context chunks (text, source path, page number, and confidence score)
  - The overall confidence score

---

## Setup & Running

### 1) Create Environment & Install Dependencies

```bash
# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2) Configuration (.env)

Create a `.env` file in the root directory (based on `.env.example`) and fill in your credentials:

```env
GROQ_API_KEY="your_groq_api_key"
PINECONE_API_KEY="your_pinecone_api_key"
HF_HUB_OFFLINE=1
```
*(Note: `HF_HUB_OFFLINE=1` forces HuggingFace to use the local model cache once downloaded, bypassing Hub checking and dramatically speeding up startup times.)*

### 3) Ingest the PDF

Place the PDF file at `data/ebook-agentic-ai.pdf` and run the ingestion script:

```bash
python -m rag.ingest
```

### 4) Running the Application

You can interact with the chatbot in two ways:

#### A) Streamlit Frontend UI (Recommended)
Launch the interactive chat interface directly:
```bash
streamlit run streamlit_app.py
```

#### B) FastAPI Backend REST API
Launch the FastAPI server (accessible at `http://127.0.0.1:8000` with Swagger docs at `/docs`):
```bash
uvicorn api.main:app --reload
```

---

## Architecture Overview

The system is split into two components: the Ingestion Pipeline and the Query/RAG Pipeline.

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion Pipeline"]
        direction TB
        PDF["data/ebook-agentic-ai.pdf (PDF Document)"] --> Loader["PyPDFLoader (Load Pages)"]
        Loader --> Splitter["SemanticChunker (Semantic Chunks)"]
        Splitter --> Embed["HuggingFaceEmbeddings (Local Model)"]
        Embed --> DB[("Pinecone Vector DB")]
    end

    subgraph Query["Query / RAG Pipeline"]
        direction TB
        Start([START]) --> RouteInput{"route_input (Is Greeting?)"}
        RouteInput -- "greet" --> GreetNode["greeting_node (Return Greeting)"]
        RouteInput -- "rag" --> HyDE["generate_hyde (llama-3.1-8b-instant)"]
        GreetNode --> End([END])
        HyDE --> Retrieve["retrieve (Retrieve settings.top_k=7)"]
        Retrieve --> Route{"route (Score >= 0.50 Confidence?)"}
        Route -- "YES" --> Generate["generate (llama-3.3-70b-versatile via Groq)"]
        Route -- "NO" --> Fallback["fallback (Return Default Message)"]
        Generate --> End
        Fallback --> End
    end

    DB -.-> Retrieve
```

### LangGraph Design Details
- **`RagState`:** Tracks the current state of the execution (`question`, `search_query`, `context`, `chunks`, `confidence`, and `final_answer`).
- **`route_input` Conditional Edge:** Evaluates the user query first. It queries `llama-3.1-8b-instant` to check if it's a greeting/casual conversation or a domain question. Routes to `greeting_node` if it is a greeting, otherwise routes to `generate_hyde`.
- **`greeting_node` Node:** Returns a friendly static greeting asking the user to query about the book/PDF.
- **`generate_hyde` Node:** Invokes a lightweight LLM (`llama-3.1-8b-instant`) to generate a hypothetical passage answering the user's question, which is stored in `search_query`.
- **`retrieve` Node:** Queries the Pinecone database using cosine similarity. It extracts the top candidate chunks (7 chunks) using the generated hypothetical passage as the search query.
- **`route` Conditional Edge:** Inspects the similarity score of the best retrieved chunk. If the score is less than `0.50`, it routes to the `fallback` node to prevent hallucinations. Otherwise, it routes to `generate`.
- **`generate` Node:** Formulates a prompt combining the user's question and all 7 retrieved context chunks, passing it to `llama-3.3-70b-versatile` hosted on Groq with instructions to answer strictly from the context.
- **`fallback` Node:** Instantly returns the message: *"I don't know based on the provided PDF. What can I help you with? Please ask questions from the PDF."*

---

## Sample Queries

Here are 5–6 sample queries you can use to test the chatbot's RAG grounding capabilities:

1. **`What are the core components of an Agentic AI system?`**
   *(Tests context retrieval and structured grounding for core definitions.)*
2. **`What is the difference between single-agent and multi-agent systems?`**
   *(Tests detailed concept comparison.)*
3. **`Explain the role of memory in AI agents.`**
   *(Tests explanation capabilities of specific technical concepts in the book.)*
4. **`What are the key design patterns or challenges of Agentic AI?`**
   *(Tests retrieval across multiple pages/paragraphs.)*
5. **`Who won the 2022 FIFA World Cup?`**
   *(Tests the strict grounding feature. Since this is outside knowledge not in the PDF, the chatbot must refuse to answer and return: "I don't know based on the provided PDF. What can I help you with? Please ask questions from the PDF.")*
