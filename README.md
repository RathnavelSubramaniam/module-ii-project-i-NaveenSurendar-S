# 🩺 AI Medical Diagnosis Assistant: RAG with Merck Manuals

An **AI-powered Medical Diagnosis Assistant** built using **Retrieval-Augmented Generation (RAG)** to provide evidence-based responses to medical queries using information retrieved from the **Merck Manuals**.

The project combines a Large Language Model (LLM) with medical knowledge retrieval to improve the reliability, relevance, and traceability of generated responses.

## Project Overview

Large Language Models can generate detailed answers to medical questions, but their responses may be based only on pre-trained knowledge and may not provide direct references to authoritative medical sources.

This project addresses this limitation by implementing a **Retrieval-Augmented Generation (RAG)** pipeline.

The system:

1. Loads medical information from the **Merck Manuals**.
2. Processes and splits the medical documents into smaller chunks.
3. Converts text chunks into vector embeddings.
4. Stores the embeddings in a vector database.
5. Retrieves relevant medical information based on a user's query.
6. Passes the retrieved context to an LLM.
7. Generates a context-aware medical response.

The notebook also compares **LLM responses using prompt engineering alone** with responses generated using the **RAG approach**.

---

## Objectives

* Develop an AI-based medical question-answering system.
* Use authoritative medical documentation as an external knowledge source.
* Implement a Retrieval-Augmented Generation pipeline.
* Improve the relevance and groundedness of LLM responses.
* Reduce dependence on the LLM's pre-trained knowledge.
* Demonstrate how retrieved context can support medical question answering.
* Compare traditional prompt engineering with RAG-based generation.

---

## Data Source

The project uses the **Merck Manuals** as the primary medical knowledge source.

The notebook describes the manual as a comprehensive medical reference covering topics such as:

* Diseases and disorders
* Medical tests
* Diagnosis
* Treatments
* Drugs
* Critical care
* Surgery
* Dermatology
* Neurology

The project uses a PDF containing **more than 4,000 pages divided into 23 sections**.

---

## System Architecture

```text
                 ┌─────────────────────┐
                 │   Merck Manuals PDF  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Document Loading  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Text Splitting /    │
                 │ Chunking             │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Text Embeddings     │
                 │ Sentence Transformer│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Chroma Vector Store │
                 └──────────┬──────────┘
                            │
                    User Medical Query
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Similarity Retrieval│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ LLaMA-2 13B Chat    │
                 │       Model         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Medical Response    │
                 └─────────────────────┘
```

---

## Models and Technologies

### Large Language Model

The project uses:

**LLaMA-2 13B Chat**

The notebook downloads the model in **GGUF format** and runs it using `llama-cpp-python`.

### Embeddings

The project uses:

```python
SentenceTransformerEmbeddings
```

for converting medical text into numerical vector representations.

### Vector Database

**Chroma** is used as the vector store for storing and retrieving relevant document embeddings.

### Framework

**LangChain** is used to build and connect the different components of the RAG pipeline.

---

## Technologies Used

| Technology             | Purpose                             |
| ---------------------- | ----------------------------------- |
| Python                 | Main programming language           |
| LangChain              | RAG pipeline and LLM integration    |
| LLaMA-2 13B            | Large Language Model                |
| llama-cpp-python       | Local LLaMA model inference         |
| Sentence Transformers  | Text embeddings                     |
| ChromaDB               | Vector database                     |
| PyMuPDF                | PDF processing                      |
| Hugging Face Hub       | Model downloading                   |
| Pandas                 | Data processing and result analysis |
| Jupyter / Google Colab | Development environment             |

## The notebook installs and uses packages including LangChain, ChromaDB, PyMuPDF, datasets, evaluate, NumPy, Sentence Transformers, and `llama-cpp-python`.

## Project Workflow

### 1. Install Dependencies

Required Python libraries are installed for the RAG pipeline and LLM inference.

### 2. Load Medical Documents

The Merck Manuals PDF is used as the external knowledge source.

### 3. Process the Documents

The medical content is divided into smaller chunks suitable for embedding and retrieval.

### 4. Generate Embeddings

Each document chunk is converted into a vector representation using Sentence Transformers.

### 5. Store Embeddings

The generated embeddings are stored in **ChromaDB**.

### 6. Retrieve Relevant Context

When the user enters a medical question, the vector database searches for relevant medical content.

### 7. Generate the Answer

The retrieved context is provided to the LLaMA-2 13B model to generate the final response.

---

## Prompt Engineering Baseline

Before implementing RAG, the project evaluates the LLaMA-2 model using prompt engineering alone.

The model is configured with:

```python
max_tokens=500
temperature=0.3
top_p=0.95
repeat_penalty=1.15
```

The system prompt instructs the model to act as a medical expert AI assistant and provide concise, evidence-based answers.

---

## Sample Medical Questions

The project tests the LLM with several medical questions, including:

### 1. Sepsis

> What is the protocol for managing sepsis in a critical care unit?

### 2. Appendicitis

> What are the common symptoms for appendicitis, and can it be cured via medicine?

### 3. Hair Loss

> What are the effective treatments or solutions for addressing sudden patchy hair loss?

### 4. Brain Injury

> What treatments are recommended for a person who has sustained a physical injury to brain tissue?

## These questions cover areas including **critical care, surgery, dermatology, and neurology**.

## Prompt Engineering Results

The project stores the questions and generated responses in a Pandas DataFrame:

```python
prompt_result_df = pd.DataFrame({
    "questions": [question_1, question_2, question_3, question_4],
    "prompt_Engineering_responses": [
        response_1,
        response_2,
        response_3,
        response_4
    ]
})
```

This allows the generated answers to be organized and compared.

---

## Prompt Engineering vs RAG

| Feature                             | Prompt Engineering | RAG      |
| ----------------------------------- | ------------------ | -------- |
| External medical knowledge          | ❌                  | ✅        |
| Uses Merck Manuals                  | ❌                  | ✅        |
| Context retrieval                   | ❌                  | ✅        |
| Knowledge groundedness              | Limited            | Improved |
| Source-based answers                | Limited            | Better   |
| Dependence on pre-trained knowledge | High               | Reduced  |
| Medical document retrieval          | ❌                  | ✅        |

The notebook specifically identifies the lack of authoritative references and direct source grounding as a major limitation of prompt-only responses.

---

## Key Features

* Medical question answering
* Medical document-based knowledge retrieval
* Semantic search
* LLaMA-2 13B language model
* Retrieval-Augmented Generation
* Vector-based document retrieval
* Prompt-engineering evaluation
* Merck Manuals knowledge base
* Local LLM inference using `llama-cpp-python`

---

## Project Structure

```text
AI-Medical-Diagnosis-Assistant/
│
├── Module_II_Project_II__final(1).ipynb
├── README.md
│
├── data/
│   └── merck_manuals.pdf
│
├── models/
│   └── llama-2-13b-chat.Q5_K_M.gguf
│
└── chroma_db/
    └── vector_store/
```

> Large model files and medical PDFs generally should not be committed directly to GitHub. Consider using Git LFS or providing download instructions instead.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AI-Medical-Diagnosis-Assistant.git
cd AI-Medical-Diagnosis-Assistant
```

Install the required packages:

```bash
pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install langchain-text-splitters
pip install langchain-chroma
pip install chromadb
pip install sentence-transformers
pip install pymupdf
pip install huggingface_hub
pip install llama-cpp-python
pip install pandas
```

---

## How to Run

### Option 1 — Google Colab

1. Open the `.ipynb` notebook in Google Colab.
2. Run the dependency installation cells.
3. Restart the runtime when required.
4. Run the notebook cells sequentially.
5. Download/load the LLaMA-2 model.
6. Load the medical document.
7. Build the vector database.
8. Enter a medical question.
9. Retrieve relevant context.
10. Generate the final answer.

The notebook itself recommends restarting the Colab runtime/kernel after dependency installation before continuing execution.

---

## Example

```text
User:
What are the symptoms of appendicitis?

        ↓

Query Embedding

        ↓

ChromaDB Similarity Search

        ↓

Relevant Medical Context

        ↓

LLaMA-2 13B

        ↓

Context-Aware Response
```

---

## Advantages

* Provides answers using an external medical knowledge source.
* Reduces reliance on the LLM's internal knowledge.
* Makes the generated responses more context-aware.
* Supports retrieval from large medical documents.
* Provides a foundation for evidence-based medical question answering.
* Demonstrates the practical application of RAG in healthcare.

---

## Limitations

* The system is an educational prototype and should not be used for clinical decision-making.
* LLM-generated responses can still contain errors.
* Medical information requires continuous updating.
* Large LLaMA models require significant computational resources.
* The notebook's prompt-only approach does not inherently provide authoritative source references.
* Dependency versions may need adjustment depending on the execution environment.

The notebook also records package/dependency conflicts during installation, so reproducibility may require matching or updating the package versions used by the notebook.

---

## Future Enhancements

* Add a web-based user interface using Streamlit or Flask.
* Add citation and page-level references from the medical documents.
* Improve retrieval using hybrid search.
* Add reranking of retrieved documents.
* Add conversation memory.
* Add multilingual medical question answering.
* Add evaluation metrics for RAG responses.
* Improve hallucination detection.
* Add automatic source verification.
* Deploy the system as a healthcare information assistant.

---

## Conclusion

The **AI Medical Diagnosis Assistant** demonstrates how **Retrieval-Augmented Generation** can combine a Large Language Model with an external medical knowledge base.

The project first demonstrates medical question answering using prompt engineering and then moves toward a RAG-based architecture to address the limitations of relying only on pre-trained LLM knowledge. The overall goal is to create a more relevant, grounded, and source-aware medical AI prototype.

---

## Author

**Naveen Surendar S**

---