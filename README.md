# AI Career Coach Using Traditional RAG

An end-to-end **Retrieval-Augmented Generation (RAG)** application that analyzes a candidate's resume against a job description and generates personalized career guidance.

The project demonstrates how traditional RAG can retrieve relevant information from uploaded documents and use a Large Language Model (LLM) to generate context-based answers.

## Technologies Used

- **Python 3.11.9** – Main programming language
- **LangChain** – RAG pipeline orchestration and prompt management
- **Groq Chat Model** – LLM-based answer generation
- **HuggingFace Sentence Transformers** – Text embeddings
- **ChromaDB** – Vector storage and similarity search
- **Streamlit** – Interactive web application
- **PyPDF** – PDF text extraction
- **python-docx / docx2txt** – Word document processing
- **Pandas and NumPy** – Data handling where required

## Features

- Upload resumes in `.txt`, `.pdf`, and `.docx` formats.
- Upload job descriptions in `.txt`, `.pdf`, and `.docx` formats.
- Split documents into smaller text chunks.
- Generate embeddings for document chunks.
- Store embeddings in ChromaDB.
- Retrieve relevant resume and job-description context for a selected question.
- Generate career guidance, including:
  - Resume and job-description match analysis
  - Missing skills and skill-gap analysis
  - Resume improvement suggestions
  - Recommended projects and learning areas
  - Interview preparation questions
- Generate a career analysis report.
- Adjust chunk size and chunk overlap through the Streamlit interface.

## RAG Pipeline

The application follows these stages:

1. **Document Loading:** Extract text from the uploaded resume and job description.
2. **Chunking:** Split the extracted text into smaller, manageable chunks.
3. **Embedding Generation:** Convert chunks into numerical vector representations using a HuggingFace embedding model.
4. **Vector Database Storage:** Store the vectors and associated document text in ChromaDB.
5. **Query Processing:** Convert the selected career-related question into an embedding through the retrieval pipeline.
6. **Similarity Search:** Retrieve document chunks relevant to the question.
7. **Prompt Construction:** Combine the retrieved context with the question and instructions.
8. **LLM Generation:** Send the prompt to the Groq-hosted language model to generate a response grounded in the retrieved context.

## Project Structure

```text
AI_CAREER_COACH/
│
├── app.py
├── requirements.txt
├── README.md
├── .env                  # Local API key; do not commit
├── .gitignore
│
├── src/
│   ├── rag_engine.py
│   └── file_utils.py
│
├── data/                 # Optional sample files
│
└── .venv311/             # Local virtual environment; do not commit
```

The ChromaDB persistence directory is created locally when the application builds its vector index. Its exact name depends on the configuration in `src/rag_engine.py`.

## Prerequisites

- Windows, macOS, or Linux
- Python 3.11.9 recommended for this project setup
- Git
- A Groq API key
- Internet access for installing dependencies and calling the Groq API

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI_CAREER_COACH
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment

For the current Windows setup:

```powershell
py -3.11 -m venv .venv311
```

If Python 3.11 is already selected as the default Python version, you can use:

```powershell
python -m venv .venv311
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv311\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
```

### 4. Install dependencies

Upgrade pip and install the packages listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the dependency versions specified in `requirements.txt` for the tested project environment.

### 5. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own API key. Never commit `.env` or publish your API key on GitHub.

### 6. Run the application

From the project root, with the virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## How to Use

1. Open the AI Career Coach application.
2. Upload your resume.
3. Upload the job description for the role you want to apply for.
4. Select the desired chunk size and overlap settings.
5. Click **Build Career Coach RAG Index**.
6. Select a career-related question.
7. Click **Get Career Coach Answer**.
8. Review the generated guidance.
9. Generate the complete career analysis report if required.

For meaningful results, use a resume and a job description with readable text.

## Example Questions

- What skills in my resume match this role?
- What are the missing skills for this role?
- How can I improve my resume for this job?
- What projects should I build to improve my profile?
- What interview questions should I prepare for this role?

## Sample Data

Optional sample resumes and job descriptions can be placed in the `data/` directory for testing. You can also upload your own documents through the application interface.

## Security and Repository Notes

Before pushing the project to GitHub, ensure that `.gitignore` excludes:

```gitignore
.venv311/
venv/
.env
__pycache__/
*.pyc
career_coach_chroma_db/
career-coach_chroma_db/
career-coach_chroma_db_v2/
```

Do not upload API keys, private resumes, confidential job descriptions, or unnecessary local vector database files.

## Limitations

- Generated recommendations depend on the quality and completeness of the uploaded documents.
- The model may occasionally produce inaccurate information; verify important career decisions independently.
- The Groq API requires a valid API key and is subject to its current usage limits and model availability.
- The project uses traditional retrieval with embeddings and vector similarity search; it does not require a separate fine-tuning step for the resume and job description.

## Future Enhancements

- Add resume match scoring with a clearly defined scoring methodology.
- Improve retrieval quality and source attribution.
- Add downloadable career reports.
- Add support for additional document formats.
- Deploy the application to a cloud platform.

## Learning Outcomes

This project demonstrates document processing, text chunking, embeddings, vector databases, semantic retrieval, prompt construction, LLM integration, and the development of an interactive RAG application using Streamlit.

---

**Project:** AI Career Coach Using Traditional RAG  
**Primary Language:** Python 3.11.9  
**Interface:** Streamlit  
**LLM Provider:** Groq  
**Vector Database:** ChromaDB