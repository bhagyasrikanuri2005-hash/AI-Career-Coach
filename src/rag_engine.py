
import os
import shutil
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DB_dir = "career-coach_chroma_db_v2"


def get_llm(
    model: str = "openai/gpt-oss-20b",
    temperature: float = 0
):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY missing. Check your .env file."
        )

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key
    )


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_text_file(
    file_path: str,
    source_name: str,
    doc_type: str
) -> List[Document]:

    path = Path(file_path)
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return [
        Document(
            page_content=text,
            metadata={
                "source": source_name,
                "doc_type": doc_type
            }
        )
    ]


def create_documents(
    resume_text: str,
    jd_text: str
) -> List[Document]:

    return [
        Document(
            page_content=resume_text,
            metadata={
                "source": "uploaded_resume",
                "doc_type": "resume"
            }
        ),
        Document(
            page_content=jd_text,
            metadata={
                "source": "uploaded_jd",
                "doc_type": "JD"
            }
        )
    ]


def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return splitter.split_documents(docs)


def build_vectorstore(
    chunks: List[Document],
    persist_directory: str = DB_dir
):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_directory,
        collection_name="career_coach_rag"
    )
    return vectorstore


def retrieve_context(
    vectorstore,
    query: str,
    k: int = 6
) -> Tuple[str, List[Document]]:

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return context, docs


def run_career_coach(
    vectorstore,
    resume_text: str,
    jd_text: str,
    question: str
):
    if not resume_text.strip():
        raise ValueError("Please upload or paste your resume.")

    if not jd_text.strip():
        raise ValueError(
            "Please upload or paste the job description."
        )

    llm = get_llm()

    # Retrieve relevant chunks from the vector database.
    retrieval_query = f"""
    Resume skills, education, experience and projects.
    Job description required skills, responsibilities,
    qualifications and experience.
    Career coaching question: {question}
    """

    retrieved_context, source_docs = retrieve_context(
        vectorstore,
        retrieval_query,
        k=6
    )

    # Always include the original resume and job description.
    # This prevents the JD from being omitted from the prompt.
    context = f"""
RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

RETRIEVED CONTEXT FROM VECTOR DATABASE:
{retrieved_context}
"""

    prompt = ChatPromptTemplate.from_template("""
You are an expert AI Career Coach for students, freshers,
and working professionals.

Compare the resume with the job description carefully.

Use the provided resume and job description as the primary
evidence. Do not invent skills, qualifications, experience,
or job requirements. If information is missing, say so.

CONTEXT:
{context}

USER QUESTION:
{question}

Answer ONLY the user's specific question.
Do not provide a complete resume analysis unless explicitly requested.
Do not include unrelated sections.
Use the resume and job description as evidence.
If information is missing, clearly say so.
Keep the answer focused and concise.

If asked for an ATS-style score, provide an approximate
score out of 100 based on the evidence available. Explain
that it is an estimate, not an actual ATS result.

Keep the answer simple, actionable, and beginner-friendly.
""")

    chain = prompt | llm | StrOutputParser()

    try:
        answer = chain.invoke({
            "context": context,
            "question": question
        })

        if not answer or not answer.strip():
            raise ValueError("The model returned an empty answer.")

        return answer, source_docs

    except Exception as e:
        raise RuntimeError(
            f"Career Coach generation failed: {e}"
        ) from e


def generate_complete_report(
    vectorstore,
    resume_text: str,
    jd_text: str
):
    question = """
    Analyze this resume against this job description.

    Provide:
    - Approximate ATS-style match score out of 100
    - Matching skills and qualifications
    - Missing or weak skills
    - Resume improvement suggestions
    - Relevant project suggestions
    - Interview questions for this role

    Support the analysis using only the supplied resume
    and job description. Clearly identify any missing information.
    """

    return run_career_coach(
        vectorstore,
        resume_text,
        jd_text,
        question
    )
