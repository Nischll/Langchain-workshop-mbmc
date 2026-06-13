import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()  

RESUME_PATH = "resume.pdf"
JOB_DESCRIPTION_PATH = "job_description.txt"
INDEX_DIR = "faiss_index"  

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(INDEX_DIR):
    vector_store = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
else:
    if not os.path.exists(RESUME_PATH):
        raise FileNotFoundError(
            f"Could not find '{RESUME_PATH}'. Place your resume PDF in this "
            f"folder and name it 'resume.pdf'."
        )

    # 1. LOAD -- read the resume PDF
    docs = PyPDFLoader(RESUME_PATH).load()

    # 2. SPLIT -- cut it into small, searchable chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(docs)

    # 3. EMBED + 4. STORE -- turn chunks into vectors, keep them in FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_DIR)  # save to disk for next time

# Load the job description (used in every question below)
if not os.path.exists(JOB_DESCRIPTION_PATH):
    raise FileNotFoundError(
        f"Could not find '{JOB_DESCRIPTION_PATH}'. Paste the job description "
        f"into this file first."
    )
with open(JOB_DESCRIPTION_PATH, "r", encoding="utf-8") as f:
    job_description = f.read().strip()

llm = ChatGroq(model="qwen/qwen3-32b")

preset_questions = [
    "What skills required by the job are missing from my resume?",
    "Which bullet points should I add or rewrite to better match this job?",
    "On a scale of 1-10, how well does my resume match this job, and why?",
]

print("Resume + job description loaded. Ready!\n")
print("Preset questions you can try (or just type your own):")
for i, q in enumerate(preset_questions, start=1):
    print(f"  {i}. {q}")
print("\nType 'exit' to quit.\n")

while True:
    user_input = input("Your question (or 1/2/3 for presets): ").strip()

    if user_input.lower() in ("exit", "quit"):
        break
    if not user_input:
        continue

    question = preset_questions[int(user_input) - 1] if user_input in ("1", "2", "3") else user_input
    print(f"\n> {question}")

    # 5. RETRIEVE -- find the resume chunks most relevant to the question
    relevant_chunks = vector_store.similarity_search(question, k=3)
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

    # 6. GENERATE -- let Groq answer using the resume context + job description
    prompt = (
        "You are a career coach AI helping a candidate improve their resume "
        "for a specific job. Use the resume excerpts and job description "
        "below to answer the candidate's question. Be specific and actionable.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Resume excerpts:\n{context}\n\n"
        f"Question: {question}"
    )
    answer = llm.invoke(prompt)
    print("\n" + answer.content + "\n")
