# Smart Resume & Job Matcher

A RAG-based assistant that compares your resume against a job description
and answers questions like:

- What skills required by the job are missing from my resume?
- Which bullet points should I add or rewrite?
- How well does my resume match this job overall (1-10)?

Built as a homework project for the **Generative AI Applications with
LangChain** workshop at MBM College, following the same pattern used in
class:

```
Load -> Split -> Embed -> Store -> Retrieve -> Generate
```

## How it works

1. **Load** — your resume PDF is read with `PyPDFLoader`.
2. **Split** — it's cut into ~1000-character chunks (100-char overlap).
3. **Embed + Store** — each chunk is converted to an embedding using the
   local, free `all-MiniLM-L6-v2` HuggingFace model and stored in a FAISS
   vector index (saved to `faiss_index/` so it's only built once).
4. **Retrieve** — when you ask a question, the most relevant resume chunks
   are found via similarity search.
5. **Generate** — those chunks, plus the full job description, are sent to
   Groq (`qwen/qwen3-32b`), which answers your question.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key

Copy `.env.example` to `.env` and add your key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Add your resume

Place your resume PDF in this folder and name it `resume.pdf`.

### 4. Add the job description

Open `job_description.txt` and paste in the full job description you want
to match your resume against.

## Usage

```bash
python main.py
```

You'll see a few preset questions (type `1`, `2`, or `3`), or type your own,
for example:

- "Rewrite my third bullet point under Experience to better match this job"
- "What keywords from the job description should I add to my skills section?"

Type `exit` to quit.

> **Note:** If you swap in a different resume, delete the `faiss_index/`
> folder so it rebuilds the vector store from the new file.

## Project structure

```
resume-matcher/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── .env.example         # Template for API key
├── job_description.txt  # Paste the target job description here
├── resume.pdf           # Your resume (add this yourself)
└── README.md
```

## Tools used

- **LangChain** (`langchain`, `langchain-community`, `langchain-huggingface`)
- **Groq** (`qwen/qwen3-32b`) — LLM for generating answers
- **HuggingFace embeddings** (`all-MiniLM-L6-v2`) — free, local embedding model
- **FAISS** — local vector store
- **PyPDFLoader** — PDF parsing

## Notes

- The resume PDF and `.env` are gitignored for privacy — add your own
  before running.
- Embeddings run locally and don't require an API key, so only the Groq
  key is needed.
