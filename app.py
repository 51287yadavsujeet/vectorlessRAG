import streamlit as st
import fitz
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import time

# =========================================================
# LOGGER CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

# =========================================================
# STREAMLIT SESSION LOGS
# =========================================================

if "logs" not in st.session_state:
    st.session_state.logs = []

# =========================================================
# CUSTOM LOG FUNCTION
# =========================================================

def log(message):

    # Console Log
    print(message, flush=True)

    # Python Logger
    logger.info(message)

    # Streamlit Sidebar Logs
    st.session_state.logs.append(message)

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

log("Loading environment variables...")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:

    log("ERROR: OPENAI_API_KEY not found")

    st.error("OPENAI_API_KEY not found in .env file")

    st.stop()

log("OpenAI API Key Loaded Successfully")

# =========================================================
# OPENAI CLIENT
# =========================================================

client = OpenAI(api_key=OPENAI_API_KEY)

log("OpenAI Client Initialized")

# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Vectorless RAG PDF QA")

st.title("📘 Vectorless RAG - PDF QA")

st.write("Upload PDF and ask questions without vector DB.")

log("Streamlit UI Loaded")

# =========================================================
# PDF FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(pdf_file):

    log("Starting PDF Text Extraction")

    start_time = time.time()

    text = ""

    pdf_document = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    total_pages = len(pdf_document)

    log(f"Total Pages Found: {total_pages}")

    for page_number, page in enumerate(pdf_document):

        log(f"Reading Page {page_number + 1}")

        page_text = page.get_text()

        text += page_text

    extraction_time = time.time() - start_time

    log(
        f"PDF Extraction Completed "
        f"in {extraction_time:.2f} seconds"
    )

    log(f"Total Extracted Characters: {len(text)}")

    return text

# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(text, chunk_size=1000):

    log("Starting Chunk Creation")

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        chunks.append(chunk)

    log(f"Total Chunks Created: {len(chunks)}")

    return chunks

# =========================================================
# VECTORLESS RETRIEVAL
# =========================================================

def retrieve_relevant_chunks(query, chunks):

    log("Starting Vectorless Retrieval")

    log(f"User Query: {query}")

    chunk_text = "\n\n".join(
        [
            f"Chunk {i}:\n{chunk}"
            for i, chunk in enumerate(chunks)
        ]
    )

    prompt = f"""
You are a retrieval system.

User Question:
{query}

Below are document chunks.

{chunk_text}

Return ONLY the chunk numbers most relevant to the question.

Example:
1,4,5
"""

    log("Sending Retrieval Prompt To LLM")

    log(f"Prompt Size: {len(prompt)} characters")

    start_time = time.time()

    try:

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        retrieval_time = time.time() - start_time

        log(
            f"Retrieval LLM Call Completed "
            f"in {retrieval_time:.2f} seconds"
        )

        result = response.choices[0].message.content.strip()

        log(f"LLM Retrieval Response: {result}")

    except Exception as e:

        log(f"ERROR During Retrieval LLM Call: {str(e)}")

        return []

    try:

        indices = [
            int(x.strip())
            for x in result.split(",")
        ]

        log(f"Parsed Chunk Indices: {indices}")

    except Exception as e:

        log(f"ERROR Parsing Chunk Indices: {str(e)}")

        indices = []

    relevant_chunks = [
        chunks[i]
        for i in indices
        if i < len(chunks)
    ]

    log(
        f"Relevant Chunks Retrieved: "
        f"{len(relevant_chunks)}"
    )

    return relevant_chunks

# =========================================================
# FINAL ANSWER GENERATION
# =========================================================

def generate_answer(query, context_chunks):

    log("Starting Final Answer Generation")

    context = "\n\n".join(context_chunks)

    log(
        f"Context Size Sent To LLM: "
        f"{len(context)} characters"
    )

    prompt = f"""
Answer the user's question using ONLY the provided context.

Context:
{context}

Question:
{query}

If answer is not found, say:
"Answer not found in document."
"""

    log("Sending Final Prompt To LLM")

    start_time = time.time()

    try:

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        generation_time = time.time() - start_time

        log(
            f"Answer Generation Completed "
            f"in {generation_time:.2f} seconds"
        )

        answer = response.choices[0].message.content

        log("Final Answer Generated Successfully")

        return answer

    except Exception as e:

        log(f"ERROR During Final LLM Call: {str(e)}")

        return "Error generating answer."

# =========================================================
# MAIN APPLICATION FLOW
# =========================================================

if uploaded_file:

    log("PDF Uploaded Successfully")

    with st.spinner("Reading PDF..."):

        pdf_text = extract_text_from_pdf(uploaded_file)

    chunks = create_chunks(pdf_text)

    st.success("PDF Loaded Successfully ✅")

    st.info(f"Total Chunks Created: {len(chunks)}")

    log("PDF Processing Completed")

    query = st.text_input("Ask Question")

    if query:

        log(f"Received User Question: {query}")

        with st.spinner("Retrieving Relevant Chunks..."):

            relevant_chunks = retrieve_relevant_chunks(
                query,
                chunks
            )

        with st.spinner("Generating Answer..."):

            answer = generate_answer(
                query,
                relevant_chunks
            )

        st.subheader("📌 Answer")

        st.write(answer)

        log("Answer Displayed On UI")

        # =================================================
        # SHOW RETRIEVED CHUNKS
        # =================================================

        with st.expander("📄 Retrieved Chunks"):

            for i, chunk in enumerate(relevant_chunks):

                st.markdown(f"### Chunk {i + 1}")

                st.write(chunk[:1500])

        log("Retrieved Chunks Displayed")

# =========================================================
# SIDEBAR LOGS
# =========================================================

with st.sidebar:

    st.subheader("📜 Application Logs")

    for item in st.session_state.logs[-100:]:

        st.text(item)

log("Application Execution Completed")
