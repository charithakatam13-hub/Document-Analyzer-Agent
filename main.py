```python
# ============================================
# LEGAL DOCUMENT ANALYZER WITH RAG
# ============================================

import os

from google.colab import userdata, files

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# ============================================
# OPENROUTER CONFIGURATION
# ============================================

os.environ["OPENAI_API_KEY"] = userdata.get("key")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["MODEL_NAME"] = "openai/gpt-4o-mini"

# ============================================
# PDF UPLOAD
# ============================================

print("Upload your legal document (PDF)")

uploaded = files.upload()

pdf_path = list(uploaded.keys())[0]

agreement_type = input(
    "\nEnter Agreement Type (NDA, Employment, Service, etc.): "
)

# ============================================
# LOAD PDF
# ============================================

loader = PyPDFLoader(pdf_path)

pages = loader.load()

agreement_text = "\n".join(
    [page.page_content for page in pages]
)

# ============================================
# LLM
# ============================================

llm = ChatOpenAI(
    model=os.environ["MODEL_NAME"],
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"]
)

# ============================================
# LEGAL ANALYSIS
# ============================================

prompt = f"""
You are an expert legal document analyzer.

Analyze this document as a {agreement_type}.

Document:
{agreement_text}

Return:

1. Document Summary
2. Parties Involved
3. Important Clauses
4. Risky Clauses
5. Missing Clauses
6. Obligations of Each Party
7. Payment or Penalty Terms
8. Confidentiality Terms
9. Termination Conditions
10. Risk Rating (Low / Medium / High)
11. Practical Recommendation

Do not provide legal advice.

Provide educational analysis only.
"""

response = llm.invoke(prompt)

print("\n")
print("=" * 70)
print("DOCUMENT ANALYSIS REPORT")
print("=" * 70)
print(response.content)

# ============================================
# CHUNKING
# ============================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120
)

chunks = splitter.split_documents(pages)

# ============================================
# EMBEDDINGS
# ============================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"]
)

# ============================================
# VECTOR STORE
# ============================================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# ============================================
# RAG QUESTION LOOP
# ============================================

print("\n")
print("=" * 70)
print("DOCUMENT QUESTION ANSWERING MODE")
print("=" * 70)
print("Ask unlimited questions.")
print("Type 'exit' to quit.\n")

while True:

    question = input("Question: ")

    if question.lower().strip() in [
        "exit",
        "quit",
        "end"
    ]:
        break

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    rag_prompt = f"""
You are a legal document assistant.

Answer ONLY using the provided context.

If the answer is not found, respond:

"Information not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    rag_response = llm.invoke(rag_prompt)

    print("\nAnswer:")
    print(rag_response.content)

    print("\n" + "-" * 70)

print("\nSession Ended.")
```
