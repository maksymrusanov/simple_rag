from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "chroma"
OLLAMA_MODEL = "llama3:8b"

# 1️⃣ Load PDF
pages = PyPDFLoader("data/data.pdf").load()

# 2️⃣ Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = text_splitter.split_documents(pages)

# 3️⃣ Embeddings + DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=CHROMA_PATH,
)

# 4️⃣ Retriever
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "lambda_mult": 0.6},
)

# 5️⃣ LLM
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)

# 6️⃣ Prompt
PROMPT = PromptTemplate(
    template="""
You are a helpful assistant.

Answer the question using ONLY the context below.
You may summarize or paraphrase.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

# 7️⃣ QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
)

is_running = True
while is_running:
    query = input("Enter your question: ")
    result = qa.invoke({"query": query})
    if query == "exit":
        is_running = False

    print("\nANSWER:\n", result["result"])
