from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
 
docs = PyPDFLoader("data/raw/my_document.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
 
emb  = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db   = Chroma.from_documents(chunks, emb, persist_directory="day08/lc_chroma")
retr = db.as_retriever(search_kwargs={"k": 5})
 
llm  = ChatOllama(model="llama3.2:3b", temperature=0)
prompt = ChatPromptTemplate.from_template(
    """Answer only from context. If missing, say "I don't know based on the provided documents."
Context:
{context}
Question: {question}"""
)
 
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)
 
chain = (
    {"context": retr | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
 
print(chain.invoke("What two RAG formulations does the paper compare?"))