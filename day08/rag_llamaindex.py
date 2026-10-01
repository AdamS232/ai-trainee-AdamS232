from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
 
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
Settings.llm         = Ollama(model="llama3.2:3b", request_timeout=120)
 
docs  = SimpleDirectoryReader("data/raw").load_data()
index = VectorStoreIndex.from_documents(docs)
qe    = index.as_query_engine(similarity_top_k=5)
print(qe.query("What two RAG formulations does the paper compare?"))