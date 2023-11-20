from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import  PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config


def faiss_vector_db():
    docs = PyPDFLoader('data/llm_eval.pdf').load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = config.CHUNK_SIZE, chunk_overlap = config.CHUNK_OVERLAP)

    text = splitter.split_documents(docs)

    embed = HuggingFaceBgeEmbeddings(model_name = config.EMBEDDER,model_kwargs= {'device':'cpu'})
    db = FAISS.from_documents(docs,embed)
    db.save_local(config.VECTOR_DB_PATH)
    print('---Vector store created and saved---')


if __name__=='__main__':
    faiss_vector_db()