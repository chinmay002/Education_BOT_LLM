from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers,ctransformers

from config import *

class Edubotor:

    def __init__(self):
      self.prompt_temp = PROMPT_TEMPLATE
      self.input_variables = INP_VARS
      self.chain_type = CHAIN_TYPE
      self.search_kwargs = SEARCH_KWARGS
      self.embedder = EMBEDDER
      self.vector_db_path = VECTOR_DB_PATH
      self.model_ckpt = MODEL_CKPT
      self.model_type = MODEL_TYPE
      self.max_new_tokens = MAX_NEW_TOKENS
      self.temperature = TEMPERATURE  

    def load_db(self):
       embed = HuggingFaceBgeEmbeddings(model_name =self.embedder, model_kwargs = {'device':'cpu'})
       vector = FAISS.load_local(self.vector_db_path,embed)
       return vector
    
    def custom_prompt_fun(self):
       custom_prompt = PromptTemplate(input_variables = self.input_variables , template= self.prompt_temp)
       return custom_prompt



    def create_llm(self):
       llm = CTransformers(model = self.model_ckpt,
                           model_type = self.model_type,
                           max_new_tokens = self.max_new_tokens,
                           temperature = self.temperature)
       return llm
    
    def retriever_bot(self,custom_prompt,llm,vectordb):
       ret_qa = RetrievalQA.from_chain_type(llm = llm,
                                            chain_type = 'stuff',
                                            retriever = self.load_db().as_retriever(search_kwargs = self.search_kwargs),
                                            return_source_documents = True,
                                            chain_type_kwargs = {'prompt':custom_prompt})
       return ret_qa
    
    def create_bot(self):

        self.custom_prompt = self.custom_prompt_fun()
        self.llm = self.create_llm()
        self.vector_db = self.load_db()
        self.retrieval_qa_chain = self.retriever_bot(self.custom_prompt,self.llm,self.vector_db)
        return self.retrieval_qa_chain