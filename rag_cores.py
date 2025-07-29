import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings 
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader, Docx2txtLoader
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA

OLLAMA_MODEL = "llama3.2:1b"
DATA_PATH = "./data"
VECTORDB_PATH = "./chroma_db"

def load_and_split_docs(data_path : str):
    docs = []
    for fname in os.listdir(data_path):
        fpath = os.path.join(data_path,fname)
        if fname.endswith(".txt"):
            print(f"Loding TXT: {fname}")
            loder = TextLoader(fpath)
            docs.extend(loder.load())
        elif fname.endswith(".pdf"):
            print(f"loding PDF: {fname}")
            loder = PyPDFLoader(fpath)
            docs.extend(loder.load())
        elif fname.endswith(".docx"):
            print(f"loding DOC:{fname}")
            loder = Docx2txtLoader(fpath)
            docs.extend(loder.load())
    print(f"total documents loaded:{len(docs)}")
    splitter =CharacterTextSplitter(chunk_size = 500, chunk_overlap =50) # how to splitter ANS : i used CharacterTextSplitter.
    split_doc =splitter.split_documents(docs)
    print(f"total splitter chunk",{len(split_doc)})
    return split_doc

def get_vectorstore(docs):

   if os.path.exists(VECTORDB_PATH):
       shutil.rmtree(VECTORDB_PATH) # remove all vectordb_path
       embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)  # how to embedding
       vectordb = Chroma.from_documents(  # how to index
           docs, embeddings,persist_directory =VECTORDB_PATH 
    )
       return vectordb

  
def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever()  # vectordb thisukoni embedding genarete chesthundi plus llm base chesukoni answer esthundi
    llm = Ollama(model = OLLAMA_MODEL)
    qa = RetrievalQA.from_chain_type(llm= llm, retriever=retriever) # rightside retriever assigned variable
    return qa
   

           
       


       
        

        
