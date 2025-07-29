import uvicorn
from fastapi import FastAPI, UploadFile,File
from pydantic import BaseModel
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from src.rag_cores import load_and_split_docs, get_vectorstore, get_qa_chain, VECTORDB_PATH, DATA_PATH,OLLAMA_MODEL

# FastAPI Setup

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# create class
 
class QuestionRequest(BaseModel):
    question : str

# FastAPI Methods
@app.post("/ingest")
def ingest_files():
    documents = load_and_split_docs(DATA_PATH)
    vectordb = get_vectorstore(documents)
    vectordb.persist()
    return {"status":"Data ingested and indexed"}


@app.post("/ask_question")
def ask_question(req: QuestionRequest):
    vectordb = chroma (
        persist_directory = VECTORDB_PATH,
        embedding_function= OllamaEmbeddings(model=OLLAMA_MODEL)

    )

    qa = get_qa_chain(vectordb)
    answer =qa.run(req.question)
    return {"answer": answer}

@app.post("/upload_file")
def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(DATA_PATH, file.filename)
    with open(file_location,"wb") as f:
        shutil.copyfileobj(file.file,f)
        return{"filename":file.filename}




