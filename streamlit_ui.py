import streamlit as st
import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from rag_cores import load_and_split_docs,get_vectorstore,get_qa_chain,VECTORDB_PATH,DATA_PATH,OLLAMA_MODEL

st.title("Rag app with langchain and ollama, fastAPI, streamlit")
st.sidebar.header("Data ingestion")
upload_file = st.sidebar.file_uploader("upload a file (.txt, .pdf, .doc")
if upload_file is not None:
    file_path =os.path.join(DATA_PATH, upload_file.name)
    with open(file_path, "wb") as f:
        f.write(upload_file.getbuffer())
        st.sidebar.success("uploaded {uploeded_file.name}")

if st. sidebar.button("Data ingestion"):
    with st. spinner("loading and indexing documents..."):
       docs =load_and_split_docs(DATA_PATH)
       vectordb =get_vectorstore(docs)
       vectordb.persist()
       st.sidebar.success("Data ingested and indexed!")

st.header("Ask question")
question = st.text_input("Enter your question:")
if st.button("submit") and question:
    with st.spinner("Retriveing answer...."):
        vectordb = Chroma(
            persist_directory =VECTORDB_PATH,
            embedding_function=OllamaEmbeddings(model=OLLAMA_MODEL)
        )
        qa = get_qa_chain(vectordb)
        answer = qa.run(question)
        st.success(answer)

st. info(
    "To add knowledge , upload .txt,.pdf,or .doc files in the sidebar and click 'ingest data' "
)        





