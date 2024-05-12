import langchain
from dotenv import load_dotenv , dotenv_values
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import requests
import sentence_transformers

load_dotenv()


def loudtxt():
    output_file = "filetxt.txt"
    # Read the file using the correct encoding
    with open(output_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Write the text back to a new file, ensuring it's in UTF-8 encoding
    with open("elon_musk_utf8.txt", "w", encoding="utf-8") as f:
        f.write(text) 


    # load text doc from URL w/ TextLoader
    loader = TextLoader("./filetxt.txt")
    txt_file_as_loaded_docs = loader.load()
    return txt_file_as_loaded_docs

def splitDoc(loaded_docs):
    # split docs into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_docs = splitter.split_documents(loaded_docs)
    return chunked_docs


def makeEmbeddings(chunked_docs):
    # Create embeddings and store them in a FAISS vector store
    embedder = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunked_docs, embedder)
    return vector_store

def askQs(vector_store, chain, q):
    # Ask a question using the QA chain
    similar_docs = vector_store.similarity_search(q)
    print(similar_docs)
    resp = chain.run(input_documents=similar_docs, question=q)
    return resp

def loadLLM():
    llm=HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

inb_msg = " The Question   ".lower().strip()

chain = loadLLM()#loading the model

LOCAL_ldocs=loudtxt()#loading the text file 

LOCAL_cdocs = splitDoc(LOCAL_ldocs) #chunked (spliting the text file into chankes)
LOCAL_vector_store = makeEmbeddings(LOCAL_cdocs) # makeEmbeddings
LOCAL_resp = askQs(LOCAL_vector_store, chain, inb_msg)
print(LOCAL_resp)