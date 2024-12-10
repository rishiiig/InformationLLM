# import os
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.embeddings import GeminiEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # from langchain.llms import Gemini
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from dotenv import load_dotenv


# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY


# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader= PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+= page.extract_text()
#     return  text


# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings()
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     return vector_store


# def get_conversational_chain(vector_store):
#     llm=ChatGoogleGenerativeAI()
#     memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
#     return conversation_chain




import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure API key is properly set
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please check your environment variables.")

# Set the API key in the environment
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# Function to extract text from a list of PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Function to split the extracted text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store using FAISS and Google Generative AI embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/z",
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to create a conversational chain for retrieving information from the vector store
def get_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY"),
        model = "models/chat-bison-001",
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain
