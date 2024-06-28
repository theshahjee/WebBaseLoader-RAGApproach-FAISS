from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

load_dotenv()
os.environ['USER_AGENT'] = 'VerticalSolsBot/1.0'

def get_documents():
    loader = WebBaseLoader('https://www.verticalsols.com/')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    split_documents = text_splitter.split_documents(documents)
    return split_documents

def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    return vector_store

def create_chain(vector_store):
    model = ChatOpenAI(temperature=0.1, model='gpt-3.5-turbo')
    prompt = ChatPromptTemplate.from_template(
        "Answer the user's questions\n"
        "Context: {context}\n"
        "Question: {input}"
    )
    doc_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    return retrieval_chain

documents = get_documents()
vector_store = create_vector_store(documents)
chain = create_chain(vector_store)

response = chain.invoke({
    "input": input("Enter question: ")
})

answer = response.get('answer', None)
print(answer)
