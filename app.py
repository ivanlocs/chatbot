from pathlib import Path
from typing import List
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
import os
import uuid
from pinecone import Pinecone, ServerlessSpec
import chainlit as cl
import logging
from langchain_community.vectorstores.azuresearch import AzureSearch

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
PDF_STORAGE_PATH = "./pdfs"

# logging for debug
logging.basicConfig(
    filename="app.log",  # Name of the log file
    level=logging.DEBUG,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
)

# Load environment variables
load_dotenv()

# OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_ADA_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_ADA_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME"
)
AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv(
    "AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME"
)
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# Initialize Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment=AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
    model=AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_ADA_DEPLOYMENT_VERSION,
)

# AI search configuration
VECTOR_STORE_ADDRESS = os.getenv("VECTOR_STORE_ADDRESS")
VECTOR_STORE_PASSWORD = os.getenv("VECTOR_STORE_PASSWORD")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name: str = "langchain-vector-demo"

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=VECTOR_STORE_ADDRESS,
    azure_search_key=VECTOR_STORE_PASSWORD,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

#llm
llm=AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION,
            openai_api_type="azure",
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
            streaming=True,
        )

def process_pdfs(pdf_storage_path: str):
    pdf_directory = Path(pdf_storage_path)
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    # Load PDFs and split into documents
    for pdf_path in pdf_directory.glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        docs += text_splitter.split_documents(documents)

    # # Convert text to embeddings
    # for doc in docs:
    #     embedding = embeddings.embed_query(doc.page_content)
    #     random_id = str(uuid.uuid4())
    #     #print (embedding)
    #     doc_search = pc.Index(index_name)
    #     #doc_search = Pinecone (doc_search, embeddings.embed_query, doc.page_content, random_id)

    # # Store the vector in Pinecone index
    #     doc_search.upsert(vectors = [{"id": random_id, "values": embedding, "metadata": {"source": doc.page_content}}])
    #     print("Vector stored in Pinecone index successfully.")
    vector_store.add_documents(documents=docs)


# process_pdfs(PDF_STORAGE_PATH)

welcome_message = "Welcome to the Chainlit Pinecone demo! Ask anything about documents you vectorized and stored in your Pinecone DB."
namespace = None

from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.vectorstores.pinecone import Pinecone
from langchain_community.retrievers import (
    AzureAISearchRetriever,
)
import pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
@cl.on_chat_start
async def start():
    await cl.Message(content=welcome_message).send()
    # docsearch = Pinecone.from_existing_index(
    #     index_name=index_name, embedding=embeddings, namespace=namespace
    # )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    retriver = AzureAISearchRetriever(
        content_key="content",
        index_name="langchain-vector-demo",
        api_key=VECTOR_STORE_ADDRESS,
        service_name="tomwong",
    )

    # chain = ConversationalRetrievalChain.from_llm(
    #    ,
    #     chain_type="stuff",
    #     retriever=retriver,
    #     memory=memory,
    #     return_source_documents=True,
    # )


    # cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain

    # cb = cl.LangchainCallbackHandler()

    # res = await chain.acall(message.content, callbacks=[cb])
    # retriever = vector_store.as_retriever()
    
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Only Use the following pieces of retrieved context to answer "
    "the question. If the answer cannot be retrieved from the source text, just answer you do not know."
    "\n\n"
    "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )

    # retriever = AzureAISearchRetriever(
    #     content_key="content",
    #     index_name="langchain-vector-demo",
    #     api_key=VECTOR_STORE_ADDRESS,
    #     service_name="tomwong",
    # )
    retriever = vector_store.as_retriever()
    # source_documents = retriever.invoke(message.content)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": message.content})

    answer = response["answer"]
    source_documents = response["context"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            metadata = source_doc.metadata
            source_name = metadata.get("source") + " Page: " + str(metadata.get("page"))
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()

# for debugging
if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)