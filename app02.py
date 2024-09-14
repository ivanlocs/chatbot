from typing import List  
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI  
from dotenv import load_dotenv  
from langchain.schema import Document  
import os  
import chainlit as cl  
import logging  
import sys  
from langchain_community.chat_message_histories import ChatMessageHistory  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain.chains.retrieval import create_retrieval_chain  
from langchain.chains.combine_documents import create_stuff_documents_chain  
from langchain_core.runnables.history import RunnableWithMessageHistory  
from langchain_core.chat_history import BaseChatMessageHistory  
from langchain.chains import create_history_aware_retriever  
import posixpath  
import urllib.parse  
import json  
from chainlit.input_widget import Select  
from MyAISearchVectorStoreRetriever import MyAzureSearchVectorStoreRetriever  
import random  
import string  
  
# Logging for debug  
logging.basicConfig(  
    level=logging.DEBUG,  # Log level  
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format  
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format  
    handlers=[logging.StreamHandler(sys.stdout)],  
)  
logger = logging.getLogger(__name__)  
  
# Load environment variables  
load_dotenv()  
  
def get_session_history(session_id: str) -> BaseChatMessageHistory:  
    store = cl.user_session.get("store")  
    session_id = cl.user_session.get("session_id")  
    if session_id not in store:  
        store[session_id] = ChatMessageHistory()  
    cl.user_session.set("store", store)  
    return store[session_id]  
  
def setup_llm():  
    # OpenAI configuration  
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  
    AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")  
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")  
    TEMPERATURE = os.getenv("TEMPERATURE")  
      
    llm = AzureChatOpenAI(  
        api_key=AZURE_OPENAI_API_KEY,  
        azure_endpoint=AZURE_OPENAI_ENDPOINT,  
        api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION,  
        openai_api_type="azure",  
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,  
        streaming=True,  
        temperature=TEMPERATURE,  
    )  
    return llm  
  
def setup_embedding():  
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  
    AZURE_OPENAI_ADA_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_ADA_DEPLOYMENT_VERSION")  
    AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME")  
    AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME")  
  
    return AzureOpenAIEmbeddings(  
        deployment=AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,  
        model=AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME,  
        azure_endpoint=AZURE_OPENAI_ENDPOINT,  
        openai_api_key=AZURE_OPENAI_API_KEY,  
        openai_api_version=AZURE_OPENAI_ADA_DEPLOYMENT_VERSION,  
    )  

def setup_retriever(index_name, embeddings):  
    VECTOR_STORE_ADDRESS = os.getenv("VECTOR_STORE_ADDRESS")  
    VECTOR_STORE_PASSWORD = os.getenv("VECTOR_STORE_PASSWORD")  
    SEMANTIC_CONFIG_NAME = os.getenv("SEMANTIC_CONFIG_NAME")  
    TOP_K = int(os.getenv("TOP_K"))  
    SEARCH_TYPE = os.getenv("SEARCH_TYPE")  
    SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD"))  
  
    vector_store = AzureSearch(  
        azure_search_endpoint=VECTOR_STORE_ADDRESS,  
        azure_search_key=VECTOR_STORE_PASSWORD,  
        index_name=index_name,  
        embedding_function=embeddings.embed_query,  
        semantic_configuration_name=SEMANTIC_CONFIG_NAME,  
    )  
  
    return MyAzureSearchVectorStoreRetriever(  
        vectorstore=vector_store,  
        search_type=SEARCH_TYPE,  
        k=TOP_K,  
        search_kwargs={"score_threshold": SCORE_THRESHOLD},  
    )  

def parse_domain(domains):  
    return {domain["domain"]: domain["index"] for domain in domains}  
  
# Generate random session ID  
def generate_random_id(length):  
    characters = string.ascii_letters + string.digits  
    return ''.join(random.choice(characters) for _ in range(length)) 

@cl.on_chat_start  
async def start():  
    session_id = generate_random_id(10)  
    store = {}  
  
    DOMAIN_CHOICE_LABEL = os.getenv("DOMAIN_CHOICE_LABEL")  
    domains = json.loads(os.getenv("DOMAINS"))  
    domain_index_dict = parse_domain(domains)  
  
    # Use the first index as default  
    index_name = domains[0]["index"]  
    domain_name = domains[0]["domain"]  
    welcome_message = f"{os.getenv('WELCOME_MESSAGE')}\nSearching {domain_name}"  
  
    llm = setup_llm()  
    embeddings = setup_embedding()  
    retriever = setup_retriever(index_name, embeddings)  
  
    settings = await cl.ChatSettings(  
        [  
            Select(  
                id="index",  
                label=DOMAIN_CHOICE_LABEL,  
                items=domain_index_dict,  
                initial=domain_name,  
                initial_value=index_name,  
            )  
        ]  
    ).send()  
  
    cl.user_session.set("retriever", retriever)  
    cl.user_session.set("llm", llm)  
    cl.user_session.set("embeddings", embeddings)  
    cl.user_session.set("store", store)  
    cl.user_session.set("session_id", session_id)  
    cl.user_session.set("domain_index_dict", domain_index_dict)  
  
    logger.info(f"A user logged in the system - {session_id}\n")  
    await cl.Message(content=welcome_message).send() 

async def retrieve_answer(message, retriever, llm):  
    session_id = cl.user_session.get("session_id")  
  
    # Contextualize question  
    contextualize_q_system_prompt = (  
        "Given a chat history and the latest user question "  
        "which might reference context in the chat history, "  
        "formulate a standalone question which can be understood "  
        "without the chat history. Do NOT answer the question, "  
        "just reformulate it if needed and otherwise return it as is."  
    )  
  
    contextualize_q_prompt = ChatPromptTemplate.from_messages(  
        [  
            ("system", contextualize_q_system_prompt),  
            MessagesPlaceholder("chat_history"),  
            ("human", "{input}"),  
        ]  
    )  
  
    history_aware_retriever = create_history_aware_retriever(  
        llm, retriever, contextualize_q_prompt  
    )  
  
    # Answer question  
    system_prompt = f"{os.getenv('SYSTEM_PROMPT')}\n\n{{context}}"  
  
    prompt = ChatPromptTemplate.from_messages(  
        [  
            ("system", system_prompt),  
            MessagesPlaceholder("chat_history"),  
            ("human", "{input}"),  
        ]  
    )  
  
    question_answer_chain = create_stuff_documents_chain(llm, prompt)  
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)  
  
    conversational_rag_chain = RunnableWithMessageHistory(  
        rag_chain,  
        get_session_history,  
        input_messages_key="input",  
        history_messages_key="chat_history",  
        output_messages_key="answer",  
    )  
  
    response = await conversational_rag_chain.ainvoke(  
        {"input": message},  
        config={  
            "configurable": {"session_id": session_id}  
        },  
    )  
  
    answer = response["answer"]  
    source_documents = response["context"]  # type: List[Document]  
    return source_documents, answer  

@cl.on_message  
async def main(message: cl.Message):  
    retriever = cl.user_session.get("retriever")  
    llm = cl.user_session.get("llm")  
  
    loading_message = await cl.Message(content="Loading...").send()  
    source_documents, answer = await retrieve_answer(message.content, retriever, llm)  
    elements = []  # type: List[cl.Text]  
  
    # Sort the documents according to the search score  
    source_documents.sort(key=lambda x: x.metadata["score"], reverse=True)  
  
    if source_documents:  
        for source_idx, source_doc in enumerate(source_documents):  
            metadata = source_doc.metadata  
            file_path = metadata.get("file_path")  
            document_score = metadata.get("score")  
            source_name = urllib.parse.unquote(posixpath.basename(file_path))  
  
            sas_token = os.getenv("SAS_TOKEN")  
            url = f"{file_path}?{sas_token}"  
  
            elements.append(cl.File(name=source_name, display="inline", url=url))  
            source_chunk = f"Score: {document_score} \n{source_doc.page_content}"  
            elements.append(cl.Text(content=source_chunk, name=source_name, display="page"))  
  
        # Remove duplicate source names  
        source_names = list(set([text_el.name for text_el in elements]))  
  
        if source_names:  
            answer += "\n\nRelevant Documents: \n"  
            for name in source_names:  
                answer += f"{name}\n"  
        else:  
            answer += "\nNo sources found"  
  
    loading_message.content = answer  
    loading_message.elements = elements  
    await loading_message.update()  

@cl.on_chat_end  
def on_chat_end():  
    session_id = cl.user_session.get("session_id")  
    logger.info(f"A user ended a session - {session_id}\n")  
    store = {}  
    cl.user_session.set("store", store)  
  
def get_domain_name(domain_index_dict, index_name):  
    for k, v in domain_index_dict.items():  
        if v == index_name:  
            return k  
  
@cl.on_settings_update  
async def setup_agent(settings):  
    loading_message = await cl.Message(content="Switching Search Domain").send()  
    index_name = settings["index"]  
    domain_index_dict = cl.user_session.get("domain_index_dict")  
    domain_name = get_domain_name(domain_index_dict, index_name)  
  
    message = f"Searching {domain_name}"  
    retriever = cl.user_session.get("retriever")  
    embeddings = cl.user_session.get("embeddings")  
    retriever = setup_retriever(index_name, embeddings)  
  
    cl.user_session.set("retriever", retriever)  
    loading_message.content = message  
    await loading_message.update()  
  
# For debugging  
if __name__ == "__main__":  
    from chainlit.cli import run_chainlit  
    run_chainlit(__file__)  