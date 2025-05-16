import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import UnstructuredPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')


### Setup Streamlit app
st.title('Conversation RAG with PDF Upload and Chat History')
st.write("Upload Pdfs and chat with their content")

## Input the Groq API Key
api_key=st.text_input("Enter your Groq API key:" ,type="password")

## Check if Groq API key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")
    session_id=st.text_input("Session ID", value="default_session")
    
    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files= st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=False)

    ## Process Uploaded files:
    if uploaded_files:
        documents=[]
        texts = ""   
        tmp_location = os.path.join('.', uploaded_files.name)
        with open(tmp_location, "wb") as file:
            file.write(uploaded_files.getvalue())
        #tmp_location.write(uploaded_files)
        reader = PyPDFLoader(tmp_location)
            
        docs=reader.load()
        documents.extend(docs)
        
        # Split and Create Embedding for the documents
        text_splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits= text_splitter.split_documents(documents)
        vectorstore= Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever= vectorstore.as_retriever()
        
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "Which might reference context in the chat history, "
            "formulate a standalone question which can be understood"
            "without the chat history. DO NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is"
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever, contextualize_q_prompt)
        
        ## Answer question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer"
            "the question. if you don't know the answer, say that you "
            "don't know. Use maximmum four sentences and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain=create_stuff_documents_chain(llm, qa_prompt)
        rag_chain= create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]= ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        user_input= st.text_input("Your question:")
        if user_input:
            session_history= get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable" : {"session_id": session_id}
                    
                },
                
            )
            
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages) 
            
else:
    st.warning("Please enter groq API key")
        
