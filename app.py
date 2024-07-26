import streamlit as st 
from dotenv import load_dotenv
from htmlTemplates import css,bot_template,user_template
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings 
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len 
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings() 
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorestore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorestore 


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFacePipeline(pipeline="text-generation",model_id = "google/flan-t5-xxl",pipeline_kwargs={"temperature":0.5,"max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(),
        memory = memory 
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDF's",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with multiplt PDF's")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}","WASSAP"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","こんにちは人間 "),unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click Process",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text 
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks 
                text_chunks = get_text_chunks(raw_text)

                # Create Vector Store 
                vector_store = get_vectorstore(text_chunks)

                # create conversation chain 
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()