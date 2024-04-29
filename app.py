import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
#use faiss as vectorStore to store numeric representation of the text chunk, and it will run local machine and therefore would be erased once we close it
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models.openai  import ChatOpenAI
# from langchain_community.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
#to chat with vectorstore and has memory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms.huggingface_hub import HuggingFaceHub


def get_pdf_text(pdf_docs):
    #initiate raw text containing on text from the pdf
    text = ""
    for pdf in pdf_docs:
        #initiate pdfReader object
        pdf_reader = PdfReader(pdf) #thie create pdf ojbects that has pages
        #loop through pages to actually read through the pdf
        for page in pdf_reader.pages:
            text += page.extract_text() #extract text from the page and append it to text variable
    return text #and return final text variable, which is a single string that will be applied to raw_text below


def get_text_chunk(text):
    text_splitter = CharacterTextSplitter(
        #define parameters
        separator="\n", #a single line break
        chunk_size=1000, #1000 character most
        chunk_overlap=200, #200 characters overlapping
        length_function=len #return the number of character, a python funciton
    )
    #split incoming text and return chunks
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings() #charged but much faster
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") #free but slow
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chaian(vectorstore):
    #llm = ChatOpenAI() #charged
    llm = HuggingFaceHub(repo_id="googl/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})#maybe??
    #initate an instance of memory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    #take the history of conversation, and return an element of which
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
    
def handle_userinput(user_question):
    #use what created in sidebar st.session_state.conversation with key-value pair of user questions
    response = st.session_state.conversation({'question': user_question})
    #st.write(response) #test it out
    st.session_state.chat_history = response['chat_history'] #based on response object shown initially
    
    #loop through the chat_history with index and content of the index
    for i, message in enumerate(st.session_state.chat_history.chat_history):
        if i % 2 == 0: #% modulo
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) #replace with content in the message
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            
    

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with multiple PDFs', page_icon=':books:')
    
    st.write(css, unsafe_allow_html=True)
    
    #to also make conversation persistent
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    #as we use chat_history session_state, we need to initialize up in the application
    if "chat_history" not in st.session_state:
        st.session_state.conversation = None
    
    st.header('Chat with multiple PDFs :books:')
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    
    # st.write(user_template.replace("{{MSG}}", "Hello robot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True) #repalce message with "<text within>"
    #as we created the for loop to track chat history before
    
    #a sidebar where users upload their pdf doc
    with st.sidebar:
        st.subheader("Your documents:")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            #spinner makes it user friendly, and the functions in the spinner will be processed meanwhile
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text) #test it out
                
                #get the text chunk
                text_chunks = get_text_chunk(raw_text)
                #st.write(text_chunks) #test it out
                
                #create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                #create conversation chain
                st.session_state.conversation = get_conversation_chaian(vectorstore)
                #set st.session_state: to tell the app to stick to the conversation and don't initialize, in case users press on random button
    
    st.session_state.conversation #to use conversation object outside of the scope
    

if __name__ == '__main__':
    main()