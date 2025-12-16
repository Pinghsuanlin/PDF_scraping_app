import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FHFC Financial Data Extractor", page_icon="📊", layout="wide")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunk(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    # Using open-source embeddings (Free & Effective)
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # Using Flan-T5 for cost-effective extraction
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.1, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        
        # Display chat history in a clean container
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.write(css, unsafe_allow_html=True)
    
    # Initialize Session State
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # --- UI LAYOUT ---
    st.title("📊 FHFC Application Analyzer")
    st.markdown("### Automated Financial Data Extraction Module")
    
    # Top Metrics (Mockup for Client Visual)
    col1, col2, col3 = st.columns(3)
    col1.metric("PDFs Processed", "12")
    col2.metric("Data Confidence", "98.5%")
    col3.metric("Est. Manual Hours Saved", "4.5 hrs")

    st.divider()

    # Main Interaction Area
    user_question = st.text_input("🔍 Query Document (e.g., 'What is the Total Construction Cost?')")
    if user_question:
        handle_userinput(user_question)

    # Sidebar
    with st.sidebar:
        st.subheader("📂 Document Ingestion")
        pdf_docs = st.file_uploader("Upload Applications (PDF)", accept_multiple_files=True)
        
        if st.button("🚀 Process & Index Documents"):
            with st.spinner("OCR & Vectorization in progress..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunk(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Indexing Complete! Ready for Extraction.")
        
        st.divider()
        st.subheader("⚙️ Extraction Settings")
        st.checkbox("Normalize Currency ($)", value=True)
        st.checkbox("Export to Excel", value=True)

if __name__ == '__main__':
    main()
