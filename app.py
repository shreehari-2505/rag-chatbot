import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot 🤖📄")

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- PDF Upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
pdf_text = ""
if uploaded_file:
    reader = PdfReader(uploaded_file)
    pdf_text = "".join(page.extract_text() for page in reader.pages)
    st.success("PDF uploaded and text extracted!")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask something about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dummy response for now (replace this with your RAG pipeline later!)
    if pdf_text:
        response = f"🤖 (Pretend this is a smart answer about your PDF!)"
    else:
        response = "🤖 Please upload a PDF first!"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Optional: PDF Download Button ---
if uploaded_file:
    st.download_button("Download PDF", uploaded_file.getvalue(), "document.pdf", mime="application/pdf")
