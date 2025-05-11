import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("🦾 PDF Q&A RAG Demo")

# --- PDF Upload ---
st.header("1. Upload your PDF")

# Use session_state to track upload status!
if "pdf_uploaded" not in st.session_state:
    st.session_state["pdf_uploaded"] = False

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
if uploaded_file is not None and not st.session_state["pdf_uploaded"]:
    with st.spinner("Uploading and indexing..."):
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        res = requests.post(f"{API_URL}/upload_pdf/", files=files)
        if res.status_code == 200:
            st.success(res.json()["message"])
            st.session_state["pdf_uploaded"] = True  # Mark as uploaded!
        else:
            st.error("Upload failed! Check backend logs.")

if st.session_state["pdf_uploaded"]:
    st.info("PDF uploaded and indexed! You can now ask questions below. 🚀")

# --- Ask a Question ---
st.header("2. Ask a Question")
question = st.text_input("Type your question about the PDF:")

if st.button("Get Answer"):
    if not st.session_state["pdf_uploaded"]:
        st.warning("Please upload a PDF first!")
    elif not question:
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking..."):
            payload = {"query": question}
            res = requests.post(f"{API_URL}/ask/", json=payload)
            if res.status_code == 200:
                st.markdown(f"**Answer:** {res.json()['answer']}")
            else:
                st.error("Something went wrong! Check backend logs.")


