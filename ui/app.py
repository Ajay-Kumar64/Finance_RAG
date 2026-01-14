import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"


st.set_page_config(page_title="Finance RAG Assistant", layout="wide")
st.title("Finance RAG Assistant")

query = st.text_input("Enter your finance question:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Fetching answer..."):
            try:
                resp = requests.get(API_URL, params={"q": query}).json()

                if "answer" in resp:
                    st.subheader("Answer:")
                    st.markdown(resp["answer"])  # markdown preserves formatting
                else:
                    st.error("No answer returned by API. Full response:")
                    st.write(resp)

                if "citations" in resp:
                    st.subheader("Citations (chunks):")
                    for c in resp["citations"]:
                        st.write(f"- {c}")
            except Exception as e:
                st.error(f"Error: {e}")
