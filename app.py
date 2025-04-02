import streamlit as st
from pdf_extractor import get_pdf_text, get_text_chunks, get_vectorstore
#from testing_rag import get_conversation_chain
import importlib
testing_rag = importlib.import_module("testing-rag")

model_used_chat = "deepseek-r1:1.5b" 
top_k = 5

@st.cache_resource
def initialization_conversation_chain():
    st.write("init chain .....")
    # do your stuff here
    pdf_docs = ['./lip-guidance-test/MERGED_cosmetic_guidances2.0.pdf']
    raw_text = get_pdf_text(pdf_docs)
    raw_text = raw_text[1:1000]
    print("raw_text:  " + str(len(raw_text)))
    text_chunks = get_text_chunks(raw_text)
    #text_chunks = text_chunks[1:10]
    print("text_chunks:  " + str(len(text_chunks)))
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = testing_rag.get_conversation_chain(vectorstore, model_used_chat, top_k)

    return raw_text,text_chunks,conversation_chain,vectorstore

raw_text,text_chunks,conversation_chain,vectorstore = initialization_conversation_chain()
st.text_area (label= "rawtext",value = raw_text)
st.write(f"Lenght document: {str(len(raw_text))}")
st.write(f"Num chunks: {str(len(text_chunks))}")

current_user_question = st.text_input("Ask a question to model:")

if current_user_question:
    st.header("User has sent the following questions: ")
    st.write(f"{current_user_question}")
    st.divider()
    #print( "type :" + str(type(current_user_question)))
    docs_with_scores = vectorstore.similarity_search_with_score(current_user_question, k=top_k)
    current_response = conversation_chain({"question": current_user_question})
    answer = current_response["answer"]
    st.divider()
    st.header(f"model has sent the following answer: {answer}")
    st.write(f"{answer}")