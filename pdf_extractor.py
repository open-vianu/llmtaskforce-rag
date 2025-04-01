from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from PyPDF2 import PdfReader
from ollama import chat
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_text_from_pdf(pdf):
    """Extract text from a single PDF file."""
    print('\n\nINSIDE EXTRACT TEXT FROM PDF FUNCTION\n\n')
    text = ""
    try:
        pdf_reader = PdfReader(pdf)
        counter = 0
        for page in pdf_reader.pages:

            #print(f'ATTEMPTING TO CORRECT THE TEXT IN PAGE {counter}')
            # OLLAMA
            #prompt = 'Correct the following text eliminating empty steps of words that are split. Text to correct is: '
            #response = chat(model='llama3.2', messages=[{'role': 'user',
            #                                             'content': prompt + page.extract_text()}])
            #text += response['message']['content']
            text += page.extract_text()
            counter +=1

    except Exception as e:
        st.warning(f"Error processing file: {pdf.name}. Skipping...")
    return text

def get_pdf_text(pdf_docs):
    """Process multiple PDFs in parallel to extract text."""
    text = ""
    with ThreadPoolExecutor() as executor:
        try:
            results = executor.map(extract_text_from_pdf, pdf_docs)
            for result in results:
                text += result
        except Exception as e:
            st.error("Error processing PDFs. Please try again.")
    #print(f'\nTHIS IS THE EXTRACTED TEXT:\n{text}')
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator=".",
        #separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print('\nLENGTHS OF INDIVIDUAL CHUNKS\n')
    print([len(chunk) for chunk in chunks])
    print(f'\nTOTAL NUMBER OF CHUNKS:\n {len(chunks)}')
    return chunks

def get_vectorstore(text_chunks):
    # Replace OpenAI embeddings with HuggingFace-based or Ollama-compatible embeddings.
    # No environment variable setup for Ollama API key required based on availability.
    try:
        print('TRY GET VECTORSTORE')
        #embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl",
        #                           model_kwargs={'device': 'mps'}, encode_kwargs={'device': 'mps'})

        embeddings = HuggingFaceEmbeddings(model_name="NovaSearch/stella_en_400M_v5")
        #embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")

        #embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={'device': 'mps'}, encode_kwargs={'device': 'mps'})

        #embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")

        #FIX:


        print('GET FAISS')
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print('FAILED VECTORSTORE')
        print("Error generating vector store: " + str(e))
        st.error("Error generating vector store: " + str(e))
        return None