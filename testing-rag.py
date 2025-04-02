#!/usr/bin/env python

import argparse
import math
import os
import re
import time
import pandas as pd

from datetime import datetime
from pdf_extractor import get_pdf_text, get_text_chunks, get_vectorstore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from huggingface_hub import login
from dotenv import dotenv_values

config = dotenv_values(".env")

qa_doc = "./mdsap/Q&A_RDC RESOLUTION No. 665, OF MARCH 30, 2022.xlsx"
text_doc = "./mdsap/merged/merged2.pdf"

custom_prompt_template = """
You are a knowledgeable assistant. Use the following context to answer the user's question.
Whenever possible cite the context in your final answer. If the answer is not in the context, 
please say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_prompt_template
)

def get_conversation_chain(vectorstore, model_name, top_k=5):
    """
    Create a conversational retrieval chain with similarity scores.
    """
    try:
        print("Initializing Ollama LLM...")
        #llm = OllamaLLM(model="llama3.2")
        llm = OllamaLLM(model=model_name)
    except Exception as e:
        print("Failed to initialize Ollama LLM.")
        raise RuntimeError(f"Error initializing Ollama model: {str(e)}")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        eturn_messages=True,
        output_key="answer"
    )

    # Custom retriever to include similarity scores
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Create the conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True  # Ensures source documents are included in responses
    )

    return conversation_chain

def get_paraphrase_with_ollama(context: str, ollama_model_name: str):
    """
    Perform a standalone paraphrase using the Ollama model and a given context.

    Parameters:
        context (str): The context information to base the answer on.
        question (str): The question to answer.
        ollama_model_name (str): The name of the Ollama model to use.

    Returns:
        str: The generated answer from the Ollama model.
    """
    try:
        custom_prompt_template = """
        Rewrite the context given in a new answer. Make the answer a statement, 
        not a thinking process or opinion.

        Context:
        {context}
        
        Answer:
        """
        PROMPT_PARAPHRASE = PromptTemplate(
            input_variables=["context"],
            template=custom_prompt_template
        )

        # Initialize the Ollama LLM
        llm = OllamaLLM(model=ollama_model_name)
        print(f"Ollama LLM initialized with model: {ollama_model_name}")

        # Format the prompt using the custom template
        prompt_paraphrase = PROMPT_PARAPHRASE.format(context=context)
        print(prompt_paraphrase)

        # Use LangChain to generate the answer
        response = llm(prompt_paraphrase)
        print(f'CHECK RESPONSE: {response}')
        #response = llm(prompt)

        # Extract and return the generated answer
        return response

    except Exception as e:
        print(f"Error in QA system: {str(e)}")
        return "An error occurred during the QA process."


def read_excel_as_dataframe(file_path):
    """
    Reads an Excel (.xlsx) file into a Pandas DataFrame.

    Parameters:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame containing the contents of the Excel file.
    """
    try:
        # Read the Excel file with the first row as headers
        df = pd.read_excel(file_path, header=0)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embedding-model')
    parser.set_defaults(command=run_default_pipeline)

    subparsers = parser.add_subparsers()

    embed_parser = subparsers.add_parser('embed')
    embed_parser.add_argument('input', action='append')
    embed_parser.add_argument('-m', '--model', default='hkunlp/instructor-xl')
    embed_parser.add_argument('-o', '--output', default='store.idx')
    embed_parser.set_defaults(command=run_embed_pipeline)

    args = parser.parse_args()
    args.command(args)


def run_default_pipeline(args):
    # Record the start time
    start_time = time.time()

    # LOGIN TO HF
    login(config['HF_TOKEN'])

    # SETTINGS
    top_k = 5
    pdf_docs = [text_doc]
    branch_name = "mdsap"  # Set this to the branch name
    model_used_embeddings = "hkunlp/instructor-xl"  # Set this to the embedding model name
    # CHAT ALTERNATIVE: "deepseek-r1:8b"
    # CHAT ALTERNATIVE: "qwen2.5:7b"
    model_used_chat = "qwen2.5:7b"  # Set this to the chat model name
    ollama_paraphrasing_model_name = "llama3.2"  # Replace with desired model

    #------------------------------------------------------------
    #
    # we read the testing document
    #
    #------------------------------------------------------------
    testing_filepath = qa_doc
    print(testing_filepath)
    test_df = read_excel_as_dataframe(testing_filepath)
    print(test_df.head(1))
    test_df = test_df[['ID', 'question', 'ground_truth_human_generated']]
    #test_df = test_df.drop_duplicates()

    print('DATAFRAME')
    print(test_df)

    #------------------------------------------------------------
    #
    # process one by one
    #
    #------------------------------------------------------------
    if args.embedding_model:
        embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.load_local(
            args.embedding_model, embeddings, allow_dangerous_deserialization=True
        )
    else:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        if not text_chunks:
            print("No valid text chunks found. Please check your documents.")
            return

        vectorstore = get_vectorstore(text_chunks)

    print('\nVECTORSTORE OBTAINED')

    # Initialize results list
    results = []

    for i in range(len(test_df)):
        print(f'\n\n\n\n\n\n\n\nProcessing question {i} in dataset')
        # SETUP CONVERSATION CHAIN
        conversation_chain = get_conversation_chain(vectorstore, model_used_chat, top_k)
        current_user_question = test_df['question'].iloc[i]
        q_id = test_df['ID'].iloc[i]
        print(f'\n--- Current user question ---\n: {current_user_question}\n')
        if type(current_user_question) != str:
            print('Invalid user question')
            continue
        
        # Get top-K results with similarity scores
        docs_with_scores = vectorstore.similarity_search_with_score(current_user_question, k=top_k)
        current_response = conversation_chain({"question": current_user_question})
        print('\n--- CURRENT ANSWER ---\n')
        print(current_response['answer'])

        # Optionally display source documents
        print("\n**Relevant Documents with Scores:**\n")
        top_matches = []
        for idx, (doc, score) in enumerate(docs_with_scores):
            print(f"\nDOCUMENT NUMBER {idx + 1}:\n {doc.page_content}... (SCORE DOCUMENT: {score})")
            top_matches.append(doc.page_content)

        # Ensure top_k columns are accounted for
        # Ensure there are exactly 10 matches by padding with empty strings if necessary
        if len(top_matches) < 10:
            top_matches.extend([""] * (10 - len(top_matches)))


        def extract_answer(input_string: str) -> str:
            match = re.search(r"</think>\s*(.+)", input_string, re.DOTALL)
            return match.group(1) if match else ""

        print(f'\nORIGINAL RESPONSE\n: {current_response["answer"]}')
        # Get the answer
        paraphrased = get_paraphrase_with_ollama(current_response['answer'],
                                                 ollama_paraphrasing_model_name)
        print("\nParaphrased:\n", paraphrased)

        # Record results for this question
        results.append({
            "Branch name": branch_name,
            "model_used_embeddings": model_used_embeddings,
            "Model name": model_used_chat,
            "QID": q_id,
            "Question": current_user_question,
            "Generated Answer": paraphrased,
            **{f"Top {idx + 1} retrieved": top_matches[idx] for idx in range(10)}
        })

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

    # Save results to Excel
    if not os.path.exists('data'):
        os.mkdir('data')
    output_filepath = "data/results.xlsx"
    results_df = pd.DataFrame(results)
    results_df['Elapsed seconds for experiment'] = elapsed_time
    results_df["Elapsed seconds for experiment"] = results_df["Elapsed seconds for experiment"].astype(float)
    results_df["date"] = datetime.now().strftime('%d.%m.%Y')
    results_df.to_excel(output_filepath, index=False)
    print(f"\nResults saved to {output_filepath}")


def run_embed_pipeline(args):
    import os

    for input_ in args.input:
        if not os.path.exists(input_):
            raise ValueError(f'"{args.input}" does not exist')
    raw_text = get_pdf_text(args.input)

    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks, args.model)
    vectorstore.save_local(args.output)


if __name__ == "__main__":
    main()
