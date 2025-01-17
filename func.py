import io
import json
import os
import cohere
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from fdk import response
from urllib.parse import urlparse, parse_qs
#from langchain_community.embeddings import CohereEmbeddings
# Load Documents
def load_documents(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return text

# Initialize Cohere client
cohere_api_key = "TX8xfSGQm7btpYjQBrf3qYHyo7M9gAXtGrp2kJtT"
co = cohere.Client(cohere_api_key)
api_url = "https"
# Function to embed texts
def embed_texts(texts):
    response = co.embed(
        texts=texts,
        model='large',
    )
    return response.embeddings

# Function to retrieve relevant documents
def retrieve_relevant_documents(query, document_texts, document_embeddings, top_k=5):
    query_embedding = embed_texts([query])[0]
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [document_texts[i] for i in top_indices]

# Function to generate API URL
def generate_api(template, question, retrieved_text):
    for doc in retrieved_text:
        prompt = template.format(api_docs=doc, question=question)
        response = co.generate(
            prompt=prompt,
            model='command',
            max_tokens=1000,
            return_likelihoods="None",
            temperature=0,
            p=1,
            truncate="END"
        )
        generated = response
        time.sleep(10)
        generated = response.generations[0].text.strip()
        return generated  # Assuming you want the first 115 characters

def handler(ctx, data: io.BytesIO=None):
    print("Entering Python Hello World handler", flush=True)
    query = "World"
    try:
        body = json.loads(data.getvalue())
        
        file_path = "hcm_api_doc.txt"
        query = body.get("query", "Which API can I use to get the project information")

        # Load documents from file
        extracted_text = load_documents(file_path)

        # Embed the extracted text
        documents = [{"content": extracted_text, "id": "txt_1"}]
        document_texts = [doc['content'] for doc in documents]
        document_embeddings = embed_texts(document_texts)

        # Retrieve relevant documents based on query
        retrieved_docs = retrieve_relevant_documents(query, document_texts, document_embeddings)
        retrieved_text = "\n".join(retrieved_docs)

        # Generate API URL using retrieved documents and query
        template = '''
        You are given the below API Documentation:
        {api_docs}
        Using this documentation, generate the full API url to call for answering the user question.
        You should build the API url in order to get a response that is as short as possible, while still getting the necessary information to answer the question.
        Pay attention to deliberately exclude any unnecessary pieces of data in the API call.
        Do not include data not in the documentation.

        Question:{question}
        API url:
        '''

        api_url = generate_api(template, query, retrieved_text)
        parsed_url = urlparse(api_url)
        query = parsed_url.query
        print(query)
        print("Vale of api_url = ", api_url, flush=True)
    except (Exception, ValueError) as ex:
        print(str(ex), flush=True)


    print("Exiting Python Hello World handler", flush=True)
    return response.Response(
        ctx, response_data=json.dumps(
            {"api_url": api_url}),
        headers={"Content-Type": "application/json"}
        )
