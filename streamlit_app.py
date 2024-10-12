import streamlit as st
from pinecone import Pinecone, PodSpec
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import requests

# Pinecone setup
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index_name = "pdf-store"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension for 'all-MiniLM-L6-v2' model
        metric='cosine',
        spec=PodSpec(environment="gcp-starter")  # Use PodSpec for the environment
    )

# Connect to the index
index = pc.Index(index_name)

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

GITHUB_API_URL = st.secrets["GITHUB_API_URL"]

def get_pdf_urls_from_github_directory():
    response = requests.get(GITHUB_API_URL)
    if response.status_code == 200:
        files = response.json()
        pdf_urls = [file['download_url'] for file in files if file['name'].endswith('.pdf')]
        return pdf_urls
    else:
        st.error(f"Failed to retrieve files from GitHub. Status code: {response.status_code}")
        return []

def download_pdf_from_github(url):
    response = requests.get(url)
    pdf_filename = url.split("/")[-1]  # Extract the PDF name from the URL
    with open(pdf_filename, "wb") as f:
        f.write(response.content)
    return pdf_filename

def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as pdf_file:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def create_embeddings(text):
    return model.encode(text)

# Get all PDF URLs from the GitHub directory
pdf_urls = get_pdf_urls_from_github_directory()

# Download and process each PDF from the list
for pdf_url in pdf_urls:
    st.write(f"Processing PDF: {pdf_url.split('/')[-1]}")
    
    pdf_file_path = download_pdf_from_github(pdf_url)
    pdf_text = extract_text_from_pdf(pdf_file_path)
    embedding = create_embeddings(pdf_text)
    
    # You can store the embedding to Pinecone or any other database
    st.write(f"Processed and created embedding for: {pdf_file_path}")

st.write("All PDFs downloaded and processed successfully!")












def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_embeddings(text):
    return model.encode(text)

def store_in_pinecone(pdf_name, embedding):
    index.upsert(vectors=[(pdf_name, embedding.tolist())])

def retrieve_from_pinecone(query):
    query_embedding = model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return results

st.title("PDF Storage and Retrieval with Pinecone")

# PDF upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    embedding = create_embeddings(pdf_text)
    store_in_pinecone(uploaded_file.name, embedding)
    st.success(f"PDF '{uploaded_file.name}' stored successfully!")

# Query interface
# st.header("Retrieve PDFs")
# query = st.text_input("Enter your query:")
# if query:
#    results = retrieve_from_pinecone(query)
#    st.subheader("Retrieved PDFs:")
#    for match in results['matches']:
#        st.write(f"- {match['id']} (Score: {match['score']})")

# RAG implementation
st.header("RAG")
rag_query = st.text_input("Enter your query:")
if rag_query:
    results = retrieve_from_pinecone(rag_query)
    context = "\n".join([match['id'] for match in results['matches']])

    # Generate response using an LLM (mock implementation)
    response = model.encode(f"Response based on context: {context}")
    st.caption("Generated Response:")
    st.write(response)
    st.caption("Context")
    st.write(context)