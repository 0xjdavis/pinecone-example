import streamlit as st

import os
from pathlib import Path

# DISPLAY IMAGES AND PDF
import fitz  # PyMuPDF
from PIL import Image
import io
#from dotenv import load_dotenv

# PINECONE
from pinecone import Pinecone, ServerlessSpec

# LLAMAINDEX STORAGE USING PINECONE
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader #, ServiceContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext

# LLM & Embeddings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

# MARKDOWN
from IPython.display import Markdown, display

# WORKFLOW
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

# WORKFLOW OBSERVABILITY
from llama_index.utils.workflow import draw_all_possible_flows

# COMPONENT - VIEWING HTML FOR WORKFLOW MONITOR
import streamlit.components.v1 as components

class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")

URL = Path("workflow.html")
async def workflow():
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    st.write(result)

if __name__ == "__workflow__":
    import asyncio

    asyncio.run(main())
    
# Load and display the HTML file
with open(URL, 'r') as f:
    html_content = f.read()

draw_all_possible_flows(MyWorkflow, filename="workflow.html")

components.html(html_content, height=600, scrolling=True)
    
# Load environment variables
#load_dotenv()

# OpenAI API
#api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"] 

# Initialize Pinecone
PINECONE_API_KEY=st.secrets["PINECONE_API_KEY"]

#api_key = os.getenv("PINECONE_API_KEY")
api_key = PINECONE_API_KEY
index_name = "llamaindex-docs"

# Pinecone Init
if not api_key:
    st.error("Pinecone API key is not set. Please check your .env file.")
    st.stop()

pc = Pinecone(api_key=api_key)


# Streamlit app title
st.title("RAG with Citations using LlamaIndex & Pinecone")

# Function to load and index documents
@st.cache_resource
def load_and_index_documents():
    # Load documents from a directory
    documents = SimpleDirectoryReader("data").load_data()
    
    # Initialize Pinecone vector store
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # Create a storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # LLM & EMBED SETTINGS with OPENAI
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 3900

    # Create and return the index
    return VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        #service_context=service_context
    )

# Load and index documents
with st.spinner("Loading and indexing documents... This may take a while."):
    try:
        index = load_and_index_documents()
    except Exception as e:
        st.error(f"An error occurred while loading and indexing documents: {str(e)}")
        st.stop()

# Function to query the index and get response with citations
def query_index(query_text):
    query_engine = index.as_query_engine(
        response_mode="tree_summarize",
        verbose=True,
        use_async=True
    )
    response = query_engine.query(query_text)
    return response

# Streamlit UI
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating response..."):
        try:
            response = query_index(query)
            
            st.subheader("Answer:")
            st.write(response.response)
            
            st.subheader("Sources:")
            for source_node in response.source_nodes:
                st.markdown(f"- [{source_node.node.metadata['file_name']}]({source_node.node.metadata['file_path']}) (Score: {source_node.score:.2f})")
                with st.expander("View source text"):
                    st.write(source_node.node.get_content())
                    
                    file_name = source_node.node.metadata['file_name'].lower()
                    file_path = source_node.node.metadata['file_path']
                    
                    if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        st.image(file_path, caption=file_name)
                    elif file_name.endswith('.pdf'):
                        try:
                            doc = fitz.open(file_path)
                            for page in doc:
                                pix = page.get_pixmap()
                                img = Image.open(io.BytesIO(pix.tobytes()))
                                st.image(img, caption=f"{file_name} - Page {page.number + 1}")
                        except Exception as e:
                            st.error(f"Error displaying PDF: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")

with st.sidebar.expander("How to use this app", expanded=False, icon="📖"):
    st.markdown("""
    1. Make sure you have documents in the 'data' directory.
    2. Enter your question in the text input.
    3. The app will generate a response using LlamaIndex and display it along with the sources.
    4. Click on the source links to view the original documents.
    5. Expand the "View source text" to see the relevant text from each source.
    """)
    
with st.sidebar.expander("Evironmental Variables", expanded=False, icon="🪝"):
    st.markdown("""
    Note: This app requires setting up environment variables for Pinecone and OpenAI API keys.
    
    Required Environment Variables:
    - PINECONE_API_KEY
    - OPENAI_API_KEY
    """)
    
    # Display environment variable status
    st.subheader("Environment Variables Status:")
    env_vars = ["PINECONE_API_KEY", "OPENAI_API_KEY"] # "PINECONE_ENVIRONMENT",
    
    for var in env_vars:
        if os.getenv(var):
            st.success(f"{var}: Set")
        else:
            st.error(f"{var}: Not Set")

st.sidebar.write("Workflow Observability")
st.sidebar.html('<a href="../src/workflow.html" target="_blank">Observe</a>')

# Load and display the HTML file
with open('workflow.html', 'r') as f:
    html_content = f.read()

components.html(html_content, height=600, scrolling=True)

