"""
Run this once to load all knowledge/ files into Pinecone.
Re-run any time you add or edit knowledge files.

Usage:
    cd backend
    venv/bin/python3 ingest.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

Settings.llm = Anthropic(
    model="claude-sonnet-4-6",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

existing_names = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing_names:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created Pinecone index: {INDEX_NAME}")
else:
    print(f"Using existing index: {INDEX_NAME}")

pinecone_index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print(f"Loading documents from {KNOWLEDGE_DIR} ...")
docs = SimpleDirectoryReader(str(KNOWLEDGE_DIR)).load_data()
print(f"Loaded {len(docs)} document chunks")

print("Ingesting into Pinecone ...")
VectorStoreIndex.from_documents(docs, storage_context=storage_context)
print("Done. Knowledge base is ready.")
