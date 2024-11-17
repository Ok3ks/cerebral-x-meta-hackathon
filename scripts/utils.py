from typing import List,Any, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from llama_parse import LlamaParse
from llama_index.readers.smart_pdf_loader import SmartPDFLoader
import nest_asyncio
import os

nest_asyncio.apply()

def parser_init():
    return LlamaParse(
        api_key=os.getenv('LLAMA_HUB_API_KEY'),
        num_workers=4,
        verbose=True,
        language="en",
    )


def parse_document(file_path, parser=parser_init()):
    """Accepts both a file path and a list of file paths"""
    documents = parser.load_data(file_path)
    return documents


def retrieve_similar(query:str, client: chromadb.Client, collection_name:str="constructionom",  n_results=2): #Maybe add type hints later
    """Retrieves similar documents from a collection"""
    collection = client.get_or_create_collection(name=collection_name)
    results = collection.query( 
        query_texts=[query], # Chroma will embed this for you #can query embedding too
        n_results=n_results # how many results to return
    )
    return results['documents']

def prep_doc_for_upsert_document(
        documents:List[Any],
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')):
    """Upserts a document to the embedding store"""

    embeddings = []
    ids = []
    text = []

    for doc in documents:
        embeddings.append(embedding_model.encode(doc.text))
        ids.append(doc.doc_id)
        text.append(doc.text)
    return (embeddings, ids, text)

def upsert_embeddings(client:chromadb.Client, document:Tuple[List], collection_name:str="constructionom"):
    collection = client.get_or_create_collection(name=collection_name)
    collection.upsert(embeddings=document[2], documents=document[1], ids = document[0])
    print("upsert done")

