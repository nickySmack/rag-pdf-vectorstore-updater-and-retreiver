import json
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from google.cloud import storage
#from google.oauth2 import service_account ##use this if you dont want to use gcloud auth
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')


index_name = os.getenv('PINECONE_INDEX')
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

processed_files_path = 'processed_files.json'



def load_processed_files():
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed_files):
    with open(processed_files_path, 'w') as f:
        json.dump(list(processed_files), f)

def process_pdf(filepath):
    loader = PyPDFLoader(filepath)
    documents = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    return text_splitter.split_documents(documents)


def main():
    #credentials = service_account.Credentials.from_service_account_file('./development-426420-1c8683726ec7.json') #use gcloud auth
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(os.getenv('CLOUD_STORAGE_BUCKET'))
    

    processed_files = load_processed_files()
    new_files = []

    blobs = list(bucket.list_blobs())
    for blob in blobs:
        if blob.name not in processed_files:
            new_files.append(blob)

    if not new_files:
        print("No new files to process.")
        return

    vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

    for blob in new_files:
        print(f"Processing {blob.name}")
        filepath = f"./tmp/{blob.name}"
        blob.download_to_filename(filepath)
        
        docs = process_pdf(filepath)
        vectorstore.add_documents(docs)
        
        os.remove(filepath)
        processed_files.add(blob.name)

    save_processed_files(processed_files)
    print(f"Processed {len(new_files)} new files.")

if __name__ == "__main__":
    main()