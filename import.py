import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from io import StringIO
from io import BytesIO
from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')


index_name = "dev1"
embeddings = OpenAIEmbeddings()

# path to an example text file
credentials = service_account.Credentials.from_service_account_file('./development-426420-1c8683726ec7.json')
storage_client = storage.Client(credentials=credentials)
bucket = storage_client.get_bucket('seeleypdfs')

blobs = list(bucket.list_blobs())
for blob in blobs:
    print(blob.name)
    filepath = f"./tmp/{blob.name}"
    blob.download_to_filename(filepath)
    loader = PyPDFLoader(filepath)
    documents = loader.load_and_split()
    os.remove(filepath)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        docs,
        index_name=index_name,
        embedding=embeddings
    )

    


# loader = TextLoader(pdf)
# # documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# vectorstore_from_docs = PineconeVectorStore.from_documents(
#     documents,
#     index_name=index_name,
#     embedding=embeddings
# )

# texts = ["Tonight, I call on the Senate to: Pass the Freedom to Vote Act.", "ne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.", "One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence."]

# vectorstore_from_texts = PineconeVectorStore.from_texts(
#     texts,
#     index_name=index_name,
#     embedding=embeddings
# )