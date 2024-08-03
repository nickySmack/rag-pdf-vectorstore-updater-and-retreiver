from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain_pinecone import PineconeVectorStore 
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os  

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')



llm = ChatOpenAI(  
    openai_api_key=os.environ.get("OPENAI_API_KEY"),  
    model_name='gpt-3.5-turbo',  
    temperature=0.0  
)  

knowledge = PineconeVectorStore.from_existing_index(
    index_name="dev1",
    embedding=OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
)

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  
    retriever=knowledge.as_retriever()  
)  
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=knowledge.as_retriever()
)

# Define a few questions about the WonderVector5000.
query1 = """How can you tell if a hive is ready to swarm?"""

query2 = """Why should I rough up the inside of my hive bodies?"""

print(f"Query 1: {query1}\n")
print("Chat with knowledge:")
print(qa.invoke(query1).get("result"))
print("\nChat without knowledge:")
print(llm.invoke(query1).content)
print(f"\nQuery 2: {query2}\n")
print("Chat with knowledge:")
print(qa.invoke(query2).get("result"))
print("\nChat without knowledge:")
print(llm.invoke(query2).content)

