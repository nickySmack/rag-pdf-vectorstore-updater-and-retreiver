from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA, ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
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
    index_name="dev",
    embedding=OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
)
retriever=knowledge.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
If the question is unrelated to bees or beekeeping, just say that you can't help with that.\

{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Create an in-memory chat history store
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# def get_session_history(session_id):
#     return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Conversation with history handling
session_id = "abc123"  # Unique session identifier

chain = conversational_rag_chain.pick("answer")
for chunk in chain.stream({"input": "Tell me an interesting bee fact"},
    config={
        "configurable": {"session_id": session_id}
    }):
    print(chunk, end="", flush=True)

answer = conversational_rag_chain.invoke(
    {"input": "Tell me an interesting bee fact"},
    config={
        "configurable": {"session_id": session_id}
    },
)["answer"]

print(answer)

answer2 = conversational_rag_chain.invoke(
    {"input": "Tell me another interesting bee fact"},
    config={
        "configurable": {"session_id": session_id}
    },
)["answer"]

print(answer2)

answer3 = conversational_rag_chain.invoke(
    {"input": "Write me a python script that adds 2 numbers together."},
    config={
        "configurable": {"session_id": session_id}
    },
)["answer"]

print(answer3)

# # Start a conversation
# print("Chatbot: Hi! How can I assist you today?")

# while True:
#     user_input = input("You: ")

#     # Exit condition
#     if user_input.lower() in ["exit", "quit"]:
#         print("Chatbot: Goodbye!")
#         break

#     #chain = conversational_rag_chain.pick("answer")
#     # Stream response from the model
#     print("Chatbot: ", end="", flush=True)
#     for chunk in rag_chain.stream(
#             {"input": [HumanMessage(content=user_input)]},
#             config={"configurable": {"session_id": session_id}}
#     ):
#         print(chunk)
#     print()  # For newline after streaming completes



# conversation = ConversationChain(llm=llm, memory=ConversationSummaryBufferMemory(
#         llm=llm,
#         max_token_limit=650
# ))

# knowledge = PineconeVectorStore.from_existing_index(
#     index_name="dev",
#     embedding=OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
# )

# qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(  
#     llm=llm,  
#     chain_type="stuff",  
#     retriever=knowledge.as_retriever(),
#     memory=ConversationSummaryBufferMemory(
#         llm=llm,
#         max_token_limit=650,
#         input_key='question', 
#         output_key='answer'
# ) 
# )  
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     # memory=ConversationSummaryBufferMemory(
#     #     llm=llm,
#     #     max_token_limit=650
#     # ),
#     memory=ConversationBufferMemory(),
#     retriever=knowledge.as_retriever()
# )

# # Define a few questions about the WonderVector5000.
# query1 = """How can you tell if a hive is ready to swarm?"""

# query2 = """Tell me a cool bee fact."""

# query3 = """Tell me a different cool bee fact."""

# print(qa.invoke(query1).get("result"))
# print(qa.invoke(query2).get("result"))
# print(qa.invoke(query3).get("result"))

# # print(f"Query 1: {query1}\n")
# # print("Chat with knowledge:")
# # print(qa.invoke(query1).get("result"))
# # print("\nChat with knowledge with cite:")
# # print(qa_with_sources(query1))
# # print("\nChat without knowledge:")
# # print(llm.invoke(query1).content)
# # print(f"\nQuery 2: {query2}\n")
# # print("Chat with knowledge:")
# # print(qa.invoke(query2).get("result"))
# # print("\nChat with knowledge with cite:")
# # print(qa_with_sources(query2))
# # print("\nChat without knowledge:")
# # print(llm.invoke(query2).content)

