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
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv
import os  

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')



llm = ChatOpenAI(  
    openai_api_key=os.environ.get("OPENAI_API_KEY"),  
    model_name='gpt-3.5-turbo',  
    temperature=0.0,
    stream_usage=True
)  

knowledge = PineconeVectorStore.from_existing_index(
    index_name=os.getenv('PINECONE_INDEX'),
    embedding=OpenAIEmbeddings(model='text-embedding-3-large', openai_api_key=os.environ.get("OPENAI_API_KEY"))
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

# def get_session_history(session_id): ## for future database memory potential
#     return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
chain = conversational_rag_chain.pick("answer")

# Conversation with history handling
session_id = "abc123"  # Unique session identifier

#Start a conversation
print("Chatbot: Hi! How can I assist you today?")

token_count = 0

while True:
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break


    # Stream response from the model
    print("Chatbot: ", end="", flush=True)  
    with get_openai_callback() as cb:  
        for chunk in chain.stream(
            {"input": user_input},
            config={"configurable": {"session_id": session_id},
        }):
            print(chunk, end="", flush=True)
    print()  # For newline after streaming completes
    # print(f"Total Tokens: {cb.total_tokens}")
    # print(f"Prompt Tokens: {cb.prompt_tokens}")
    # print(f"Completion Tokens: {cb.completion_tokens}")
    # print(f"Total Cost (USD): ${cb.total_cost}")


