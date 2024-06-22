# QA Bot with LlamaIndex
import os
import streamlit as st

try:
    from llama_index.legacy import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
    from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

# --Config
st.set_page_config(
    page_title="Chat with the streamlit docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# --OPENAI_API_KEY
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

## --LLM
import openai
from llama_index.llms.openai import OpenAI

system_prompt = """
You are an expert in Python, Stremlit python library, LlamaIndex, AI based agentic workflows. Your job is to answer technical questions. Assume that all questions are related to the streamlit python library. keep your answers technical and based on the facts - DO NOT Hallucinate features.
"""
llm_llamaindex = OpenAI(
    model="gpt-3.5-turbo",
    temperature=.0,
    api_key=OPENAI_API_KEY,
    system_prompt=system_prompt
)

# --HEADER
# [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)
st.title("💬Chat: Streamlit w/ LlamaIndex🦙")

# initiate chat session
if "messages" not in st.session_state.keys():
    st.session_state.messages=[
        {
            "role":"assistant",
            "content":"Hi, I am a bot with expertise in Streamlit and LlamaIndex"
        }
    ]

# load data
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner():
        reader = SimpleDirectoryReader(
            input_dir="/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_AI/dev_proj_AI/pract-llama-rag/ref_llama_streamlit/data",
            recursive=True
        )
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=llm_llamaindex)
        # index = VectorStoreIndex.from_documents(docs)
        index = VectorStoreIndex.from_documents(
            documents=docs, # List of documents to build the index from
            service_context=service_context,
        )
        return index

index = load_data()

# initiate chat engine (create a retriever as chat engine, includes an llm)
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# get user prompt
if prompt := st.chat_input("your question"):
    st.session_state.messages.append(
        {
            "role":"user",
            "content":prompt,
        }
    )

# display prior messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# # use the user prompt to get context and llm response from retriever
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = st.session_state.chat_engine.chat(prompt)
#             st.write(response.response)
#             message = {
#                 "role":"assistant",
#                 "content":response.response,
#             }
#             st.session_state.messages.append(message)


# -- show session messages
st.write(st.session_state)
