from typing import Sequence
import asyncio
import os

import langsmith
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import (ContextualCompressionRetriever,
                                  TavilySearchAPIRetriever)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline, EmbeddingsFilter)
from langchain.schema import Document
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langserve import add_routes
from langsmith import Client
from src.chat_types import ChatRequest
from langchain.vectorstores import Weaviate
from langchain.schema.retriever import BaseRetriever
import weaviate

client = Client()
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


# def get_base_retriever():
    # return TavilySearchAPIRetriever(k=6, include_raw_content=True, include_images=True)

def get_retriever() -> BaseRetriever:
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    weaviate_client = Weaviate(
        client=weaviate_client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=OpenAIEmbeddings(chunk_size=200),
        by_text=False,
        attributes=["source", "title"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=6))


def _get_retriever():
    # embeddings = OpenAIEmbeddings()
    # splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    # relevance_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.8)
    # pipeline_compressor = DocumentCompressorPipeline(
    #     transformers=[splitter, relevance_filter]
    # )
    base_retriever = get_base_retriever()
    return ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    ).with_config(run_name="FinalSourceRetriever")


# TODO: Update when async API is available
async def _arun(func, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, func, *args, **kwargs)


async def aget_trace_url(run_id: str) -> str:
    for i in range(5):
        try:
            await _arun(client.read_run, run_id)
            break
        except langsmith.utils.LangSmithError:
            await asyncio.sleep(1**i)

    if await _arun(client.run_is_shared, run_id):
        return await _arun(client.read_run_shared_link, run_id)
    return await _arun(client.share_run, run_id)