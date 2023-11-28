"""Main entrypoint for the app."""
import asyncio
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

import langsmith
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.retrievers import (
    ContextualCompressionRetriever,
    TavilySearchAPIRetriever,
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain.schema.runnable.base import RunnableEach
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langserve import add_routes
from langsmith import Client
from typing_extensions import TypedDict

from dotenv import find_dotenv, load_dotenv

load_dotenv()


REPHRASE_TEMPLATE = """Given the following conversation and a follow up user input, rephrase the follow up input to be a standalone statement containing all the needed information.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

DECOMPOSE_TEMPLATE = """Given the following statement, decompose it into a series of up to 3 simple short questions that are required to answer the statement.

Statement: {statement}
Questions:"""


def get_base_retriever():
    return TavilySearchAPIRetriever(
        k=3,
        include_raw_content=True,
        include_images=False,
        include_domains=["https://airpassengerrights.ca/en/practical-guides"],
    )


# simple retriever that returns the top k results from the search api
retriever = itemgetter("question") | get_base_retriever()
# res = retriever.invoke({"question": "What are my rights as a passenger?"})
# print(res)

# a chain that first rephrases the question and then decomposes it into a series of questions
llm = ChatOpenAI(temperature=0.0)

# condense question chain
condense_question_chain = (
    PromptTemplate.from_template(REPHRASE_TEMPLATE) | llm | StrOutputParser()
)

# decompose question chain
decompose_question_chain = (
    PromptTemplate.from_template(DECOMPOSE_TEMPLATE)
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

# a chain that first rephrases the question and then decomposes it into a series of questions
condense_decompose_chain = (
    RunnablePassthrough()
    | RunnablePassthrough.assign(statement=condense_question_chain)
    | RunnablePassthrough.assign(decomposed_q=decompose_question_chain)
    | RunnablePassthrough.assign(docs=get_base_retriever().map())
)

question = """
URGENT

Dear All,

Seeking your input here

Just landed in Paris and found my checked luggage severely damaged. My Air Canada flight landed at 5:00PM CET(14th March), and I'm left with a broken suitcase. What steps should I take for compensation?
I've been waiting at the baggage claim office for over an hour now.

Much appreciated!
""".strip()

res = condense_decompose_chain.invoke({"chat_history": "", "question": question})

# save the output to a json file
import json

with open("output.json", "w") as f:
    json.dump(res, f, indent=2)
