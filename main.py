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
from langchain.output_parsers import NumberedListOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langserve import add_routes
from langsmith import Client
from typing_extensions import TypedDict

from dotenv import find_dotenv, load_dotenv

load_dotenv()

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


DECOMPOSITION_TASK_DESCRIPTION = """Dissect the query into simple, singular sub-questions. Each sub-question should be answerable on its own, focusing on a single aspect of the original input. Limit the number of sub-questions to 3.

Input: {question}
Sub-Questions:"""

client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatRequest(TypedDict):
    question: str
    chat_history: Optional[List[Dict[str, str]]]
    # conversation_id: Optional[str]


def get_base_retriever():
    return TavilySearchAPIRetriever(
        k=5,
        include_raw_content=True,
        include_images=False,
        include_domains=["https://airpassengerrights.ca/en/practical-guides"],
    )


def _get_retriever():
    embeddings = OpenAIEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)

    relevance_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.85,
        # k=2,
    )

    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, relevance_filter]
    )
    base_retriever = get_base_retriever()
    return ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )


def create_retriever_chain(
    llm: BaseLanguageModel, retriever: BaseRetriever
) -> Runnable:
    condense_question_chain = (
        # itemgetter("question")
        PromptTemplate.from_template(REPHRASE_TEMPLATE)
        | llm
        | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )

    DECOMPOSITION_PROMPT = PromptTemplate.from_template(DECOMPOSITION_TASK_DESCRIPTION)
    decompose_question_chain = DECOMPOSITION_PROMPT | llm | NumberedListOutputParser()

    def add_queries_to_docs(doc_collection: Sequence[Document], queries: Sequence[str]):
        for collection_idx in range(len(doc_collection)):
            for doc_idx in range(len(doc_collection[collection_idx])):
                doc_collection[collection_idx][doc_idx].metadata["query"] = queries[
                    collection_idx
                ]
        return doc_collection

    retriever_chain = (
        RunnablePassthrough()
        | RunnablePassthrough.assign(question=condense_question_chain)
        | RunnablePassthrough.assign(queries=decompose_question_chain)
        | RunnablePassthrough.assign(docs=itemgetter("queries") | retriever.map())
        | RunnableLambda(
            lambda x: add_queries_to_docs(x["docs"], x["queries"])
        ).with_config(run_name="QueryDocsMapping")
    )

    return retriever_chain
    # return RunnableBranch(
    #     (
    #         RunnableLambda(lambda x: bool(x.get("chat_history"))),
    #         (RunnablePassthrough.assign(question=condense_question_chain) | retriever_chain),
    #     ),
    #     retriever_chain,  # default branch
    # ).with_config(run_name="RouteDependingOnChatHistory")


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


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    # model="gpt-4",
    # model="gpt-4-1106-preview",
    streaming=True,
    temperature=0,
)

# retriever = _get_retriever()
retriever = get_base_retriever()
chain = create_retriever_chain(llm, retriever)
add_routes(app, chain, path="/api/chat", input_type=ChatRequest)


#################
# Feedback routes
#################


@app.post("/api/feedback")
async def send_feedback(request: Request):
    data = await request.json()
    run_id = data.get("run_id")
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    key = data.get("key", "user_score")
    vals = {**data, "key": key}
    client.create_feedback(**vals)
    return {"result": "posted feedback successfully", "code": 200}


@app.patch("/api/feedback")
async def update_feedback(request: Request):
    data = await request.json()
    feedback_id = data.get("feedback_id")
    if feedback_id is None:
        return {
            "result": "No feedback ID provided",
            "code": 400,
        }
    client.update_feedback(
        feedback_id,
        score=data.get("score"),
        comment=data.get("comment"),
    )
    return {"result": "patched feedback successfully", "code": 200}


################
# Test routes
################


# Test route that returns a simple response
@app.post("/api/test")
def test():
    return {"result": "test successful"}


# Test route that returns a simple response from a language model
test_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
    ]
)
add_routes(app, test_prompt | llm, path="/api/test_llm")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
