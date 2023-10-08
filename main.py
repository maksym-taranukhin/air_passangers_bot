"""Main entrypoint for the app."""
from fastapi.middleware.cors import CORSMiddleware
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.retrievers import (ContextualCompressionRetriever,
                                  TavilySearchAPIRetriever)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline, EmbeddingsFilter)
from langchain.schema import Document
from langchain.schema.document import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (Runnable, RunnableBranch,
                                       RunnableLambda, RunnableMap)
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Backup
from langchain.utilities import GoogleSearchAPIWrapper
from langserve import add_routes
from src.chat_types import ChatRequest
from langchain.chat_models import ChatOpenAI
from src.utils import _get_retriever
from src.chains import create_chain
from src.routs import router
from src.constants import app


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


class BackupRetriever(BaseRetriever):
    search: GoogleSearchAPIWrapper = GoogleSearchAPIWrapper()
    num_search_results = 6

    def clean_search_query(self, query: str) -> str:
        # Some search tools (e.g., Google) will
        # fail to return results if query has a
        # leading digit: 1. "LangCh..."
        # Check if the first character is a digit
        if query[0].isdigit():
            # Find the position of the first quote
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                # Extract the part of the string after the quote
                query = query[first_quote_pos + 1 :]
                # Remove the trailing quote if present
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()

    def search_tool(self, query: str, num_search_results: int = 1) -> List[dict]:
        """Returns num_search_results pages per Google search."""
        query_clean = self.clean_search_query(query)
        result = self.search.results(query_clean, num_search_results)
        return result

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ):
        # Get search questions
        print("Generating questions for Google Search ...")

        # Get urls
        print("Searching for relevant urls...")
        urls_to_look = []
        search_results = self.search_tool(query, self.num_search_results)
        print("Searching for relevant urls...")
        print(f"Search results: {search_results}")
        for res in search_results:
            if res.get("link", None):
                urls_to_look.append(res["link"])

        print(search_results)
        loader = AsyncHtmlLoader(urls_to_look)
        html2text = Html2TextTransformer()
        print("Indexing new urls...")
        docs = loader.load()
        docs = list(html2text.transform_documents(docs))
        for i in range(len(docs)):
            if search_results[i].get("title", None):
                docs[i].metadata["title"] = search_results[i]["title"]
        return docs


def get_base_retriever():
    # return TavilySearchAPIRetriever(k=6, include_raw_content=True, include_images=True)
    return BackupRetriever()


def _get_retriever():
    embeddings = OpenAIEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    relevance_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.8)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, relevance_filter]
    )
    base_retriever = get_base_retriever()
    return ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    ).with_config(run_name="FinalSourceRetriever")


def create_retriever_chain(
    llm: BaseLanguageModel, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


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


def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
) -> Runnable:
    retriever_chain = create_retriever_chain(llm, retriever) | RunnableLambda(
        format_docs
    ).with_config(run_name="FormatDocumentChunks")
    _context = RunnableMap(
        {
            "context": retriever_chain.with_config(run_name="RetrievalChain"),
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(itemgetter("chat_history")).with_config(
                run_name="Itemgetter:chat_history"
            ),
        }
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return (
        {
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(serialize_history).with_config(
                run_name="SerializeHistory"
            ),
        }
        | _context
        | response_synthesizer
    )


llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    # model="gpt-4",
    streaming=True,
    temperature=0,
)

retriever = _get_retriever()

chain = create_chain(llm, retriever)

add_routes(app, chain, path="/chat", input_type=ChatRequest)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
