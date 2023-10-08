"""Main entrypoint for the app."""
from fastapi.middleware.cors import CORSMiddleware
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
