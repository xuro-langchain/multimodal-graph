import base64
import asyncio
import httpx
import dotenv
import uuid
from typing_extensions import Literal, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.types import Send
from tavily import AsyncTavilyClient

dotenv.load_dotenv()

tavily_client = AsyncTavilyClient()

# Initialize a multimodal chat model that supports PDFs
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def append_or_reset(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    """Reducer for messages that allows resetting when right is None or empty"""
    if right is None or len(right) == 0:
        return []  # Reset to empty list
    if left is None:
        return right
    return left + right

class State(TypedDict):
    question: str
    pdf_url: str
    answer: str
    queries: list[str]
    context: Annotated[list[str], append_or_reset]
    messages: Annotated[list[AnyMessage], append_or_reset]

class Input(TypedDict):
    question: str
    pdf_url: str

class Output(TypedDict):
    answer: str


class Query(BaseModel):
    """Always use this tool to structure your response to the user."""
    queries: list[str] = Field(description="A list of Google search queries")

async def research_terminology(state: State) -> State:
    question = state["question"]
    prompt = f"""You are an expert web researcher, specializing in financial and business terminology. 
    You will be given a question. Please generate a list of Google search queries that will explain what data from a 10-K filing is needed to answer the question.
    These queries should pull any formulas, accounting terms, or other financial terminology that is needed to answer the question.

    Please return 5 queries maximum. 
    Make sure queries are diverse and do not ask for the same information.
    """
    structured_llm = llm.with_structured_output(Query)
    response = await structured_llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=question)])
    return {"queries": response.queries, "messages": [HumanMessage(content=question)]}

def start_parallel_search(state: State) -> Literal["search_web"]:
    return [Send("search_web", {"query": query}) for query in state["queries"]]

async def search_web(state: State) -> State:
    query = state["query"]
    search_results = await tavily_client.search(query)
    context = "\n".join([result["content"] for result in search_results["results"]])
    return {"context": [context]}

async def process_pdf(state: State) -> State:
    question = state["question"]
    pdf_url = state["pdf_url"]
    compiled_context = "\n".join(state["context"])

    system_prompt = f"""
    You are an expert financial analyst. You will be given a question and a pdf.
    Use the pdf to answer the question. 
    You also have available to you context on terminology related to the question.
    This will contain useful information on formulas, terminology, and necessary information to answer the question.

    Please include the calculations used to answer the question in your response.
    
    <question>
    {question}
    </question>

    <context>
    {compiled_context}
    </context>
    """

    async with httpx.AsyncClient() as client:
        response = await client.get(pdf_url)
        pdf_bytes = response.content
        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # Create multimodal message with PDF file content block
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": "Here's the 10-k filing you should analyze."},
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": pdf_b64,
                    "mime_type": "application/pdf",
                    "filename": "10k.pdf",
                },
            ]
        )

        # Invoke the model
        response = await llm.ainvoke([SystemMessage(content=system_prompt), human_message])
    return {"answer": response.content, "messages": [response]}


# Define the graph
def make_graph():
    graph = StateGraph(State, input_schema=Input, output_schema=Output)

    graph.add_node("research_terminology", research_terminology)
    graph.add_node("search_web", search_web)
    graph.add_node("process_pdf", process_pdf)

    graph.add_conditional_edges("research_terminology", start_parallel_search)

    graph.add_edge(START, "research_terminology")
    graph.add_edge("search_web", "process_pdf")
    graph.add_edge("process_pdf", END)

    return graph.compile()


def print_messages(response):
    if isinstance(response, dict):
        print("STATE UPDATE ----------------")
        for key in response:
            if key == "context":
                for context in response["context"]:
                    print("context: " + context[:100] + "...\n\n")
            elif key == "messages":
                continue
            else:
                print(key + ": " + str(response[key]))
        print("\n")

async def run_graph(question: str, pdf_url: str):
    graph = make_graph()
    turn_input = {"question": question, "pdf_url": pdf_url, "context": [], "queries": []}
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    async for output in graph.astream(turn_input, config, stream_mode="updates"):
        if END in output or START in output:
            continue
        # Print any node outputs
        for key, value in output.items():
            print_messages(value)

if __name__ == "__main__":
    question = "Calculate the EBITDA for this company"
    pdf_url = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/c87043b9-5d89-4717-9f49-c4f9663d0061.pdf"
    asyncio.run(run_graph(question, pdf_url))