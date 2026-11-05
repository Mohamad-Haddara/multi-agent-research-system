
import operator
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from typing_extensions import TypedDict, Annotated
from langchain.agents import create_agent

from typing import List, Optional, Literal
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

from dotenv import load_dotenv
import json
import asyncio

import os 
from tavily import AsyncTavilyClient, TavilyClient
from langchain_tavily import TavilySearch, tavily_search

from langgraph.prebuilt import ToolNode , tools_condition

load_dotenv()



# 1) Create state: The data moves betweent the agents (graph/state)
# Create an Actual state
class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # agent conversation will be saves
    topic: str # What to search
    search_queries: List[str]
    findings: Annotated[List[dict], operator.add] # finding should have reducer, when 2 search agents run in parrallel
    report: str
    analysis: str
    quality_score: float
    quality_feedback: str
    iteration: int


# 2) Define 2 llm (normal llm and creative LLM)
llm = ChatOpenAI(model="gpt-4o-mini", temperature= 0)
creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 3) Define Prompt (Start with Supervisor Prompt)
SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are research supervisor. Given a topic, generate 3 specific search queries that will cover different angle of the topic. Return ONLY JSON array of strings. No markdown formatting."),
    # Keep a topic as a placeholder and not as f-string
    ("human", "Research topic {topic}")
])

# # Schema: what the supervisor must return
# class SearchPlan(BaseModel):
#     queries: List[str] = Field(description="3 specific web search queries, each covering a different angle of the topic")

# One agent at a time
# shcema.py: What the agent must return
# prompt.py: Its instruction
# node.py: takes state, calls the LLM, check the output, returns an update
# test it alone

# Wiring: add the nodes and edges (builder.py)

# Build graph then ainvoke, astream

# Debug: watch step, then use Langfuse



# Create Supervisor Agent
async def supervisor(state: ResearchState) -> dict:
    """
    Plan the research: Turn the topic into search queries
    """
    # Format messages to take the keyword arguments, not dic
    messages = SUPERVISOR_PROMPT.format_messages(topic = state['topic'])
    response = await llm.ainvoke(messages)
    
    print("="*60)
    print(f"Response: {response.content}")
    queries = json.loads(response.content)


    print("="*60)
    print(f"Queries = {queries}")

    return {
        "search_queries": queries[:3]
    }



# Instantiate Tavily search tool - To make complete search queries using Tavily Search API endpoint


MAX_RESULTS = 5  # Maximum number of search results to return 
MAX_CHARS = 1500
SEARCH_DEPTH = "advanced" # Depth of the search ~ basic or advanced
TIMEOUT_S = 20
MAX_SEARCHES = 4

# Initialize the Tavily Search tool
# tavily_search = TavilySearch(
#     max_results = MAX_RESULTS,
#     search_depth = SEARCH_DEPTH,
# )

# Invoke directly with args
# The Tavily search tool accepts the following arguments
# 1) query: search query
# 2) set during invocation: include_images,... . 


# Initialize the agent with the search tool
# agent = create_agent(
#     model=ChatOpenAI(model="gpt-4o-mini"),
#     tools=[tavily_search],
#     system_prompt="You are helpful research assistant. Use web search to find accurate, up-to-date information."

# )


# # use the agent
# response = agent.invoke({
#     "messages":[
#         {"role": "user", "content": "What is the most popular sport in the world? Include only Wikipedia source."}
#     ]
# })

# print("Response:\n")
# print(response)


# ===========================================================

#A) Initialize the objects
_client = TavilyClient()
_aclient = AsyncTavilyClient()


# B) Defiene the schema for function calling
class WebSearchInput(BaseModel):
    """
    The schema the model sees (converted to JSON schema automatically)
    """

    query: str = Field(description="A search query")
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None, description="Only return results from this period."
    )


def _params(query: str, time_range: Optional[str]) -> dict:
    return dict(query = query, time_range = time_range, search_depth = SEARCH_DEPTH, max_results = MAX_RESULTS, timeout = TIMEOUT_S)


 
def _format(raw: dict) -> str:
    # Keep only what downstream agents need: title + url (for citations) + trimmed content
    results = [
        {"title": r.get("title"), "url": r.get("url"), "content": (r.get("content") or "")[:MAX_CHARS]}
        for r in raw.get("results", [])
    ]
    if not results:
        return json.dumps({"results": [], "note": "No results. Try a broader or rephrased query."})
    return json.dumps({"results": results}, ensure_ascii=False)
 
 
def _error(e: Exception) -> str:
    # Return the error to the model instead of crashing the graph; it can retry or move on
    return json.dumps({"error": f"search failed ({type(e).__name__}): {e}"})
 
 
def _search(query: str, time_range: Optional[str] = None) -> str:
    try:
        return _format(_client.search(**_params(query, time_range)))
    except Exception as e:
        return _error(e)
 
 
async def _asearch(query: str, time_range: Optional[str] = None) -> str:
    try:
        return _format(await _aclient.search(**_params(query, time_range)))
    except Exception as e:
        return _error(e)
 
 
web_search = StructuredTool.from_function(
    func=_search,
    coroutine=_asearch,  # used when the graph runs with ainvoke (FastAPI)
    name="web_search",
    description=(
        "Search the live web. Returns JSON {results: [{title, url, content}]}. "
        "Use focused queries; if results are weak, refine the query and search again. "
        "Cite the url of every fact you use."
    ),
    args_schema=WebSearchInput,
)

# LangGraph: one search agent
# START -> agent -> (tool call?) -> tools -> agents
# noo tool call -> END

SEARCH_AGENT_PROMPT = """
You are a web research agent. Answer the sub-question you are given using web_search.
    - Run 2-4 focused searches. Stop once you have evidence.
    - Return findings as short bullet points, each with its source URL
    - Say clearly when sources disagree or evidence is thin.
    - Never put personal, client or internal data in a search query.

"""

def build_search_agent(llm: BaseChatModel, max_searches: int = MAX_SEARCHES):

    # Add all tools as a list
    tools = [web_search]
    llm_can_search = llm.bind_tools(tools)



    def search_agent(state: ResearchState) -> dict:
        """
        This will will search about the queries that is related to the topic.
        """
        # Generate response using the LLM with tool access
        
        response = llm_can_search.invoke([SystemMessage(SEARCH_AGENT_PROMPT), *state["messages"]])

        return {
            "messages": [response]
        }
        

    graph = StateGraph(MessagesState)
    graph.add_node("agent", search_agent)
    graph.add_node("tools", ToolNode([web_search]))       # runs the tool calls the model made
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool call -> "tools", else END
    graph.add_edge("tools", "agent")
    return graph.compile(name="search_agent")


 
search_graph = build_search_agent(llm )

# NEW: bridge between the search subgraph and ResearchState.
# In builder.py, fan out one search agent per query:
#   graph.add_conditional_edges("supervisor", lambda s: [Send("search", {"query": q}) for q in s["search_queries"]])
class SearchTask(TypedDict):
    query: str
 
 
async def search_node(state: SearchTask) -> dict:
    result = await search_graph.ainvoke({"messages": [HumanMessage(state["query"])]})
    # Each parallel run returns a one-item list; operator.add on `findings` merges them
    return {"findings": [{"query": state["query"], "summary": result["messages"][-1].content}]}
 
 
 
 
async def main():
    update = await supervisor({"topic": "Solar vs wind energy in the GCC"})
    print("\nUpdate:")
    print(update)
 
    # Test one search agent alone on the first query (every search costs Tavily credits)
    found = await search_node({"query": update["search_queries"][0]})
    print("\nFindings:")
    print(found["findings"][0]["summary"])
 
 
 
 
if __name__ == "__main__":
    asyncio.run(main())