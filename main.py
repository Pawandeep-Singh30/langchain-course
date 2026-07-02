from typing import List
from pydantic import BaseModel, Field 
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The url of the source")
class AgentResponse(BaseModel):
    """Schema for agent reponse with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOllama(temperature=0, model="llama3.2")
tools = [TavilySearch()]
agent = create_agent(llm, tools=tools, response_format=AgentResponse)
def main():
    print("Hello from LangChain!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in penang on linked in and list their details"
                )
            }
    )
    # Get just the final answer
    final_message = result["messages"][-1]
    print(final_message.content)
if __name__ == "__main__":
    main()
