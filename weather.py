
# =============================================================================
# WEATHER AGENT — LangChain 1.x + Ollama
# =============================================================================
# This app uses LangChain's create_agent() — a built-in shortcut that wraps
# a LangGraph loop internally: LLM thinks → calls tools → LLM answers.
#
# Tools available to the agent:
#   - get_weather      → one city, today or tomorrow
#   - compare_weather  → two cities, today + tomorrow (4 API calls in one shot)
#   - get_time         → current time in any IANA timezone
# =============================================================================
from dotenv import load_dotenv
load_dotenv()  # Loads LANGSMITH_* from .env so runs appear in LangSmith
import json
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

MODEL = "llama3.1:8b"

# --- WEATHER API HELPERS (plain Python, not LangChain) -----------------------
# These fetch real data from wttr.in. The agent never calls them directly —
# only the @tool functions below are visible to the LLM.

def _fetch_weather_data(city: str) -> dict:
    """HTTP GET to wttr.in; returns raw JSON weather payload."""
    encoded = urllib.parse.quote(city.strip())
    url = f"https://wttr.in/{encoded}?format=j1"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "langchain-course/1.0"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))

def _place_name(area: dict) -> str:
    """Turn wttr.in location fields into a readable place string."""
    place = area["areaName"][0]["value"]
    region = area["region"][0]["value"]
    country = area["country"][0]["value"]
    return f"{place}, {region}, {country}"

def _format_weather(city: str, day: str = "today") -> str:
    """Shared weather formatter used by get_weather and compare_weather.
    Not a @tool — internal helper so we don't duplicate wttr.in parsing logic.
    """
    data = _fetch_weather_data(city)
    area = data["nearest_area"][0]
    location = _place_name(area)
    if day.lower() in ("tomorrow", "next day"):
        forecast = data["weather"][1]
        max_temp = forecast["maxtempC"]
        min_temp = forecast["mintempC"]
        desc = forecast["hourly"][4]["weatherDesc"][0]["value"]
        return (
            f"{location} tomorrow: {min_temp}-{max_temp}°C, "
            f"expected {desc.lower()}"
        )
    current = data["current_condition"][0]
    temp_c = current["temp_C"]
    desc = current["weatherDesc"][0]["value"]
    humidity = current["humidity"]
    return f"{location}: {temp_c}°C, {desc}, humidity {humidity}%"

# --- LANGCHAIN TOOLS -----------------------------------------------------------
# @tool registers a function so the LLM can call it by name.
# The docstring is shown to the model — write it like instructions for WHEN to use.

@tool
def get_weather(city: str, day: str = "today") -> str:
    """Get live weather or forecast for a single city.
    Use for one city at a time. For comparing two cities, use compare_weather instead.
    Args:
        city: City name, e.g. 'Taiping, Perak, Malaysia'
        day: 'today' for current weather, 'tomorrow' for next-day forecast
    """
    try:
        return _format_weather(city, day)
    except Exception as e:
        return f"Could not fetch weather for {city}: {e}"

@tool
def compare_weather(city_a: str, city_b: str) -> str:
    """Compare weather between two cities (today AND tomorrow for both).
    Always use this when the user asks to compare weather between two places.
    Fetches all 4 data points in one call — more reliable than multiple get_weather calls.
    Args:
        city_a: First city, e.g. 'Taiping, Malaysia'
        city_b: Second city, e.g. 'Kuala Lumpur, Malaysia'
    """
    try:
        lines = [
            "=== TODAY ===",
            f"A) {_format_weather(city_a, 'today')}",
            f"B) {_format_weather(city_b, 'today')}",
            "",
            "=== TOMORROW ===",
            f"A) {_format_weather(city_a, 'tomorrow')}",
            f"B) {_format_weather(city_b, 'tomorrow')}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Could not compare {city_a} and {city_b}: {e}"

def _resolve_timezone(location: str) -> tuple[str, str]:
    """Resolve a city name or IANA timezone to (display_name, iana_timezone)."""
    location = location.strip()
    if "/" in location:
        return location, location

    query = urllib.parse.quote(location)
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "langchain-course/1.0"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))

    results = data.get("results") or []
    if not results:
        raise ValueError(f"Could not find timezone for '{location}'")

    place = results[0]
    name = place.get("name", location)
    country = place.get("country", "")
    timezone = place["timezone"]
    display = f"{name}, {country}" if country else name
    return display, timezone

@tool
def get_time(location: str) -> str:
    """Get the current local time for a city or IANA timezone.
    Use when the user asks what time it is somewhere — NOT for weather questions.
    Args:
        location: City name like 'Amritsar, India' or IANA timezone like 'Asia/Kuala_Lumpur'
    """
    try:
        display, timezone = _resolve_timezone(location)
        now = datetime.now(ZoneInfo(timezone))
        formatted = now.strftime("%I:%M %p, %A %d %B %Y")
        return f"Current time in {display}: {formatted}"
    except Exception as e:
        return (
            f"Could not get time for {location}: {e}. "
            "Try a city like 'Amritsar, India' or a timezone like 'Asia/Kuala_Lumpur'."
        )

# --- AGENT SETUP (LangChain) --------------------------------------------------
# create_agent() builds a LangGraph under the hood:
#   START → model → (tool calls?) → tools → model → END
#
# The LLM reads each tool's docstring + system_prompt to decide which tool to call.

def build_agent():
    llm = ChatOllama(model=MODEL, temperature=0)
    return create_agent(
        llm,
        # Pass all tools — the model picks the right one per question
        tools=[get_weather, compare_weather, get_time],
        system_prompt=(
            "You are a helpful weather and time assistant. "
            "Always call the appropriate tool instead of guessing. "
            "Use get_weather for a single city's weather. "
            "Use compare_weather when comparing TWO cities — always prefer it for comparisons. "
            "Use get_time for time questions with the location argument "
            "(city like 'Amritsar, India' or timezone like 'Asia/Kuala_Lumpur'). "
            "Never invent data. Use full city names like 'Kuala Lumpur, Malaysia'. "
            "Answer directly without preamble."
        ),
    )

# --- CHAT LOOP ----------------------------------------------------------------
# Keeps a message history so follow-ups like 'what about tomorrow?' work.

def main():
    agent = build_agent()
    print("Weather agent (type 'quit' to exit)\n")
    print("Try: 'weather in Taiping', 'compare Taiping and Kuala Lumpur',")
    print("     'time in Amritsar, India', 'what time is it in Asia/Kuala_Lumpur'\n")
    history = []
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        history.append(HumanMessage(content=question))
        # agent.invoke runs the full LangGraph loop for this turn
        result = agent.invoke({"messages": history})
        # Keep ALL messages (including tool calls/results), not just the answer.
        # Dropping tool messages breaks memory on follow-up questions.
        history = result["messages"]
        print(f"Agent: {history[-1].content}\n")

if __name__ == "__main__":
    main()

# =============================================================================
# IDEAS FOR WHAT TO BUILD NEXT
# =============================================================================
#
# 1. STREAMING — print tokens as they arrive:
#      for chunk in agent.stream({"messages": history}, stream_mode="messages"):
#          ...
#
# 2. RAG — load a PDF/text file, embed chunks, retrieve relevant parts, then ask:
#      pip packages: langchain-community, langchain-chroma
#
# 3. MANUAL LANGGRAPH — replace create_agent with your own StateGraph when you
#      need custom routing (e.g. always send "compare" to compare_weather node).
#
# 4. LANGSMITH EVALS — create a dataset of test questions and auto-score answers
#      that contain "Not specified" or missing city names.
# =============================================================================
