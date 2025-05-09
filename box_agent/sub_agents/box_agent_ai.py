import logging  # Optional: for better logging
from google.adk.agents import LlmAgent  # Use LlmAgent
from ..tools.box_agent_tools import (
    box_ask_ai_tool,
    box_ai_extract_data,
)

# Setup logging (optional but recommended)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

box_ai_agent = LlmAgent(
    model="gemini-2.0-flash",
    # model="gemini-2.5-pro",
    name="box_ai_agent",
    description="""
    You are a helpful assistant designed to interact with Box content using specialized tools.
    Your primary goal is to answer user questions using Box AI.
    """,
    tools=[
        box_ask_ai_tool,
        box_ai_extract_data,
    ],
)
