import logging  # Optional: for better logging
from google.adk.agents import LlmAgent  # Use LlmAgent
from ..tools.box_agent_tools import (
    box_ask_ai_tool,
    box_ai_extract_data,
    # box_who_am_i_tool,
    # box_search_tool,
    # box_read_tool,
    # box_search_folder_by_name,
    # box_list_folder_content_by_folder_id,
)

# Setup logging (optional but recommended)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

box_ai_agent_llm = LlmAgent(
    model="gemini-2.0-flash",
    # model="gemini-2.5-pro",
    name="box_ai_agent",
    description="""
    
    Your primary goal is to answer user questions using Box AI.
    """,
    tools=[
        box_ask_ai_tool,
        box_ai_extract_data,
        # box_who_am_i_tool,
        # box_search_tool,
        # box_read_tool,
        # box_search_folder_by_name,
        # box_list_folder_content_by_folder_id,
    ],
)
