import logging  # Optional: for better logging
from google.adk.agents import LlmAgent  # Use LlmAgent
from ..tools.box_agent_tools import (
    box_who_am_i,
    box_search_tool,
    box_read_tool,
    box_search_folder_by_name,
    box_list_folder_content_by_folder_id,
)

# Setup logging (optional but recommended)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

box_generic_agent = LlmAgent(
    model="gemini-2.0-flash",
    # model="gemini-2.5-pro",
    name="box_generic_agent",
    description="""
    You are a helpful assistant designed to interact with Box content using specialized tools.
    Your primary goal is to answer user questions about documents stored in Box.
    """,
    tools=[
        box_who_am_i,
        box_search_tool,
        box_read_tool,
        box_search_folder_by_name,
        box_list_folder_content_by_folder_id,
    ],
)
