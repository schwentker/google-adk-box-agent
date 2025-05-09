# File: agent.py

import logging
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

# Import sub-agents
from .sub_agents.box_agent_generic import box_generic_agent_llm
from .sub_agents.box_agent_ai import box_ai_agent_llm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the decision router agent
decision_router = LlmAgent(
    model="gemini-2.0-flash",
    name="DecisionRouter",
    instruction="""
    You are a decision router for Box content queries. Your task is to determine whether a query should be 
    directed to the Box Search agent or the Box Hub agent based on the content of the query.
    
    INSTRUCTIONS:
    
    1. Analyze the user's query to determine its nature.
    
    2. If the query is about:
       - Content stored in Box
       - Search files and folders
       - Read files from Box
       - Check Box connectivity or Identify who am I in box
       => Output "box_generic_agent"
    
    3. If the query is trying to interact with Box AI:
       - Questions about a document content
       - Extracting data from a document
       => Output "box_ai_agent"
    
    4. Your output should ONLY be the exact text "box_generic_agent" or "box_ai_agent" - nothing else.
    
    5. If you're uncertain, default to "box_generic_agent" as it has broader capabilities.
    
    Examples:
    - Query: "Locate my invoices under my procurement folder" -> Output: "box_generic_agent"
    - Query: "Read my file Invoice 123" -> Output: "box_generic_agent"
    - Query: "Ask Box AI to summarize this file" -> Output: "box_ai_agent"
    - Query: "Ask Box AI to extract name, email, contract date from my document ABC123" -> Output: "box_ai_agent"
    """,
    output_key="routing_decision",  # Key for storing the routing decision in session state
)


class BoxFlowAgent(BaseAgent):
    """
    Custom agent for Box content search workflow.

    This agent decides whether to route queries to the Box Search agent
    or the Box Hub agent based on the content of the query.
    """

    # Field declarations for Pydantic
    decision_router: LlmAgent
    box_generic_agent: LlmAgent
    box_ai_agent: LlmAgent

    # Allow arbitrary types for Pydantic
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        decision_router: LlmAgent,
        box_generic_agent: LlmAgent,
        box_ai_agent: LlmAgent,
    ):
        """
        Initializes the BoxFlowAgent.

        Args:
            name: The name of the agent.
            decision_router: An LlmAgent to decide which path to take.
            box_generic_agent: An LlmAgent for Box generic.
            box_ai_agent: An LlmAgent for Box AI interactions.
        """
        # Define the sub_agents list for the framework
        sub_agents_list = [
            decision_router,
            box_generic_agent,
            box_ai_agent,
        ]

        # Pydantic will validate and assign them based on the class annotations
        super().__init__(
            name=name,
            decision_router=decision_router,
            box_generic_agent=box_generic_agent,
            box_ai_agent=box_ai_agent,
            sub_agents=sub_agents_list,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the Box search workflow.
        Uses the decision router to determine which path to take.
        """
        logger.info(f"[{self.name}] Starting Box content search workflow.")

        # 1. Run Decision Router
        logger.info(f"[{self.name}] Running DecisionRouter...")
        async for event in self.decision_router.run_async(ctx):
            logger.info(
                f"[{self.name}] Event from DecisionRouter: {event.model_dump_json(indent=2, exclude_none=True)}"
            )
            yield event

        # 2. Check the routing decision
        routing_decision = ctx.session.state.get("routing_decision")
        logger.info(f"[{self.name}] Routing decision: {routing_decision}")

        if not routing_decision:
            logger.error(
                f"[{self.name}] Failed to make routing decision. Defaulting to Box Search."
            )
            routing_decision = "box_generic_agent"

        # 3. Execute the appropriate agent based on the decision
        if routing_decision == "box_generic_agent":
            logger.info(f"[{self.name}] Running box_generic_agent...")
            async for event in self.box_generic_agent.run_async(ctx):
                logger.info(
                    f"[{self.name}] Event from BoxGenericAgent: {event.model_dump_json(indent=2, exclude_none=True)}"
                )
                yield event
        else:  # Default to box_search
            logger.info(f"[{self.name}] Running box_ai_agent...")
            async for event in self.box_ai_agent.run_async(ctx):
                logger.info(
                    f"[{self.name}] Event from BoxAIAgent: {event.model_dump_json(indent=2, exclude_none=True)}"
                )
                yield event

        logger.info(f"[{self.name}] Workflow finished.")


# Create and export the root_agent
# THIS IS THE CRITICAL LINE FOR THE ADK CLI
root_agent = BoxFlowAgent(
    name="BoxFlowAgent",
    decision_router=decision_router,
    box_generic_agent=box_generic_agent_llm,
    box_ai_agent=box_ai_agent_llm,
)
