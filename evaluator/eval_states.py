from typing import TypedDict, List, Dict, Any, Optional
from typing import Annotated
from langgraph.pregel.io import AddableValuesDict
from langgraph.graph.message import add_messages
from collections import deque

class InteractionMetrics(TypedDict):
    original_intent: str
    total_retries_needed: int
    success_without_retry: bool
    start_time: float
    end_time: Optional[float]

class GraphState(TypedDict):
    experiment: dict
    experiment_config: dict
    experiment_eval: List[Dict[str, Any]]

    evaluator: object
    turns: deque
    max_retries: int
    debug_mode: bool
    actual_number_of_retries: int
    proceed: bool
    go_next_interaction: bool
    interactions_counting: int
    dialogue_agent_config: dict
    
    interaction_metrics: Dict[int, InteractionMetrics]
    current_interaction_start_time: float
    query_complexity: str
    retry_reason: Optional[str]
    
    actual_turn: dict
    last_user_input: str
    last_response: AddableValuesDict
    interaction_history: Annotated[list, add_messages]