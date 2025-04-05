import evaluator.eval_prompts as prompts
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from models_utils.llm_config import LLMConfig
from evaluator.eval_states import GraphState
from typing import Optional

GPT4O = LLMConfig(provider="azure").get_llm(model="gpt-4o")
O3MINI_LOW_REASONING = LLMConfig(provider="azure").get_llm(model="o3-mini", reasoning_effort="low")

def calculate_tables_recall(tables_ground_truth, tables_from_schema_linking, debug_mode=False):
        """
        Calculate the recall of the schema linking.
        """

        # True Positives / False negatives + True Positives = Recall
        TP, FN = 0, 0

        # Verify if tables_from_schema_linking is a string and try to convert it to a list
        if isinstance(tables_from_schema_linking, str):
            try:
                import ast
                tables_from_schema_linking = ast.literal_eval(tables_from_schema_linking)
            except (ValueError, SyntaxError):
                # If the conversion fails, treat it as a list with a single item
                tables_from_schema_linking = [tables_from_schema_linking]
        
        # If the conversion fails, treat it as a list with a single item
        if not tables_from_schema_linking:
            tables_from_schema_linking = []

        # normalizing tables names for lowercase
        tables_ground_truth = [table.replace("MONDIAL_", "").lower() for table in tables_ground_truth]
        tables_from_schema_linking = [table.replace("MONDIAL_", "").lower() for table in tables_from_schema_linking]

        for table in tables_ground_truth:
            if table in tables_from_schema_linking:
                TP += 1
            else:
                FN += 1

        if debug_mode: print(f"[Schema Linking Recall calculus]\n Ground Truths: {tables_ground_truth}\n Tables from Schema Linking: {tables_from_schema_linking}\n Recall = True Positives / (False Negatives + True Positives) = {TP} / ({FN} + {TP}) = {TP / (FN + TP)}.")

        return TP / (FN + TP)

def messages_to_string_list(messages):
    string = ""
    for msg in messages:
        # Get the class name (ex: "HumanMessage", "AIMessage", "ToolMessage")
        msg_type = msg.__class__.__name__
        # Create a string combining the type and the message content
        string += f"{msg_type}: {msg.content}" + "\n"

    return string

def compare_intentions(function_input, intention, chat_history):
    # Check if the function input is aligned with the intentions
    prompt = prompts.AI_JUDGE_INTENTION_PROMPT.format(
        function_input=function_input,
        ground_truth=intention,
        chat_history=messages_to_string_list(chat_history)
    )

    print(
        f"[AI as JUDGE] Comparing intention between queries '{function_input}' and '{intention}' using AI as Judge method.")
    input = HumanMessage(content=prompt)

    chain = GPT4O | StrOutputParser()

    result = chain.invoke([input])
    print(f"[AI as JUDGE] Result: {result}.")

    return result.strip().lower() == "true"

def extract_outer_json(text):
    """
    Utility function to avoid cases when llm answers like:
    "No results found.
    {
        "input": "Query.",
        "schema_linking": [],
        "answer": "No results found.",
        "sql": ""
    }"
    In benchmark it was a case of hallucination, this function mitigates this problem.
    """
    start_index = None
    brace_count = 0
    for i, char in enumerate(text):
        if char == '{':
            if start_index is None:
                start_index = i  # marks the start of the JSON
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_index is not None:
                # Found the closing of the outer object
                end_index = i
                return text[start_index:end_index + 1]
    return None

def need_feedback(chat_history):
    prompt = prompts.FEEDBACK_CLASSIFICATION_PROMPT.format(chat_history=convert_story_to_string(chat_history))

    print(f"[AI as JUDGE] Judging if a feedback is needed for the last message in chat history.")
    msg = HumanMessage(content=prompt)

    chain = GPT4O | StrOutputParser()

    result = chain.invoke([msg])
    print(f"[AI as JUDGE] Result: {result}.")

    return result.strip().lower() == "true"

def convert_story_to_string(chat_history):
    return "\n".join([msg.content for msg in chat_history])

def append_turn_evaluation(state: GraphState, turn_eval: dict) -> None:
    """
    Group the turn in the current interaction.
    If there is already an interaction with the current interaction_id, add the turn (with relative turn_id);
    otherwise, create a new interaction with the collected metrics.
    """
    current_interaction_id = state["interactions_counting"]
    found = None
    for interaction in state["experiment_eval"]:
        if interaction["interaction_id"] == current_interaction_id:
            found = interaction
            break
            
    if found:
        turn_eval["turn_id"] = len(found["turns"]) + 1
        found["turns"].append(turn_eval)
        
        # Update interaction metrics if necessary
        metrics = state["interaction_metrics"].get(current_interaction_id, {})
        found["total_retries_needed"] = metrics.get("total_retries_needed", 0)
        found["success_without_retry"] = metrics.get("success_without_retry", True)
        
        # If the interaction was finished, calculate the total time
        if metrics.get("end_time"):
            found["execution_time"] = metrics.get("end_time") - metrics.get("start_time", 0)
    else:
        turn_eval["turn_id"] = 1
        
        # Get the metrics of the current interaction
        metrics = state["interaction_metrics"].get(current_interaction_id, {})
        
        new_interaction = {
            "interaction_id": current_interaction_id,
            "original_intent": metrics.get("original_intent", "Unknown"),
            "total_retries_needed": metrics.get("total_retries_needed", 0),
            "success_without_retry": metrics.get("success_without_retry", True),
            "turns": [turn_eval]
        }
        
        # If the interaction was finished, include the execution time
        if metrics.get("end_time"):
            new_interaction["execution_time"] = metrics.get("end_time") - metrics.get("start_time", 0)
        
        state["experiment_eval"].append(new_interaction)

def classify_query_complexity(sql: Optional[str]) -> str:
    """Classifies the complexity of a SQL query based on simple heuristics."""
    if sql is None:
        return "unknown"
    
    sql = sql.lower()
    
    # Count joins
    joins = sql.count("join")
    
    # Count where conditions
    where_clauses = 0
    if "where" in sql:
        where_part = sql.split("where")[1].split("order by")[0].split("group by")[0]
        where_clauses = where_part.count("and") + where_part.count("or") + 1
    
    # Check usage of complex operations
    has_aggregation = any(agg in sql for agg in ["count(", "sum(", "avg(", "min(", "max("])
    has_grouping = "group by" in sql
    has_ordering = "order by" in sql
    has_limit = "limit" in sql
    
    # Calculate complexity score
    complexity_score = joins + where_clauses
    if has_aggregation: complexity_score += 1
    if has_grouping: complexity_score += 1
    if has_ordering: complexity_score += 0.5
    if has_limit: complexity_score += 0.5
    
    # Classify based on score
    if complexity_score <= 1:
        return "simple"
    elif complexity_score <= 3:
        return "medium"
    else:
        return "complex"