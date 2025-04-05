import json
import time
from pathlib import Path
from typing import Literal
from collections import deque
from datetime import datetime
from langgraph.graph import END
from models_utils.llm_config import LLMConfig
import evaluator.eval_prompts as prompts
import evaluator.utils as utils 
from langchain_core.messages import HumanMessage
from evaluator.eval_states import GraphState
from langchain_core.output_parsers import StrOutputParser

root_path = Path().absolute()

class EvaluatorNodes:
    def __init__(self, agent_endpoint: str):
        self.agent_endpoint = agent_endpoint

        self.llm = LLMConfig(provider="azure").get_llm(max_tokens=2000)

    def setup(self, state: GraphState) -> GraphState:
        """
        Start node, get all data from experiment structure and setup graph variables.
        """
        print("[NODE] Setup Node entered.")

        # We will store the evaluation results here
        state["experiment_eval"] = []
        
        state["experiment_config"] = {
            "max_retries": state.get("max_retries", 2),
            "model_version": state.get("model_version", "default"),
            "timestamp": datetime.now().isoformat(),
            "experiment_type": state.get("experiment_type", "standard")
        }
        
        state["interaction_metrics"] = {}
        state["query_complexity"] = "unknown"
        state["retry_reason"] = None

        state["turns"] = deque(state['experiment']["interactions"])
        state["max_retries"] = 2 if "max_retries" not in state else state["max_retries"]
        state["debug_mode"] = False if "debug_mode" not in state else state["debug_mode"]

        state["actual_number_of_retries"] = 0

        state["proceed"] = True
        state["go_next_interaction"] = True
        state["interactions_counting"] = 0

        state["dialogue_agent_config"] = {
            # Pattern will be dialogue_agent_<id>
            "configurable": {"thread_id": "dialogue_agent_" + state["experiment"]["experiment_id"]}
        }
        print("----" * 10)
        return state

    def user_node(self, state: GraphState) -> GraphState:
        """
        User Interaction node, this node simulate a real user behavior. 
        We send a message to the agent and wait for a response, if the answer is the expected, 
        we go to next turn. If not, we reply using feedbacks until get the expected answer 
        (limited by a max of retries).
        """
        print("[NODE] User Interaction Node entered.")

        nl_query = ""  # NL query will be or the next interaction or a feedback query

        # Check if we can go to next interaction.
        if state["go_next_interaction"]:
            state["interactions_counting"] += 1
            current_interaction_id = state["interactions_counting"]
            
            # Start tracking time of the interaction
            state["current_interaction_start_time"] = time.time()
            
            # Initialize metrics for this interaction
            state["actual_number_of_retries"] = 0
            state["actual_turn"] = state["turns"].popleft() if state["turns"] else None

            if state["actual_turn"] is None:  # End evaluation
                state["proceed"] = False
                return state
            
            original_intent = state["actual_turn"]["intention"]
                
            state["interaction_metrics"][current_interaction_id] = {
                "original_intent": original_intent,
                "total_retries_needed": 0,
                "success_without_retry": True,
                "start_time": state["current_interaction_start_time"],
                "end_time": None
            }
            
            nl_query = state["actual_turn"]["utterance"]

        # If we can't go to next interaction, we will use the last user input and the chat 
        # history to generate a new feedback query.
        else:
            current_interaction_id = state["interactions_counting"]
            
            if current_interaction_id in state["interaction_metrics"]:
                state["interaction_metrics"][current_interaction_id]["total_retries_needed"] += 1
                state["interaction_metrics"][current_interaction_id]["success_without_retry"] = False
            
            prompt = prompts.USER_INTERACTION_PROMPT.format(
                chat_history=utils.messages_to_string_list(state["last_response"]["messages"]),  # Use utils function
                user_intention=state["actual_turn"]["intention"]
            )

            msg = HumanMessage(content=prompt)
            feedback_chain = self.llm | StrOutputParser()
            feedback = feedback_chain.invoke([msg])

            nl_query = feedback

        state["last_user_input"] = nl_query

        if state["debug_mode"]: print("[INFO] Enviando a query para o agente: ", nl_query)

        messages = [HumanMessage(content=nl_query)]
        state["last_response"] = {"messages": [HumanMessage(content='{"input": "test input", "schema_linking": [], "answer": "dummy answer", "sql": "SELECT 1"}')]} # Placeholder
        # TODO: Call the agent via endpoint
        # Example using requests (replace with actual implementation):
        # import requests
        # response = requests.post(self.agent_endpoint, json={
        #     "messages": [msg.dict() for msg in messages], 
        #     "config": state["dialogue_agent_config"]
        # })
        # state["last_response"] = response.json()

        state["interaction_history"] = state["last_response"]["messages"]

        # Sanitize the last response to remove the json format if it exists
        last_msg = state["last_response"]["messages"][-1]
        if "```json" in last_msg.content:
            last_msg.content = last_msg.content.replace("```json", "").replace("```", "")

        if state["debug_mode"]: print(f"[INFO] The result of the execution was: {last_msg.content}.\n")

        print("----" * 10)
        return state

    def check_response(self, state: GraphState) -> GraphState:
        """
        Node that checks if the agent's response is as expected.
        Handles JSON parsing, evaluates alignment and correctness, determines if feedback is needed,
        manages retries, updates metrics, and decides the next step.
        """
        print("[NODE] Check Response Node entered.")
        current_interaction_id = state["interactions_counting"]
        agent_response_content = state['last_response']['messages'][-1].content
        response_data = None
        function_input = ""
        tables_from_schema_linking = []
        answer = ""
        danke_sql = ""
        
        # 1. Parse Agent Response
        try:
            print(f"[INFO] Raw agent response: {agent_response_content}.")
            json_str = utils.extract_outer_json(agent_response_content)
            if json_str is None:
                raise json.JSONDecodeError("No JSON object found in response", agent_response_content, 0)
            
            response_data = json.loads(json_str)
            function_input = response_data.get("input", "")
            tables_from_schema_linking = response_data.get("schema_linking", [])
            answer = response_data.get("answer", "")
            danke_sql = response_data.get("sql", "").replace("\\n", " ")
            state["retry_reason"] = None # Clear previous reason if parsing succeeds

        except json.JSONDecodeError as e:
            print(f"[ERROR] Error decoding JSON: {e}. Starting retry.")
            state["retry_reason"] = "json_decode_error"
            state["actual_number_of_retries"] += 1
            # Update interaction metrics for failure due to JSON error
            if current_interaction_id in state["interaction_metrics"]:
                state["interaction_metrics"][current_interaction_id]["total_retries_needed"] += 1
                state["interaction_metrics"][current_interaction_id]["success_without_retry"] = False
            
            # Decide if we can proceed with retry
            state["go_next_interaction"] = False
            state["proceed"] = state["actual_number_of_retries"] <= state["max_retries"]
            # Append a minimal evaluation indicating the JSON error
            turn_eval = {
                "user_query": state["last_user_input"],
                "agent_reply": "Error: Could not parse agent response.",
                "evaluation": {
                    "error": "json_decode_error",
                    "raw_response": agent_response_content,
                    "is_retry": True,
                    "retry_count": state["actual_number_of_retries"],
                    "retry_reason": state["retry_reason"],
                    "execution_time": time.time() - state.get("current_interaction_start_time", time.time()),
                }
            }
            utils.append_turn_evaluation(state, turn_eval)
            print("----" * 10)
            return state

        # 2. Gather Ground Truths and History
        chat_history = state["interaction_history"]
        ground_truths = state["actual_turn"]["ground_truths"]
        intention = state["actual_turn"]["intention"]
        ground_truth_danke_sql = ground_truths.get("danke_sql", None)
        
        if state["debug_mode"]: print(f"[INFO] Evaluating the result: {response_data}.")

        # 3. Calculate Core Metrics
        need_user_feedback = utils.need_feedback(chat_history)
        tables_ground_truth = ground_truths["tables_from_schema_linking"]
        recall = utils.calculate_tables_recall(tables_ground_truth, tables_from_schema_linking, state.get("debug_mode", False))
        alignment = utils.compare_intentions(function_input, intention, chat_history)
        query_complexity = utils.classify_query_complexity(ground_truth_danke_sql)
        execution_time = time.time() - state["current_interaction_start_time"]
        
        # 4. Evaluate Correctness
        correctness = False # Default to False
        evaluator = state.get("evaluator", None)
        if evaluator and ground_truth_danke_sql and danke_sql:
            try:
                result_table = evaluator.run_sql_query(danke_sql)
                true_table = evaluator.run_sql_query(ground_truth_danke_sql)
                correctness, sim, col_match = evaluator.compare_sql_query_similarity_and_semantic(
                    user_query=function_input,
                    generated_query=danke_sql,
                    result_table=result_table,
                    true_query=ground_truth_danke_sql,
                    true_table=true_table,
                    similarity_threshold=0.8,
                    column_matching_threshold=0.5,
                    debug_mode=state.get("debug_mode", False)
                )
            except Exception as e:
                print(f"[ERROR] Error executing or comparing SQL query: {e}")
                state["retry_reason"] = "query_execution_error"
                correctness = False
        elif not ground_truth_danke_sql or not danke_sql:
             print("[INFO] Correctness check skipped: Missing ground truth or generated SQL.")
             state["retry_reason"] = "missing_sql_for_comparison"
             correctness = False
        else: # Case where evaluator is None
             print("[WARN] Correctness check skipped: Evaluator not available in state.")
             state["retry_reason"] = "evaluator_missing"
             correctness = False # Or True depending on desired behavior

        # 5. Determine Success and Need for Retry
        # Success condition: Alignment or Correctness is True, AND no feedback needed.
        # Modify this condition based on exact requirements. Maybe feedback implies failure?
        is_successful_turn = (alignment or correctness) and not need_user_feedback
        
        if need_user_feedback:
            # If feedback is needed, treat as failure for retry logic, unless alignment/correctness is already true
            # This logic might need refinement: Does feedback ALWAYS mean a retry, even if alignment/correctness is somehow True?
            # Assuming feedback needed implies a retry unless already correct/aligned.
            is_successful_turn = alignment or correctness # Override: succeed if aligned/correct despite feedback request?
            if not is_successful_turn:
                 state["retry_reason"] = "feedback_needed"

        # 6. Update State based on Success/Failure
        if is_successful_turn:
            # Finalize interaction metrics successfully
            if current_interaction_id in state["interaction_metrics"]:
                state["interaction_metrics"][current_interaction_id]["end_time"] = time.time()
            
            state["go_next_interaction"] = True
            state["proceed"] = bool(state["turns"]) # Proceed if more turns exist
            state["retry_reason"] = None # Reset retry reason on success
            state["actual_number_of_retries"] = 0 # Reset retries for the next interaction
        
        else: # Turn failed or requires feedback retry
            state["actual_number_of_retries"] += 1
            
            # Update interaction metrics for failure/retry needed
            if current_interaction_id in state["interaction_metrics"]:
                state["interaction_metrics"][current_interaction_id]["total_retries_needed"] += 1
                state["interaction_metrics"][current_interaction_id]["success_without_retry"] = False

            # Determine retry reason if not already set (e.g., feedback_needed, query_execution_error)
            if not state["retry_reason"]:
                if not alignment and not correctness:
                    state["retry_reason"] = "alignment_and_correctness_failure"
                elif not alignment:
                    state["retry_reason"] = "alignment_failure"
                elif not correctness:
                    state["retry_reason"] = "correctness_failure"
                else: # Should not happen if is_successful_turn is False, but as fallback
                     state["retry_reason"] = "unknown_failure"

            # Decide if we continue this interaction (retry) or move to the next
            can_retry = state["actual_number_of_retries"] <= state["max_retries"]
            
            if can_retry:
                state["go_next_interaction"] = False # Stay on this interaction
                state["proceed"] = True # Continue the graph flow for retry
            else: # Max retries reached for this interaction
                print(f"[INFO] Max retries ({state['max_retries']}) reached for interaction {current_interaction_id}.")
                 # Finalize interaction metrics even with failure
                if current_interaction_id in state["interaction_metrics"]:
                    state["interaction_metrics"][current_interaction_id]["end_time"] = time.time()

                state["go_next_interaction"] = True # Move to the next interaction
                state["proceed"] = bool(state["turns"]) # Proceed only if more turns exist
                state["retry_reason"] = "max_retries_reached" # Update reason for the final failed turn
                # Reset retry counter for the *next* interaction if we proceed
                if state["proceed"]:
                    state["actual_number_of_retries"] = 0 


        # 7. Construct and Append Evaluation Turn
        turn_eval = {
            "user_query": state["last_user_input"],
            "agent_reply": answer,
            "evaluation": {
                "text_to_sql_input": function_input,
                "user_intention": intention,
                "recall": recall,
                "alignment": alignment,
                "correctness": correctness,
                "expected_sql": ground_truth_danke_sql,
                "generated_sql": danke_sql,
                "needs_feedback": need_user_feedback,
                "is_retry": state["actual_number_of_retries"] > 0 and not is_successful_turn, # Mark as retry only if it wasn't the final successful attempt
                "retry_count": state["actual_number_of_retries"] if not is_successful_turn else state["actual_number_of_retries"] -1 , # Show count leading to this state
                "retry_reason": state["retry_reason"], # Captures the reason for failure/retry
                "execution_time": execution_time,
                "query_complexity": query_complexity
            }
        }
        
        # Adjust retry count if it was the last attempt and failed due to max retries
        if state["retry_reason"] == "max_retries_reached":
             turn_eval["evaluation"]["retry_count"] = state["max_retries"]


        utils.append_turn_evaluation(state, turn_eval)

        if state["debug_mode"]: print("[INFO] Evaluation for this turn: ", state["experiment_eval"][-1])

        print("----" * 10)
        return state