from typing import Literal
from langgraph.graph import END
from evaluator.eval_states import GraphState

class EvaluatorEdges:

    def keep_going(self, state: GraphState) -> Literal["User Interaction", END]:
        """
        Function that serves as conditional edge, if state["proceed"] == True, we continue to the next node, otherwise we end the evaluation.
        Motives to end the evaluation:
        No more turns at experiment, or the last turn was not properly answered. Also decide to finish evaluation if max retries on a tool is reached.
        """
        print(f"[Conditional Edge] Continuamos a avaliação? {state['proceed']}")
        print("----" * 10)
        if state["proceed"]:
            return "User Interaction"
        else:
            return END
