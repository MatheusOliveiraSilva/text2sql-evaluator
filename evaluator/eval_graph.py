from evaluator.eval_states import GraphState
from evaluator.eval_nodes import EvaluatorNodes
from evaluator.eval_edges import EvaluatorEdges

from langgraph.graph import StateGraph, START

evaluator_nodes = EvaluatorNodes()
evaluator_edges = EvaluatorEdges()

graph_builder = StateGraph(GraphState)
graph_builder.add_node("Setup", evaluator_nodes.setup)
graph_builder.add_node("User Interaction", evaluator_nodes.user_node)
graph_builder.add_node("Check Response", evaluator_nodes.check_response)

graph_builder.add_edge(START, "Setup")
graph_builder.add_edge("Setup", "User Interaction")
graph_builder.add_edge("User Interaction", "Check Response")
graph_builder.add_conditional_edges("Check Response", evaluator_edges.keep_going)

eval_graph = graph_builder.compile()