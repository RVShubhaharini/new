from orchestrator.workflow import run_pipeline

from orchestration.langgraph_flow import graph

result = graph.invoke({
    "forget_size": 50
})

print("\nFINAL SYSTEM STATE:\n", result)


if __name__ == "__main__":
    run_pipeline("data/bank.csv", customer_id=123)
