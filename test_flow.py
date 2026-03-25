import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from agents.action_agent import perform_unlearning

def test_full_flow():
    print("Testing Full Agentic Flow...")
    # Test with a small forget size to trigger DL unlearning (strategy < 100 -> DL)
    result = perform_unlearning(forget_size=50)
    print("Result:", result)
    
if __name__ == "__main__":
    test_full_flow()
