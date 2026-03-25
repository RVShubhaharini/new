from agents.training_agent import run_training
from agents.compliance_agent import detect_deletion_request
from agents.strategy_agent import choose_strategy
from agents.unlearning_agent import run_unlearning
from agents.validation_agent import validate

def run_pipeline(data_path, customer_id):

    run_training(data_path)

    if detect_deletion_request(customer_id):
        strategy = choose_strategy("DL")
        run_unlearning("DL")

    validate()
