import csv
import os
from datetime import datetime

LOG_FILE = "unlearning_log.csv"

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Event_Type", "Forget_Size", "Model_Type", "Accuracy", "Compliance_Status", "Details"])

def log_event(event):
    # Backward compatibility for text log
    with open("audit_log.txt", "a") as f:
        f.write(f"[{datetime.now()}] {event}\n")
    print("Audit Logged:", event)

def log_to_csv(event_type, forget_size, model_type, accuracy, compliance, details):
    init_log()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event_type,
            forget_size,
            model_type,
            accuracy,
            compliance,
            details
        ])
