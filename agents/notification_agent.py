import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, EMAIL_SENDER_USER, EMAIL_SENDER_PASSWORD

def send_notification(email_address, message):
    """
    Sends an Email notification to the customer.
    If credentials are not set, it mocks the sending by logging it.
    """
    if not EMAIL_SENDER_USER or "your_email" in EMAIL_SENDER_USER or not email_address:
        # Mock Mode
        print(f"--- [MOCK EMAIL] To: {email_address} ---")
        print(f"Message: {message}")
        print("---------------------------------------")
        return "Mock Email Sent"

    try:
        # Create a secure SSL context
        context = ssl.create_default_context()
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER_USER
        msg['To'] = email_address
        msg['Subject'] = "Bank Security Notification"
        msg.attach(MIMEText(message, 'plain'))

        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(EMAIL_SENDER_USER, EMAIL_SENDER_PASSWORD)
            server.sendmail(EMAIL_SENDER_USER, email_address, msg.as_string())
            
        return "Email Sent Successfully"
    except Exception as e:
        print(f"Failed to send Email: {e}")
        return f"Email Failed: {e}"
