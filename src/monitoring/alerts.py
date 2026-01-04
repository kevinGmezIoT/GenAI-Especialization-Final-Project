def send_alert(message):
    """
    Sends an alert. Currently implemented as a console print, 
    but can be extended to email/Slack.
    """
    # Simple, academic-defensible alerting
    alert_msg = f"🚨 MONITORING ALERT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): {message}"
    print("\n" + "="*50)
    print(alert_msg)
    print("="*50 + "\n")
    
    # In a real scenario, this could write to a dedicated alerts log or trigger a webhook.

from datetime import datetime
