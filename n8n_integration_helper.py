#!/usr/bin/env python3
"""
n8n Integration Helper for Anirach's Research Automation
Provides easy functions to trigger n8n workflows from ClawdBot
"""

import requests
import json
from datetime import datetime
import os

class N8nIntegration:
    def __init__(self, webhook_base_url="https://anirach.app.n8n.cloud/webhook"):
        self.webhook_base_url = webhook_base_url
        self.research_webhook_id = "YOUR_RESEARCH_WEBHOOK_ID"  # Set after importing workflow
        self.alerts_webhook_id = "YOUR_ALERTS_WEBHOOK_ID"    # Set after importing workflow
        
    def trigger_daily_research_report(self, topics=None, custom_date=None):
        """Trigger daily research report generation via n8n"""
        if topics is None:
            topics = ["ai", "longevity", "healthcare", "machine_learning"]
            
        if custom_date is None:
            custom_date = datetime.now().strftime('%Y-%m-%d')
            
        payload = {
            "action": "generate_daily_research_report",
            "date": custom_date,
            "topics": topics,
            "format": "professional_docx",
            "requester": "clawdbot_automation"
        }
        
        url = f"{self.webhook_base_url}/{self.research_webhook_id}"
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return {
                "success": True,
                "message": "Research report generation triggered",
                "response": response.json() if response.content else {}
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def send_urgent_alert(self, title, category, institution, summary, source_url, impact_score=8):
        """Send urgent AI research alert via n8n"""
        payload = {
            "title": title,
            "category": category,
            "priority": "high",
            "impact_score": impact_score,
            "institution": institution,
            "summary": summary,
            "url": source_url,
            "keywords": f"{category},ai,research",
            "date": datetime.now().isoformat(),
            "source": "clawdbot_monitoring"
        }
        
        url = f"{self.webhook_base_url}/{self.alerts_webhook_id}"
        
        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            return {
                "success": True,
                "message": "Urgent alert sent",
                "alert_id": response.json().get("alert_id") if response.content else None
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def check_workflow_status(self, workflow_id):
        """Check status of running n8n workflow"""
        # Note: This would require n8n API access, not webhook
        # For now, return a placeholder
        return {
            "status": "running",
            "message": "Use n8n dashboard to check detailed workflow status"
        }

# Easy-to-use functions for ClawdBot integration
def trigger_morning_report(topics=None):
    """Simple function to trigger morning research report"""
    n8n = N8nIntegration()
    return n8n.trigger_daily_research_report(topics=topics)

def send_ai_breakthrough_alert(title, institution, summary, url, category="ai", impact=8):
    """Simple function to send AI breakthrough alert"""
    n8n = N8nIntegration()
    return n8n.send_urgent_alert(
        title=title,
        category=category,
        institution=institution, 
        summary=summary,
        source_url=url,
        impact_score=impact
    )

def send_longevity_alert(title, institution, summary, url, impact=8):
    """Simple function to send longevity research alert"""
    return send_ai_breakthrough_alert(
        title=title,
        institution=institution,
        summary=summary,
        url=url,
        category="longevity",
        impact=impact
    )

# Example usage functions
def example_morning_report():
    """Example: Trigger daily research report"""
    print("🔄 Triggering daily research report...")
    result = trigger_morning_report(topics=["ai", "longevity", "healthcare"])
    
    if result["success"]:
        print("✅ Report generation started successfully!")
        print(f"📋 Details: {result['message']}")
    else:
        print(f"❌ Error: {result['error']}")
        
    return result

def example_urgent_alert():
    """Example: Send urgent research alert"""
    print("🚨 Sending urgent research alert...")
    result = send_ai_breakthrough_alert(
        title="Stanford's AI Predicts Disease from Sleep Patterns",
        institution="Stanford Medicine",
        summary="Revolutionary AI system analyzes single night sleep data to predict 100+ health conditions with high accuracy",
        url="https://www.sciencedaily.com/releases/2026/01/260109023114.htm",
        impact=9
    )
    
    if result["success"]:
        print("✅ Alert sent successfully!")
        print(f"📋 Alert ID: {result.get('alert_id', 'N/A')}")
    else:
        print(f"❌ Error: {result['error']}")
        
    return result

def example_longevity_alert():
    """Example: Send longevity research alert"""
    print("🧬 Sending longevity research alert...")
    result = send_longevity_alert(
        title="Breakthrough in Cellular Aging Reversal",
        institution="Harvard Medical School", 
        summary="New gene therapy approach successfully reverses cellular aging markers in human trials",
        url="https://example-longevity-research.com/breakthrough",
        impact=10
    )
    
    if result["success"]:
        print("✅ Longevity alert sent!")
        print(f"📋 Alert ID: {result.get('alert_id', 'N/A')}")
    else:
        print(f"❌ Error: {result['error']}")
        
    return result

if __name__ == "__main__":
    """Test the integration functions"""
    print("🧪 Testing n8n Integration Helper...\n")
    
    # Test morning report trigger
    print("1️⃣ Testing morning report trigger:")
    example_morning_report()
    print()
    
    # Test urgent AI alert
    print("2️⃣ Testing AI breakthrough alert:")
    example_urgent_alert()
    print()
    
    # Test longevity alert
    print("3️⃣ Testing longevity research alert:")
    example_longevity_alert()
    print()
    
    print("✅ Integration tests completed!")
    print("\n📋 Next steps:")
    print("1. Import the n8n workflow JSON files to https://anirach.app.n8n.cloud/")
    print("2. Copy the webhook IDs from imported workflows")
    print("3. Update webhook_id variables in this script")
    print("4. Test with real data")