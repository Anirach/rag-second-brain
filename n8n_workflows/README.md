# n8n Workflows for Anirach's Research Automation

## 📋 **Available Workflows**

### 1. **Daily Research Automation** (`daily_research_automation.json`)
- **Trigger**: Daily at 7:00 AM Bangkok time
- **Function**: Generate AI + Longevity research morning update
- **Output**: Professional DOCX report uploaded to Google Drive
- **Notifications**: Telegram + Email alerts when ready

### 2. **Instant AI Alerts** (`instant_ai_alerts.json`) 
- **Trigger**: Webhook from external news sources
- **Function**: Filter high-priority AI/longevity breakthroughs
- **Output**: Instant Telegram + Email notifications for urgent research
- **Features**: Impact scoring and relevance filtering

### 3. **Document Distribution System** (`document_distribution.json`)
- **Trigger**: New files in Google Drive research folder
- **Function**: Auto-distribute research reports to team
- **Output**: Telegram, Email, Slack notifications + University system logging

## 🚀 **How to Import Workflows**

### Step 1: Access Your n8n Cloud
Go to https://anirach.app.n8n.cloud/

### Step 2: Import Workflow
1. Click "**+ Add Workflow**" 
2. Click "**Import from JSON**"
3. Copy the content from any `.json` file above
4. Paste into the text area
5. Click "**Import**"

### Step 3: Configure Credentials
Set up these credentials in n8n:

**Telegram Bot:**
- Type: `Telegram Bot`
- Bot Token: Your Telegram bot token
- Name: `telegramBot`

**Email (SMTP):**
- Type: `SMTP`
- Host: Your email provider SMTP
- Port: 587 (or your provider's port)
- Username: Your email
- Password: Your email password

**Google Drive:**
- Type: `Google Drive OAuth2`
- Follow n8n's Google OAuth setup guide

## 🔧 **Configuration Required**

### For Daily Research Automation:
1. **Replace webhook URL**: Update `https://your-clawdbot-webhook-endpoint.com`
2. **Set API token**: Replace `YOUR_API_TOKEN` with your ClawdBot token
3. **Configure email**: Replace `anirach@your-email.com` with your email

### For Instant AI Alerts:
1. **Webhook URL**: Will be generated when you save the workflow
2. **Telegram credentials**: Configure your bot token
3. **Database URL**: Replace with your research database endpoint (optional)

### For Document Distribution:
1. **Google Drive Folder ID**: Replace `YOUR_GOOGLE_DRIVE_FOLDER_ID`
2. **Email recipients**: Update distribution list
3. **Slack webhook**: Replace with your Slack incoming webhook URL
4. **University system**: Configure your institution's API endpoint

## 🔗 **ClawdBot Integration Endpoints**

To connect these workflows with your ClawdBot system, you'll need these webhook endpoints:

### Research Report Generation:
```bash
# POST endpoint for triggering report generation
curl -X POST https://your-clawdbot-endpoint.com/generate-report \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "action": "generate_daily_research_report",
    "date": "2026-01-29",
    "topics": ["ai", "longevity", "healthcare"],
    "format": "professional_docx"
  }'
```

### Status Check:
```bash
# GET endpoint for checking report status
curl -X GET https://your-clawdbot-endpoint.com/status/REPORT_ID \\
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 📊 **Webhook Response Format**

### Report Generation Response:
```json
{
  "status": "accepted",
  "report_id": "unique-report-id",
  "estimated_completion": "2026-01-29T07:05:00Z"
}
```

### Status Check Response:
```json
{
  "status": "completed",
  "report_id": "unique-report-id", 
  "google_drive_url": "https://docs.google.com/document/d/...",
  "papers_count": 15,
  "summary": "Key findings from today's research..."
}
```

## 🔐 **Security Notes**

1. **Never commit API keys** to version control
2. **Use n8n credential system** for all sensitive data
3. **Restrict webhook access** with proper authentication
4. **Monitor workflow executions** regularly

## 📞 **Next Steps**

1. **Import one workflow** (start with Daily Research Automation)
2. **Configure credentials** (Telegram, Email, Google Drive)
3. **Test the workflow** with manual execution
4. **Set up ClawdBot webhook endpoints** 
5. **Monitor and adjust** as needed

## ⚡ **Quick Start Command**

To get the webhook URL after importing the Instant AI Alerts workflow:

1. Go to the "AI News Webhook" node
2. Copy the "Production URL" 
3. Use this URL to send high-priority alerts:

```bash
curl -X POST https://anirach.app.n8n.cloud/webhook/YOUR-WEBHOOK-ID \\
  -H "Content-Type: application/json" \\
  -d '{
    "title": "Stanford Breakthrough in AI Longevity",
    "category": "longevity",
    "priority": "high",
    "impact_score": 9,
    "institution": "Stanford Medicine",
    "summary": "Revolutionary AI system predicts disease...",
    "url": "https://source-url.com",
    "keywords": "longevity,healthcare,AI"
  }'
```

Ready to automate your research workflow! 🤖