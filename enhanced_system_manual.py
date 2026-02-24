#!/usr/bin/env python3
"""
Enhanced ClawdBot System Configuration Manual Generator
Adds detailed technical sections, code examples, and troubleshooting procedures
"""

import sys
sys.path.append('/home/clawdbot/clawd/professional-docx-generator')
from scripts.docx_generator import ProfessionalDocumentGenerator
from datetime import datetime

def create_enhanced_manual():
    """Create comprehensive system configuration manual"""
    generator = ProfessionalDocumentGenerator()
    
    # Cover page
    generator.add_cover_page(
        title="ClawdBot Enterprise System Configuration Manual",
        subtitle="Complete Setup Guide with Local LLM and n8n Integration",
        author="Enterprise AI Solutions Team",
        date=datetime.now().strftime('%B %d, %Y'),
        organization="ClawdBot Enterprise Documentation"
    )
    
    # Table of contents
    sections = [
        "Executive Overview",
        "Architecture and Components", 
        "Hardware Requirements",
        "Local LLM Setup (Ollama)",
        "ClawdBot Enterprise Installation",
        "n8n Workflow Integration",
        "Security Configuration",
        "Department Agent Configuration",
        "API Reference and Integration",
        "Testing and Validation",
        "Troubleshooting Guide",
        "Maintenance Procedures",
        "Appendices"
    ]
    generator.add_table_of_contents(sections)
    
    # Executive overview
    generator.add_section("Executive Overview", 
        "This manual provides comprehensive instructions for deploying ClawdBot Enterprise with local Large Language Model (LLM) infrastructure and n8n workflow automation. The deployment creates a secure, scalable AI assistant platform that maintains data sovereignty while enabling powerful automation capabilities.")
    
    # Architecture overview
    generator.add_section("Architecture and Components", "", {
        "Three-Tier Architecture": "Tier 1: Local LLM Processing (Ollama), Tier 2: Intelligent Routing Engine, Tier 3: Enterprise Integrations (n8n)",
        "Core Components": "ClawdBot Enterprise platform, Ollama LLM infrastructure, n8n workflow automation, Security and audit systems",
        "Data Flow": "User queries → Security classification → Local/Cloud routing → AI processing → Response generation → Audit logging",
        "Integration Points": "Communication platforms (Discord, Telegram, Slack), Enterprise systems (CRM, ERP), Monitoring and analytics"
    })
    
    # Hardware requirements table
    generator.add_professional_table(
        ["Component", "Minimum", "Recommended", "Enterprise"],
        [
            ["CPU", "8 cores", "16 cores", "Mac Studio M3 Ultra"],
            ["Memory", "16GB", "32GB", "64GB+ unified"],
            ["Storage", "500GB SSD", "1TB NVMe", "2TB+ enterprise"],
            ["GPU", "Integrated", "Dedicated 8GB", "Apple Silicon / RTX 4090"],
            ["Network", "1Gbps", "10Gbps", "Enterprise backbone"]
        ],
        "Hardware Requirements Matrix"
    )
    
    # Detailed installation sections
    generator.add_section("Local LLM Setup (Ollama)", "", {
        "Installation Command": "curl -fsSL https://ollama.ai/install.sh | sh",
        "Service Configuration": "sudo systemctl enable ollama && sudo systemctl start ollama",
        "Model Deployment": "ollama pull llama3.1:8b && ollama pull llama3.1:70b && ollama pull codellama:7b",
        "Memory Optimization": "export OLLAMA_HOST=127.0.0.1:11434 && export OLLAMA_MAX_LOADED_MODELS=4",
        "Security Settings": "Configure firewall rules, enable local-only binding, implement access controls"
    })
    
    generator.add_section("ClawdBot Enterprise Installation", "", {
        "Prerequisites": "Node.js 18+, npm 9+, enterprise license key, SSL certificates",
        "Base Installation": "npm install -g clawdbot-enterprise && clawdbot init --enterprise",
        "Configuration": "Edit config/enterprise.json with organization settings and security policies",
        "Service Setup": "Install systemd service, configure auto-restart, enable monitoring",
        "Agent Deployment": "Deploy department-specific agents with appropriate security levels"
    })
    
    generator.add_section("n8n Workflow Integration", "", {
        "Docker Installation": "docker run -d --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n n8nio/n8n",
        "Database Setup": "Configure PostgreSQL backend for production deployment",
        "ClawdBot Integration": "Install n8n-nodes-clawdbot package for native integration",
        "Workflow Templates": "Import pre-built templates for common enterprise scenarios",
        "Security Configuration": "Configure OAuth, API keys, and secure credential storage"
    })
    
    # Security configuration table
    generator.add_professional_table(
        ["Security Layer", "Component", "Configuration", "Validation"],
        [
            ["Network", "Firewall Rules", "Allow 443, 8080, 11434", "Test connectivity"],
            ["Authentication", "LDAP/SAML", "Configure SSO integration", "Test user login"],
            ["Encryption", "TLS Certificates", "Deploy valid SSL certs", "Verify HTTPS"],
            ["Access Control", "RBAC Policies", "Define role permissions", "Test access"],
            ["Audit Logging", "Log Management", "Configure comprehensive logging", "Review logs"],
            ["Data Protection", "Classification", "Implement data policies", "Test DLP"]
        ],
        "Security Configuration Matrix"
    )
    
    # Department configuration
    generator.add_section("Department Agent Configuration", "", {
        "HR Department": "Local-only processing, employee data protection, benefits integration",
        "Finance Department": "Maximum security mode, financial data isolation, compliance controls",
        "IT Department": "Hybrid processing, technical documentation, incident management",
        "Marketing Department": "Cloud-enabled processing, creative tools, campaign management",
        "Executive Team": "Strategic analysis, confidential processing, board reporting"
    })
    
    # API reference table
    generator.add_professional_table(
        ["Endpoint", "Method", "Description", "Security Level"],
        [
            ["/api/v1/agents", "GET", "List department agents", "Internal"],
            ["/api/v1/llm/query", "POST", "Process LLM query", "Authenticated"],
            ["/api/v1/workflows", "GET", "List n8n workflows", "Internal"],
            ["/api/v1/security/audit", "GET", "Security audit logs", "Admin Only"],
            ["/api/v1/health", "GET", "System health check", "Monitoring"],
            ["/api/v1/config", "PUT", "Update configuration", "Admin Only"]
        ],
        "API Endpoint Reference"
    )
    
    # Testing procedures
    generator.add_section("Testing and Validation", "", {
        "Local LLM Testing": "Verify model loading, test response times, validate memory usage",
        "Agent Testing": "Test department-specific responses, verify security controls",
        "Workflow Testing": "Validate n8n integrations, test automation triggers",
        "Security Testing": "Penetration testing, access control validation, audit verification",
        "Performance Testing": "Load testing, stress testing, capacity planning"
    })
    
    # Troubleshooting guide
    generator.add_section("Troubleshooting Guide", "", {
        "Ollama Issues": "Check service status: sudo systemctl status ollama. Restart: sudo systemctl restart ollama",
        "Memory Problems": "Monitor usage: ollama ps. Clear cache: ollama rm <model>. Optimize allocation",
        "Network Connectivity": "Verify firewall: sudo ufw status. Test ports: telnet localhost 11434",
        "Authentication Errors": "Check LDAP config, verify certificates, test SSO integration",
        "Performance Issues": "Monitor resources, optimize model selection, scale horizontally"
    })
    
    # Maintenance procedures
    generator.add_section("Maintenance Procedures", "", {
        "Daily Monitoring": "Check system health, review logs, monitor performance metrics",
        "Weekly Updates": "Update models, review security alerts, optimize workflows",
        "Monthly Reviews": "Capacity planning, security audits, performance analysis",
        "Quarterly Upgrades": "Software updates, hardware assessment, business review"
    })
    
    # Support contact information
    generator.add_section("Support and Resources", "", {
        "Technical Support": "enterprise-support@clawdbot.com | 24/7 critical issues",
        "Documentation": "https://docs.clawdbot.com/enterprise | Complete reference",
        "Community": "https://discord.gg/clawdbot | Community support and updates",
        "Training": "https://training.clawdbot.com | Certification programs available"
    })
    
    return generator

def main():
    """Generate the enhanced system manual"""
    print("🚀 Generating Enhanced ClawdBot System Configuration Manual...")
    
    generator = create_enhanced_manual()
    filename = generator.save("ClawdBot_Enterprise_System_Manual_Complete.docx")
    
    print(f"✅ Enhanced system manual created: {filename}")
    return filename

if __name__ == "__main__":
    main()