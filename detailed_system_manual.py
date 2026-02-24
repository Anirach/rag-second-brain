#!/usr/bin/env python3
"""
ClawdBot System Configuration Manual with Detailed Step-by-Step Commands
Professional format with clear command blocks and verification steps
"""

import sys
sys.path.append('/home/clawdbot/clawd/professional-docx-generator')
from scripts.docx_generator import ProfessionalDocumentGenerator
from datetime import datetime

def create_detailed_system_manual():
    """Create system manual with detailed command-line instructions"""
    generator = ProfessionalDocumentGenerator()
    
    # Cover page
    generator.add_cover_page(
        title="ClawdBot Enterprise System Configuration Manual",
        subtitle="Step-by-Step Installation Guide with Local LLM and n8n Integration",
        author="Enterprise AI Solutions Team",
        date=datetime.now().strftime('%B %d, %Y'),
        organization="ClawdBot Enterprise Documentation"
    )
    
    # Table of contents
    sections = [
        "Prerequisites and Planning",
        "System Preparation", 
        "Install Homebrew (Package Manager)",
        "Install Node.js (Version 22+)",
        "Install Docker Desktop",
        "Install Ollama (Local LLM Platform)",
        "Deploy Local LLM Models",
        "Install ClawdBot Enterprise",
        "Configure ClawdBot Agents",
        "Install n8n Workflow Automation",
        "Configure n8n Integration",
        "Security Configuration",
        "Testing and Validation",
        "Troubleshooting and Maintenance"
    ]
    generator.add_table_of_contents(sections)
    
    # Prerequisites
    generator.add_section("Prerequisites and Planning", "", {
        "Hardware Requirements": "Mac Studio M3 Ultra (recommended) or equivalent with 32GB+ RAM, 1TB+ SSD storage",
        "Operating System": "macOS 13.0+ or Ubuntu 22.04+ with admin privileges",
        "Network Access": "Internet connectivity for downloads, HTTPS access for integrations",
        "Preparation Time": "Allow 2-4 hours for complete installation and configuration",
        "Backup Recommendation": "Create system backup before beginning installation process"
    })
    
    # System preparation
    generator.add_section("System Preparation", "", {
        "Update System": "Ensure your system is fully updated before beginning installation",
        "Free Disk Space": "Verify at least 100GB free space for models and applications", 
        "Admin Access": "Confirm administrator privileges for system-level installations",
        "Network Configuration": "Configure firewall settings if running in enterprise environment"
    })
    
    # Section 1: Install Homebrew
    generator.add_section("1. Install Homebrew (Package Manager)", "")
    
    generator.doc.add_paragraph("Homebrew is required to install most development tools on macOS. Open Terminal and run:")
    
    generator.doc.add_paragraph().text = "    /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    generator.doc.add_paragraph().runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("After installation, add Homebrew to your PATH:")
    
    para = generator.doc.add_paragraph()
    para.text = "    echo 'eval \"$(/opt/homebrew/bin/brew shellenv)\"' >> ~/.zprofile\n    eval \"$(/opt/homebrew/bin/brew shellenv)\""
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Verify installation:")
    
    para = generator.doc.add_paragraph()
    para.text = "    brew --version    # Should show Homebrew version"
    para.runs[0].font.name = 'Courier New'
    
    # Section 2: Install Node.js
    generator.add_section("2. Install Node.js (Version 22+)", "")
    
    generator.doc.add_paragraph("ClawdBot requires Node.js version 22 or higher. Install using Homebrew:")
    
    para = generator.doc.add_paragraph()
    para.text = "    # Install Node.js using Homebrew\n    brew install node@22"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Alternative: Use nvm (Node Version Manager) for flexibility:")
    
    para = generator.doc.add_paragraph()
    para.text = "    # Install nvm\n    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash\n    source ~/.zshrc\n    \n    # Install and use Node.js 22\n    nvm install 22\n    nvm use 22"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Verify installation:")
    
    para = generator.doc.add_paragraph()
    para.text = "    node --version    # Should show v22.x.x\n    npm --version     # Should show 10.x.x or higher"
    para.runs[0].font.name = 'Courier New'
    
    # Section 3: Install Docker
    generator.add_section("3. Install Docker Desktop", "")
    
    generator.doc.add_paragraph("Docker is required for n8n and database services:")
    
    generator.doc.add_paragraph("Option 1: Download from Docker website")
    para = generator.doc.add_paragraph()
    para.text = "    1. Visit https://docker.com/products/docker-desktop\n    2. Download Docker Desktop for Mac (Apple Silicon)\n    3. Open the downloaded .dmg file\n    4. Drag Docker to Applications folder\n    5. Launch Docker Desktop and complete setup wizard"
    
    generator.doc.add_paragraph("Option 2: Install via Homebrew")
    para = generator.doc.add_paragraph()
    para.text = "    brew install --cask docker"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Configure Docker resources:")
    para = generator.doc.add_paragraph()
    para.text = "    1. Open Docker Desktop → Settings → Resources\n    2. Allocate at least 32GB RAM and 8 CPUs\n    3. Set disk space to 100GB+\n    4. Click \"Apply & Restart\""
    
    generator.doc.add_paragraph("Verify installation:")
    para = generator.doc.add_paragraph()
    para.text = "    docker --version    # Should show Docker version\n    docker ps           # Should show empty container list"
    para.runs[0].font.name = 'Courier New'
    
    # Section 4: Install Ollama
    generator.add_section("4. Install Ollama (Local LLM Platform)", "")
    
    generator.doc.add_paragraph("Ollama provides local LLM processing capabilities:")
    
    para = generator.doc.add_paragraph()
    para.text = "    # Install Ollama\n    curl -fsSL https://ollama.ai/install.sh | sh"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Alternative manual installation:")
    para = generator.doc.add_paragraph()
    para.text = "    # Download and install manually\n    brew install ollama"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Configure Ollama service:")
    para = generator.doc.add_paragraph()
    para.text = "    # Set environment variables\n    export OLLAMA_HOST=127.0.0.1:11434\n    export OLLAMA_MAX_LOADED_MODELS=4\n    export OLLAMA_MODELS_DIR=/opt/ollama/models\n    \n    # Add to shell profile\n    echo 'export OLLAMA_HOST=127.0.0.1:11434' >> ~/.zprofile\n    echo 'export OLLAMA_MAX_LOADED_MODELS=4' >> ~/.zprofile"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Start Ollama service:")
    para = generator.doc.add_paragraph()
    para.text = "    # Start Ollama service\n    ollama serve\n    \n    # Or run as background service\n    brew services start ollama"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Verify installation:")
    para = generator.doc.add_paragraph()
    para.text = "    ollama --version    # Should show Ollama version\n    curl http://localhost:11434/api/tags    # Should return JSON response"
    para.runs[0].font.name = 'Courier New'
    
    # Section 5: Deploy LLM Models
    generator.add_section("5. Deploy Local LLM Models", "")
    
    generator.doc.add_paragraph("Download and deploy essential LLM models:")
    
    generator.doc.add_paragraph("Primary models for ClawdBot:")
    para = generator.doc.add_paragraph()
    para.text = "    # Llama 3.1 8B (General purpose - 16GB RAM)\n    ollama pull llama3.1:8b\n    \n    # Llama 3.1 70B (Complex analysis - 45GB RAM)\n    ollama pull llama3.1:70b\n    \n    # Code Llama 7B (Technical support - 14GB RAM)\n    ollama pull codellama:7b\n    \n    # Mistral 7B (Fast responses - 12GB RAM)\n    ollama pull mistral:7b"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Verify model deployment:")
    para = generator.doc.add_paragraph()
    para.text = "    # List installed models\n    ollama list\n    \n    # Test model response\n    ollama run llama3.1:8b \"Hello, test message\""
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Monitor system resources:")
    para = generator.doc.add_paragraph()
    para.text = "    # Check running models\n    ollama ps\n    \n    # Monitor memory usage\n    top -o MEM | head -20"
    para.runs[0].font.name = 'Courier New'
    
    # Section 6: Install ClawdBot Enterprise
    generator.add_section("6. Install ClawdBot Enterprise", "")
    
    generator.doc.add_paragraph("Install ClawdBot Enterprise platform:")
    
    para = generator.doc.add_paragraph()
    para.text = "    # Install ClawdBot CLI globally\n    npm install -g clawdbot-enterprise\n    \n    # Verify installation\n    clawdbot --version"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Initialize ClawdBot workspace:")
    para = generator.doc.add_paragraph()
    para.text = "    # Create enterprise workspace\n    mkdir /opt/clawdbot-enterprise\n    cd /opt/clawdbot-enterprise\n    \n    # Initialize with enterprise settings\n    clawdbot init --enterprise --workspace ."
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Configure enterprise license:")
    para = generator.doc.add_paragraph()
    para.text = "    # Add enterprise license key\n    clawdbot license add --key YOUR_ENTERPRISE_LICENSE_KEY\n    \n    # Verify license\n    clawdbot license verify"
    para.runs[0].font.name = 'Courier New'
    
    # Section 7: Configure ClawdBot Agents
    generator.add_section("7. Configure ClawdBot Agents", "")
    
    generator.doc.add_paragraph("Create department-specific AI agents:")
    
    generator.doc.add_paragraph("HR Department Agent (Local-only processing):")
    para = generator.doc.add_paragraph()
    para.text = "    # Create HR agent with maximum security\n    clawdbot agent create hr \\\n      --model llama3.1:8b \\\n      --security-level confidential \\\n      --local-only \\\n      --department hr"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Finance Department Agent:")
    para = generator.doc.add_paragraph()
    para.text = "    # Create Finance agent with strict security\n    clawdbot agent create finance \\\n      --model llama3.1:70b \\\n      --security-level strict \\\n      --local-only \\\n      --department finance"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("IT Department Agent (Hybrid processing):")
    para = generator.doc.add_paragraph()
    para.text = "    # Create IT agent with technical capabilities\n    clawdbot agent create it \\\n      --model codellama:7b \\\n      --security-level internal \\\n      --hybrid-routing \\\n      --department it"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Verify agent configuration:")
    para = generator.doc.add_paragraph()
    para.text = "    # List configured agents\n    clawdbot agent list\n    \n    # Test agent response\n    clawdbot agent test hr \"Hello from HR department\""
    para.runs[0].font.name = 'Courier New'
    
    # Section 8: Install n8n
    generator.add_section("8. Install n8n Workflow Automation", "")
    
    generator.doc.add_paragraph("Install n8n for workflow automation:")
    
    para = generator.doc.add_paragraph()
    para.text = "    # Install n8n globally\n    npm install -g n8n\n    \n    # Verify installation\n    n8n --version"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Alternative: Docker deployment (recommended for production):")
    para = generator.doc.add_paragraph()
    para.text = "    # Create n8n directory\n    mkdir -p /opt/n8n/data\n    \n    # Run n8n with Docker\n    docker run -d \\\n      --name n8n \\\n      -p 5678:5678 \\\n      -e N8N_ENCRYPTION_KEY=\"your_encryption_key\" \\\n      -v /opt/n8n/data:/home/node/.n8n \\\n      n8nio/n8n"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Configure PostgreSQL database for n8n:")
    para = generator.doc.add_paragraph()
    para.text = "    # Start PostgreSQL container\n    docker run -d \\\n      --name n8n-postgres \\\n      -e POSTGRES_USER=n8n \\\n      -e POSTGRES_PASSWORD=your_password \\\n      -e POSTGRES_DB=n8n \\\n      -v postgres_data:/var/lib/postgresql/data \\\n      postgres:15"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Start n8n with database:")
    para = generator.doc.add_paragraph()
    para.text = "    # Run n8n with PostgreSQL\n    docker run -d \\\n      --name n8n \\\n      --link n8n-postgres \\\n      -p 5678:5678 \\\n      -e DB_TYPE=postgresdb \\\n      -e DB_POSTGRESDB_HOST=n8n-postgres \\\n      -e DB_POSTGRESDB_USER=n8n \\\n      -e DB_POSTGRESDB_PASSWORD=your_password \\\n      -e DB_POSTGRESDB_DATABASE=n8n \\\n      -v /opt/n8n/data:/home/node/.n8n \\\n      n8nio/n8n"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Verify n8n installation:")
    para = generator.doc.add_paragraph()
    para.text = "    # Check n8n is running\n    curl http://localhost:5678/healthz\n    \n    # Check Docker containers\n    docker ps | grep n8n"
    para.runs[0].font.name = 'Courier New'
    
    # Section 9: Configure n8n Integration  
    generator.add_section("9. Configure n8n Integration with ClawdBot", "")
    
    generator.doc.add_paragraph("Install ClawdBot n8n integration:")
    para = generator.doc.add_paragraph()
    para.text = "    # Install ClawdBot nodes for n8n\n    npm install -g n8n-nodes-clawdbot\n    \n    # Restart n8n to load new nodes\n    docker restart n8n"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Configure ClawdBot API connection:")
    generator.doc.add_paragraph("1. Open n8n web interface: http://localhost:5678")
    generator.doc.add_paragraph("2. Go to Settings → Credentials")
    generator.doc.add_paragraph("3. Add ClawdBot API credentials:")
    
    para = generator.doc.add_paragraph()
    para.text = "    API URL: http://localhost:3000/api/v1\n    API Key: your_clawdbot_api_key\n    Timeout: 30000ms"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Test integration:")
    para = generator.doc.add_paragraph()
    para.text = "    # Test API connection from command line\n    curl -H \"Authorization: Bearer your_api_key\" \\\n         http://localhost:3000/api/v1/health"
    para.runs[0].font.name = 'Courier New'
    
    # Section 10: Security Configuration
    generator.add_section("10. Security Configuration", "")
    
    generator.doc.add_paragraph("Configure SSL/TLS certificates:")
    para = generator.doc.add_paragraph()
    para.text = "    # Generate self-signed certificates for development\n    openssl req -x509 -newkey rsa:4096 \\\n      -keyout /opt/clawdbot/ssl/key.pem \\\n      -out /opt/clawdbot/ssl/cert.pem \\\n      -days 365 -nodes\n    \n    # Set proper permissions\n    chmod 600 /opt/clawdbot/ssl/key.pem\n    chmod 644 /opt/clawdbot/ssl/cert.pem"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Configure firewall rules:")
    para = generator.doc.add_paragraph()
    para.text = "    # Allow necessary ports\n    sudo ufw allow 3000    # ClawdBot API\n    sudo ufw allow 5678    # n8n Web Interface  \n    sudo ufw allow 11434   # Ollama API\n    sudo ufw allow 443     # HTTPS\n    \n    # Enable firewall\n    sudo ufw enable"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Set up authentication:")
    para = generator.doc.add_paragraph()
    para.text = "    # Configure LDAP integration\n    clawdbot auth configure \\\n      --provider ldap \\\n      --server ldap://your-ldap-server \\\n      --base-dn \"dc=company,dc=com\"\n    \n    # Test authentication\n    clawdbot auth test --username testuser"
    para.runs[0].font.name = 'Courier New'
    
    # Section 11: Testing and Validation
    generator.add_section("11. Testing and Validation", "")
    
    generator.doc.add_paragraph("Run comprehensive system tests:")
    
    generator.doc.add_paragraph("Test Ollama models:")
    para = generator.doc.add_paragraph()
    para.text = "    # Test all deployed models\n    ollama run llama3.1:8b \"Summarize the benefits of local LLM processing\"\n    ollama run codellama:7b \"Write a Python function to calculate fibonacci\"\n    ollama run mistral:7b \"What is artificial intelligence?\""
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Test ClawdBot agents:")
    para = generator.doc.add_paragraph()
    para.text = "    # Test each department agent\n    clawdbot agent test hr \"What are our vacation policies?\"\n    clawdbot agent test finance \"Show me quarterly budget status\"\n    clawdbot agent test it \"How do I reset my password?\""
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Test n8n workflows:")
    para = generator.doc.add_paragraph()
    para.text = "    # Create test workflow via API\n    curl -X POST http://localhost:5678/rest/workflows \\\n      -H \"Content-Type: application/json\" \\\n      -d '{\"name\": \"Test Workflow\", \"active\": true}'\n    \n    # Execute test workflow\n    curl -X POST http://localhost:5678/rest/workflows/1/execute"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Validate system integration:")
    para = generator.doc.add_paragraph()
    para.text = "    # Check all services are running\n    systemctl status ollama\n    docker ps | grep -E \"(n8n|postgres)\"\n    clawdbot status\n    \n    # Test API endpoints\n    curl http://localhost:11434/api/tags    # Ollama\n    curl http://localhost:3000/api/v1/health    # ClawdBot  \n    curl http://localhost:5678/healthz    # n8n"
    para.runs[0].font.name = 'Courier New'
    
    # Section 12: Troubleshooting
    generator.add_section("12. Troubleshooting and Maintenance", "")
    
    generator.doc.add_paragraph("Common installation issues and solutions:")
    
    generator.doc.add_paragraph("Ollama service not starting:")
    para = generator.doc.add_paragraph()
    para.text = "    # Check service status\n    brew services list | grep ollama\n    \n    # Restart service\n    brew services restart ollama\n    \n    # Check logs\n    tail -f /opt/homebrew/var/log/ollama.log"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Memory issues with large models:")
    para = generator.doc.add_paragraph()
    para.text = "    # Check memory usage\n    ollama ps\n    \n    # Unload unused models\n    ollama rm llama3.1:70b\n    \n    # Optimize memory settings\n    export OLLAMA_MAX_LOADED_MODELS=2"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("n8n connection issues:")
    para = generator.doc.add_paragraph()
    para.text = "    # Check container logs\n    docker logs n8n\n    \n    # Restart n8n container\n    docker restart n8n\n    \n    # Verify network connectivity\n    docker network inspect bridge"
    para.runs[0].font.name = 'Courier New'
    
    generator.doc.add_paragraph("Performance monitoring:")
    para = generator.doc.add_paragraph()
    para.text = "    # Monitor system resources\n    top -o CPU\n    htop\n    \n    # Check disk usage\n    du -sh /opt/ollama/models/*\n    df -h\n    \n    # Network monitoring\n    netstat -tulpn | grep -E \"(3000|5678|11434)\""
    para.runs[0].font.name = 'Courier New'
    
    # Support section
    generator.add_section("Support and Resources", "", {
        "Enterprise Support": "enterprise-support@clawdbot.com | 24/7 critical issues",
        "Documentation": "https://docs.clawdbot.com/enterprise | Complete technical reference",
        "Community": "https://discord.gg/clawdbot | Community support and discussions",
        "Training": "https://training.clawdbot.com | Certification and training programs"
    })
    
    return generator

def main():
    """Generate the detailed system manual with step-by-step commands"""
    print("🚀 Generating Detailed ClawdBot System Configuration Manual...")
    
    generator = create_detailed_system_manual()
    filename = generator.save("ClawdBot_System_Configuration_Detailed.docx")
    
    print(f"✅ Detailed system manual created: {filename}")
    return filename

if __name__ == "__main__":
    main()