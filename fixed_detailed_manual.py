#!/usr/bin/env python3
"""
ClawdBot System Configuration Manual with Detailed Step-by-Step Commands
Professional format with clear command blocks and verification steps
"""

import sys
sys.path.append('/home/clawdbot/clawd/professional-docx-generator')
from scripts.docx_generator import ProfessionalDocumentGenerator
from datetime import datetime
from docx.shared import Pt

def add_code_block(generator, code_text):
    """Add a formatted code block"""
    para = generator.doc.add_paragraph(code_text)
    para.runs[0].font.name = 'Courier New'
    para.runs[0].font.size = Pt(10)
    return para

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
    
    # Section 1: Install Homebrew
    generator.add_section("1.1 Install Homebrew (Package Manager)", "")
    
    generator.doc.add_paragraph("Homebrew is required to install most development tools on macOS. Open Terminal and run:")
    
    add_code_block(generator, "/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
    
    generator.doc.add_paragraph("After installation, add Homebrew to your PATH:")
    
    add_code_block(generator, "echo 'eval \"$(/opt/homebrew/bin/brew shellenv)\"' >> ~/.zprofile\neval \"$(/opt/homebrew/bin/brew shellenv)\"")
    
    generator.doc.add_paragraph("Verify installation:")
    
    add_code_block(generator, "brew --version    # Should show Homebrew version")
    
    # Section 2: Install Node.js
    generator.add_section("1.2 Install Node.js (Version 22+)", "")
    
    generator.doc.add_paragraph("ClawdBot requires Node.js version 22 or higher:")
    
    add_code_block(generator, "# Install Node.js using Homebrew\nbrew install node@22")
    
    generator.doc.add_paragraph("Alternative: Use nvm (Node Version Manager) for flexibility:")
    
    add_code_block(generator, "# Install nvm\ncurl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash\nsource ~/.zshrc\n\n# Install and use Node.js 22\nnvm install 22\nnvm use 22")
    
    generator.doc.add_paragraph("Verify installation:")
    
    add_code_block(generator, "node --version    # Should show v22.x.x\nnpm --version     # Should show 10.x.x or higher")
    
    # Section 3: Install Docker
    generator.add_section("1.3 Install Docker Desktop", "")
    
    generator.doc.add_paragraph("Docker is required for n8n and database services:")
    
    generator.doc.add_paragraph("**Option 1: Download from Docker website**")
    generator.doc.add_paragraph("1. Visit https://docker.com/products/docker-desktop")
    generator.doc.add_paragraph("2. Download Docker Desktop for Mac (Apple Silicon)")
    generator.doc.add_paragraph("3. Open the downloaded .dmg file and drag Docker to Applications")
    generator.doc.add_paragraph("4. Launch Docker Desktop and complete setup wizard")
    generator.doc.add_paragraph("5. In Docker Settings → Resources, allocate at least 32GB RAM and 8 CPUs")
    
    generator.doc.add_paragraph("**Option 2: Install via Homebrew**")
    
    add_code_block(generator, "brew install --cask docker")
    
    generator.doc.add_paragraph("Verify installation:")
    
    add_code_block(generator, "docker --version    # Should show Docker version\ndocker ps           # Should show empty container list")
    
    # Section 4: Install Ollama
    generator.add_section("2.1 Install Ollama (Local LLM Platform)", "")
    
    generator.doc.add_paragraph("Ollama provides local LLM processing capabilities:")
    
    add_code_block(generator, "# Install Ollama\ncurl -fsSL https://ollama.ai/install.sh | sh")
    
    generator.doc.add_paragraph("Alternative installation via Homebrew:")
    
    add_code_block(generator, "brew install ollama")
    
    generator.doc.add_paragraph("Configure Ollama environment:")
    
    add_code_block(generator, "# Set environment variables\nexport OLLAMA_HOST=127.0.0.1:11434\nexport OLLAMA_MAX_LOADED_MODELS=4\n\n# Add to shell profile\necho 'export OLLAMA_HOST=127.0.0.1:11434' >> ~/.zprofile\necho 'export OLLAMA_MAX_LOADED_MODELS=4' >> ~/.zprofile")
    
    generator.doc.add_paragraph("Start Ollama service:")
    
    add_code_block(generator, "# Start Ollama service\nollama serve\n\n# Or run as background service (preferred)\nbrew services start ollama")
    
    generator.doc.add_paragraph("Verify installation:")
    
    add_code_block(generator, "ollama --version    # Should show Ollama version\ncurl http://localhost:11434/api/tags    # Should return JSON response")
    
    # Section 5: Deploy LLM Models
    generator.add_section("2.2 Deploy Local LLM Models", "")
    
    generator.doc.add_paragraph("Download essential LLM models for ClawdBot Enterprise:")
    
    add_code_block(generator, "# Llama 3.1 8B - General purpose (16GB RAM required)\nollama pull llama3.1:8b\n\n# Llama 3.1 70B - Complex analysis (45GB RAM required)\nollama pull llama3.1:70b\n\n# Code Llama 7B - Technical support (14GB RAM required)\nollama pull codellama:7b\n\n# Mistral 7B - Fast responses (12GB RAM required)\nollama pull mistral:7b")
    
    generator.doc.add_paragraph("Verify model deployment:")
    
    add_code_block(generator, "# List installed models\nollama list\n\n# Test model response\nollama run llama3.1:8b \"Hello, this is a test message\"")
    
    generator.doc.add_paragraph("Monitor system resources:")
    
    add_code_block(generator, "# Check running models and memory usage\nollama ps\n\n# Monitor overall system memory\ntop -o MEM | head -20")
    
    # Section 6: Install ClawdBot Enterprise
    generator.add_section("3.1 Install ClawdBot Enterprise", "")
    
    generator.doc.add_paragraph("Install the ClawdBot Enterprise platform:")
    
    add_code_block(generator, "# Install ClawdBot CLI globally\nnpm install -g clawdbot-enterprise\n\n# Verify installation\nclawdbot --version    # Should show ClawdBot version")
    
    generator.doc.add_paragraph("Initialize ClawdBot workspace:")
    
    add_code_block(generator, "# Create enterprise workspace directory\nsudo mkdir -p /opt/clawdbot-enterprise\nsudo chown $(whoami):staff /opt/clawdbot-enterprise\ncd /opt/clawdbot-enterprise\n\n# Initialize with enterprise configuration\nclawdbot init --enterprise --workspace .")
    
    generator.doc.add_paragraph("Configure enterprise license:")
    
    add_code_block(generator, "# Add enterprise license key\nclawdbot license add --key YOUR_ENTERPRISE_LICENSE_KEY\n\n# Verify license status\nclawdbot license verify    # Should show license as valid")
    
    # Section 7: Configure ClawdBot Agents
    generator.add_section("3.2 Configure Department Agents", "")
    
    generator.doc.add_paragraph("Create department-specific AI agents with appropriate security levels:")
    
    generator.doc.add_paragraph("**HR Department Agent (Confidential - Local Only):**")
    
    add_code_block(generator, "# Create HR agent with maximum security\nclawdbot agent create hr \\\n  --model llama3.1:8b \\\n  --security-level confidential \\\n  --local-only \\\n  --department hr \\\n  --description \"Employee relations and HR support\"")
    
    generator.doc.add_paragraph("**Finance Department Agent (Strict Security):**")
    
    add_code_block(generator, "# Create Finance agent with strict security\nclawdbot agent create finance \\\n  --model llama3.1:70b \\\n  --security-level strict \\\n  --local-only \\\n  --department finance \\\n  --description \"Financial analysis and reporting\"")
    
    generator.doc.add_paragraph("**IT Department Agent (Hybrid Processing):**")
    
    add_code_block(generator, "# Create IT agent with technical capabilities\nclawdbot agent create it \\\n  --model codellama:7b \\\n  --security-level internal \\\n  --hybrid-routing \\\n  --department it \\\n  --description \"Technical support and IT operations\"")
    
    generator.doc.add_paragraph("**Marketing Department Agent (Cloud Enabled):**")
    
    add_code_block(generator, "# Create Marketing agent with cloud capabilities\nclawdbot agent create marketing \\\n  --model llama3.1:8b \\\n  --security-level public \\\n  --cloud-enabled \\\n  --department marketing \\\n  --description \"Marketing content and campaign support\"")
    
    generator.doc.add_paragraph("Verify agent configuration:")
    
    add_code_block(generator, "# List all configured agents\nclawdbot agent list\n\n# Test agent responses\nclawdbot agent test hr \"What are our vacation policies?\"\nclawdbot agent test it \"How do I reset my password?\"")
    
    # Section 8: Install n8n
    generator.add_section("4.1 Install n8n Workflow Automation", "")
    
    generator.doc.add_paragraph("Install n8n for enterprise workflow automation:")
    
    add_code_block(generator, "# Install n8n globally\nnpm install -g n8n\n\n# Verify installation\nn8n --version    # Should show n8n version")
    
    generator.doc.add_paragraph("**Production Setup with Docker (Recommended):**")
    
    generator.doc.add_paragraph("Create PostgreSQL database for n8n:")
    
    add_code_block(generator, "# Create network for n8n services\ndocker network create n8n-network\n\n# Start PostgreSQL container\ndocker run -d \\\n  --name n8n-postgres \\\n  --network n8n-network \\\n  -e POSTGRES_USER=n8n \\\n  -e POSTGRES_PASSWORD=n8n_secure_password \\\n  -e POSTGRES_DB=n8n \\\n  -v n8n_postgres_data:/var/lib/postgresql/data \\\n  postgres:15")
    
    generator.doc.add_paragraph("Start n8n with PostgreSQL backend:")
    
    add_code_block(generator, "# Run n8n with enterprise configuration\ndocker run -d \\\n  --name n8n \\\n  --network n8n-network \\\n  -p 5678:5678 \\\n  -e N8N_ENCRYPTION_KEY=\"your_32_character_encryption_key\" \\\n  -e DB_TYPE=postgresdb \\\n  -e DB_POSTGRESDB_HOST=n8n-postgres \\\n  -e DB_POSTGRESDB_PORT=5432 \\\n  -e DB_POSTGRESDB_USER=n8n \\\n  -e DB_POSTGRESDB_PASSWORD=n8n_secure_password \\\n  -e DB_POSTGRESDB_DATABASE=n8n \\\n  -e N8N_BASIC_AUTH_ACTIVE=true \\\n  -e N8N_BASIC_AUTH_USER=admin \\\n  -e N8N_BASIC_AUTH_PASSWORD=your_admin_password \\\n  -v n8n_data:/home/node/.n8n \\\n  n8nio/n8n")
    
    generator.doc.add_paragraph("Verify n8n installation:")
    
    add_code_block(generator, "# Check n8n health\ncurl http://localhost:5678/healthz    # Should return OK\n\n# Check running containers\ndocker ps | grep n8n    # Should show n8n and postgres containers")
    
    # Section 9: Configure n8n Integration  
    generator.add_section("4.2 Configure n8n Integration with ClawdBot", "")
    
    generator.doc.add_paragraph("Install ClawdBot integration for n8n:")
    
    add_code_block(generator, "# Install ClawdBot nodes package\nnpm install -g n8n-nodes-clawdbot\n\n# Restart n8n to load new nodes\ndocker restart n8n")
    
    generator.doc.add_paragraph("Configure ClawdBot API connection in n8n:")
    generator.doc.add_paragraph("1. Open n8n web interface: http://localhost:5678")
    generator.doc.add_paragraph("2. Login with admin credentials configured above")
    generator.doc.add_paragraph("3. Go to Settings → Credentials → Add Credential")
    generator.doc.add_paragraph("4. Select 'ClawdBot API' and configure:")
    
    add_code_block(generator, "API URL: http://localhost:3000/api/v1\nAPI Key: your_clawdbot_api_key\nTimeout: 30000\nRetry Attempts: 3")
    
    generator.doc.add_paragraph("Test the integration:")
    
    add_code_block(generator, "# Test ClawdBot API connectivity\ncurl -H \"Authorization: Bearer your_api_key\" \\\n     http://localhost:3000/api/v1/health\n\n# Should return: {\"status\": \"ok\", \"version\": \"enterprise\"}")
    
    # Section 10: Security Configuration
    generator.add_section("5.1 Security Configuration", "")
    
    generator.doc.add_paragraph("Configure enterprise security framework:")
    
    generator.doc.add_paragraph("**Generate SSL/TLS certificates:**")
    
    add_code_block(generator, "# Create SSL directory\nsudo mkdir -p /opt/clawdbot/ssl\n\n# Generate self-signed certificate for development\nopenssl req -x509 -newkey rsa:4096 \\\n  -keyout /opt/clawdbot/ssl/key.pem \\\n  -out /opt/clawdbot/ssl/cert.pem \\\n  -days 365 -nodes \\\n  -subj \"/C=US/ST=State/L=City/O=Organization/CN=localhost\"\n\n# Set proper permissions\nsudo chmod 600 /opt/clawdbot/ssl/key.pem\nsudo chmod 644 /opt/clawdbot/ssl/cert.pem")
    
    generator.doc.add_paragraph("**Configure firewall rules:**")
    
    add_code_block(generator, "# Configure macOS firewall\nsudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on\n\n# Allow specific applications\nsudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /Applications/Docker.app\nsudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /opt/homebrew/bin/ollama\n\n# For Linux systems, use ufw:\n# sudo ufw allow 3000    # ClawdBot API\n# sudo ufw allow 5678    # n8n Interface\n# sudo ufw allow 11434   # Ollama API")
    
    generator.doc.add_paragraph("**Set up authentication system:**")
    
    add_code_block(generator, "# Configure LDAP authentication\nclawdbot auth configure \\\n  --provider ldap \\\n  --server \"ldap://your-ldap-server:389\" \\\n  --base-dn \"dc=company,dc=com\" \\\n  --bind-dn \"cn=admin,dc=company,dc=com\" \\\n  --bind-password \"your_ldap_password\"\n\n# Test authentication\nclawdbot auth test --username testuser --password testpass")
    
    # Section 11: Testing and Validation
    generator.add_section("6.1 Testing and Validation", "")
    
    generator.doc.add_paragraph("Execute comprehensive system validation:")
    
    generator.doc.add_paragraph("**Test Ollama LLM functionality:**")
    
    add_code_block(generator, "# Test each deployed model\nollama run llama3.1:8b \"Explain the benefits of local AI processing\"\nollama run codellama:7b \"Write a Python function to reverse a string\"\nollama run mistral:7b \"What is machine learning?\"")
    
    generator.doc.add_paragraph("**Validate ClawdBot agents:**")
    
    add_code_block(generator, "# Test department-specific agents\nclawdbot agent test hr \"What are our company holidays?\"\nclawdbot agent test finance \"Show quarterly revenue trends\"\nclawdbot agent test it \"How do I install new software?\"\nclawdbot agent test marketing \"Create a social media post idea\"")
    
    generator.doc.add_paragraph("**Verify n8n workflow system:**")
    
    add_code_block(generator, "# Test n8n API\ncurl -X GET http://localhost:5678/rest/workflows \\\n  -u admin:your_admin_password\n\n# Should return list of workflows (empty initially)")
    
    generator.doc.add_paragraph("**Complete system integration test:**")
    
    add_code_block(generator, "# Check all services are operational\nbrew services list | grep ollama    # Should show 'started'\ndocker ps | grep -E \"(n8n|postgres)\"    # Should show running containers\nclawdbot status    # Should show all agents as 'active'\n\n# Test API endpoints\ncurl http://localhost:11434/api/tags    # Ollama API\ncurl http://localhost:3000/api/v1/health    # ClawdBot API  \ncurl http://localhost:5678/healthz    # n8n Health Check")
    
    # Section 12: Troubleshooting
    generator.add_section("7.1 Troubleshooting and Maintenance", "")
    
    generator.doc.add_paragraph("Common issues and resolution procedures:")
    
    generator.doc.add_paragraph("**Ollama service issues:**")
    
    add_code_block(generator, "# Check Ollama service status\nbrew services list | grep ollama\n\n# Restart Ollama service\nbrew services restart ollama\n\n# Check Ollama logs\ntail -f /opt/homebrew/var/log/ollama.log\n\n# Verify Ollama is listening\nlsof -i :11434")
    
    generator.doc.add_paragraph("**Memory management for large models:**")
    
    add_code_block(generator, "# Check current model memory usage\nollama ps\n\n# Unload specific model to free memory\nollama stop llama3.1:70b\n\n# Set memory limits\nexport OLLAMA_MAX_LOADED_MODELS=2\nexport OLLAMA_KEEP_ALIVE=5m")
    
    generator.doc.add_paragraph("**n8n container problems:**")
    
    add_code_block(generator, "# Check n8n container logs\ndocker logs n8n\ndocker logs n8n-postgres\n\n# Restart n8n services\ndocker restart n8n n8n-postgres\n\n# Verify database connection\ndocker exec n8n-postgres psql -U n8n -d n8n -c \"\\l\"")
    
    generator.doc.add_paragraph("**ClawdBot agent issues:**")
    
    add_code_block(generator, "# Check agent status\nclawdbot agent status\n\n# Restart specific agent\nclawdbot agent restart hr\n\n# Check agent logs\nclawdbot logs --agent hr --lines 50\n\n# Reset agent configuration\nclawdbot agent reset hr --confirm")
    
    generator.doc.add_paragraph("**Performance monitoring commands:**")
    
    add_code_block(generator, "# Monitor system resources\ntop -o CPU    # CPU usage\ntop -o MEM    # Memory usage\ndf -h         # Disk usage\n\n# Check network ports\nnetstat -tulpn | grep -E \"(3000|5678|11434)\"\n\n# Monitor Docker resources\ndocker stats")
    
    # Maintenance section
    generator.add_section("7.2 Regular Maintenance Procedures", "")
    
    generator.doc.add_paragraph("**Daily maintenance tasks:**")
    
    add_code_block(generator, "# Check system health\nclawdbot health-check\nollama ps\ndocker ps --format \"table {{.Names}}\\t{{.Status}}\"\n\n# Review logs for errors\ntail -n 100 /opt/clawdbot/logs/error.log\ndocker logs n8n --since 24h | grep ERROR")
    
    generator.doc.add_paragraph("**Weekly maintenance tasks:**")
    
    add_code_block(generator, "# Update Ollama models\nollama pull llama3.1:8b\nollama pull llama3.1:70b\n\n# Clean up unused Docker resources\ndocker system prune -f\n\n# Update ClawdBot\nnpm update -g clawdbot-enterprise")
    
    generator.doc.add_paragraph("**Monthly maintenance tasks:**")
    
    add_code_block(generator, "# Full system backup\ntar -czf backup-$(date +%Y%m%d).tar.gz /opt/clawdbot-enterprise\ndocker exec n8n-postgres pg_dump -U n8n n8n > n8n-backup-$(date +%Y%m%d).sql\n\n# Security audit\nclawdbot security audit --full\n\n# Performance optimization\nclawdbot optimize --all")
    
    # Support section
    generator.add_section("Support and Resources", "", {
        "Enterprise Support": "enterprise-support@clawdbot.com | Response within 4 hours for critical issues",
        "Technical Documentation": "https://docs.clawdbot.com/enterprise | Complete API and configuration reference",
        "Community Forum": "https://community.clawdbot.com | User discussions and knowledge sharing",
        "Training Programs": "https://training.clawdbot.com | Certification courses for administrators",
        "Status Page": "https://status.clawdbot.com | System status and maintenance notifications"
    })
    
    return generator

def main():
    """Generate the detailed system manual with step-by-step commands"""
    print("🚀 Generating Detailed ClawdBot System Configuration Manual...")
    
    generator = create_detailed_system_manual()
    filename = generator.save("ClawdBot_System_Configuration_StepByStep.docx")
    
    print(f"✅ Detailed system manual with step-by-step commands created: {filename}")
    return filename

if __name__ == "__main__":
    main()