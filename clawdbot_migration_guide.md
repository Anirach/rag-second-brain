# ClawdBot Migration Guide
## Personal to Enterprise Multi-Agent Setup

### Overview
This guide covers migrating from a personal ClawdBot installation to an enterprise multi-agent setup without losing your existing configuration, knowledge base, or conversation history.

---

## Pre-Migration Assessment

### What You Can Migrate
✅ **Configuration settings**  
✅ **Knowledge base content**  
✅ **User preferences**  
✅ **API keys and authentication**  
✅ **Custom skills and tools**  
✅ **Channel integrations**  

### What Requires Reconfiguration
⚠️ **Multi-agent routing**  
⚠️ **Department-specific permissions**  
⚠️ **RBAC (Role-Based Access Control)**  
⚠️ **Advanced authentication (LDAP/SAML)**  
⚠️ **Enterprise monitoring**  

---

## Migration Process

### Step 1: Backup Current Setup

```bash
#!/bin/bash
# backup-personal-setup.sh

BACKUP_DATE=$(date +"%Y-%m-%d_%H-%M-%S")
BACKUP_DIR="$HOME/.clawdbot-backup-${BACKUP_DATE}"

echo "🔄 Backing up personal ClawdBot setup..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r ~/.clawdbot "$BACKUP_DIR/config"

# Backup any custom knowledge
if [ -d ~/Documents/ClawdBot ]; then
    cp -r ~/Documents/ClawdBot "$BACKUP_DIR/knowledge"
fi

# Backup conversation history
if [ -f ~/.clawdbot/conversations.db ]; then
    cp ~/.clawdbot/conversations.db "$BACKUP_DIR/"
fi

# Create backup manifest
cat > "$BACKUP_DIR/backup-info.json" << EOF
{
    "backup_date": "${BACKUP_DATE}",
    "clawdbot_version": "$(clawdbot --version)",
    "source": "personal_setup",
    "migration_target": "enterprise_multi_agent"
}
EOF

echo "✅ Backup completed: $BACKUP_DIR"
echo "📋 Keep this backup until migration is verified successful"
```

### Step 2: Install Enterprise ClawdBot

```bash
#!/bin/bash
# install-enterprise.sh

echo "🚀 Installing Enterprise ClawdBot..."

# Stop personal ClawdBot if running
clawdbot stop

# Create enterprise directory
sudo mkdir -p /Applications/ClawdBot
sudo chown $(whoami):staff /Applications/ClawdBot

# Update ClawdBot to latest version
npm update -g clawdbot@latest

# Initialize enterprise configuration
cd /Applications/ClawdBot
clawdbot init --mode=enterprise --template=multi-agent

echo "✅ Enterprise ClawdBot base installed"
```

### Step 3: Migration Script

```bash
#!/bin/bash
# migrate-to-enterprise.sh

PERSONAL_BACKUP="$1"
ENTERPRISE_HOME="/Applications/ClawdBot"

if [ -z "$PERSONAL_BACKUP" ]; then
    echo "Usage: $0 <backup-directory>"
    exit 1
fi

echo "🔄 Starting migration from personal to enterprise setup..."

# Function to migrate configuration
migrate_configuration() {
    echo "📝 Migrating configuration..."
    
    # Extract API keys and basic settings
    if [ -f "$PERSONAL_BACKUP/config/clawdbot.json" ]; then
        # Parse personal config
        ANTHROPIC_KEY=$(jq -r '.auth.profiles["anthropic:default"].api_key // empty' "$PERSONAL_BACKUP/config/clawdbot.json")
        OPENAI_KEY=$(jq -r '.auth.profiles["openai:default"].api_key // empty' "$PERSONAL_BACKUP/config/clawdbot.json")
        
        # Apply to enterprise config
        if [ ! -z "$ANTHROPIC_KEY" ] && [ "$ANTHROPIC_KEY" != "null" ]; then
            clawdbot config set auth.profiles.anthropic:default.api_key "$ANTHROPIC_KEY"
        fi
        
        if [ ! -z "$OPENAI_KEY" ] && [ "$OPENAI_KEY" != "null" ]; then
            clawdbot config set auth.profiles.openai:default.api_key "$OPENAI_KEY"
        fi
    fi
    
    echo "✅ Configuration migrated"
}

# Function to migrate knowledge base
migrate_knowledge() {
    echo "📚 Migrating knowledge base..."
    
    # Create shared knowledge directory
    mkdir -p "$ENTERPRISE_HOME/knowledge/shared"
    
    # Copy personal knowledge to shared
    if [ -d "$PERSONAL_BACKUP/knowledge" ]; then
        cp -r "$PERSONAL_BACKUP/knowledge/"* "$ENTERPRISE_HOME/knowledge/shared/"
        
        # Index shared knowledge
        clawdbot knowledge index --source="$ENTERPRISE_HOME/knowledge/shared" --target="shared"
    fi
    
    echo "✅ Knowledge base migrated to shared repository"
}

# Function to setup initial departments
setup_initial_departments() {
    echo "🏢 Setting up initial departments..."
    
    # Create basic departments (can customize later)
    departments=("general" "it" "admin")
    
    for dept in "${departments[@]}"; do
        echo "Setting up $dept department..."
        
        # Create agent directory
        mkdir -p "$ENTERPRISE_HOME/agents/$dept"
        
        # Create basic agent config
        cat > "$ENTERPRISE_HOME/agents/$dept/config.json" << EOF
{
    "agent_id": "${dept}_assistant",
    "display_name": "${dept^} Assistant",
    "personality": {
        "role": "${dept^} Assistant",
        "communication_style": "Professional and helpful",
        "expertise_domains": ["${dept}_operations", "general_assistance"]
    },
    "capabilities": {
        "knowledge_domains": ["shared", "${dept}"],
        "tools": ["document_search", "general_assistance"]
    },
    "security": {
        "access_level": "internal",
        "audit_logging": true
    }
}
EOF
        
        # Initialize agent
        clawdbot agents init "$dept" \
            --workspace="$ENTERPRISE_HOME/agents/$dept" \
            --knowledge="$ENTERPRISE_HOME/knowledge/shared"
    done
    
    echo "✅ Initial departments configured"
}

# Function to migrate channels
migrate_channels() {
    echo "📡 Migrating channel integrations..."
    
    if [ -f "$PERSONAL_BACKUP/config/clawdbot.json" ]; then
        # Extract Discord configuration
        DISCORD_TOKEN=$(jq -r '.channels.discord.botToken // empty' "$PERSONAL_BACKUP/config/clawdbot.json")
        if [ ! -z "$DISCORD_TOKEN" ] && [ "$DISCORD_TOKEN" != "null" ]; then
            clawdbot channels add discord --token="$DISCORD_TOKEN"
        fi
        
        # Extract Telegram configuration  
        TELEGRAM_TOKEN=$(jq -r '.channels.telegram.botToken // empty' "$PERSONAL_BACKUP/config/clawdbot.json")
        if [ ! -z "$TELEGRAM_TOKEN" ] && [ "$TELEGRAM_TOKEN" != "null" ]; then
            clawdbot channels add telegram --token="$TELEGRAM_TOKEN" --agent="general"
        fi
    fi
    
    echo "✅ Channels migrated"
}

# Function to create migration report
create_migration_report() {
    cat > "$ENTERPRISE_HOME/migration-report.md" << EOF
# ClawdBot Migration Report

## Migration Details
- **Date**: $(date)
- **Source**: Personal ClawdBot setup
- **Target**: Enterprise multi-agent setup
- **Backup Source**: $PERSONAL_BACKUP

## Migrated Components
- ✅ API keys and authentication
- ✅ Knowledge base → Shared repository
- ✅ Channel integrations
- ✅ Basic department setup (general, it, admin)

## Next Steps
1. **Customize Departments**: Add/modify departments based on your organization
2. **Configure RBAC**: Set up role-based access control
3. **Add Department Knowledge**: Upload department-specific documents
4. **Setup Authentication**: Configure LDAP/SAML if needed
5. **Configure Monitoring**: Set up enterprise monitoring and alerts

## Manual Configuration Required
- Department-specific knowledge bases
- Advanced authentication (LDAP/SAML)  
- Enterprise monitoring and alerting
- Advanced security policies
- Custom department agents

## Original Backup Location
Your original personal setup is backed up at: $PERSONAL_BACKUP
Keep this backup until you've verified the migration is successful.

## Testing Your Migration
Run these commands to test:
\`\`\`bash
# Test enterprise gateway
clawdbot doctor

# Test agents
clawdbot agents general test "Hello"
clawdbot agents it test "System status"

# Test knowledge search
clawdbot knowledge search --query="help" --department=shared
\`\`\`
EOF

    echo "📋 Migration report created: $ENTERPRISE_HOME/migration-report.md"
}

# Execute migration steps
migrate_configuration
migrate_knowledge  
setup_initial_departments
migrate_channels
create_migration_report

echo "🎉 Migration completed successfully!"
echo "📖 Check migration report: $ENTERPRISE_HOME/migration-report.md"
echo "🔍 Run 'clawdbot doctor' to verify everything is working"
```

### Step 4: Post-Migration Configuration

```bash
#!/bin/bash
# post-migration-setup.sh

ENTERPRISE_HOME="/Applications/ClawdBot"

echo "🔧 Post-migration enterprise configuration..."

# Configure enterprise gateway
cat > "$ENTERPRISE_HOME/configs/enterprise-gateway.json" << EOF
{
    "gateway": {
        "mode": "multi-agent",
        "port": 18789,
        "agents": {
            "routing": "intelligent",
            "fallback_agent": "general"
        }
    },
    "agents": {
        "defaults": {
            "workspace": "$ENTERPRISE_HOME/agents",
            "knowledge_base": "$ENTERPRISE_HOME/knowledge",
            "session_timeout": "4h"
        }
    }
}
EOF

# Start enterprise gateway
clawdbot gateway start --config="$ENTERPRISE_HOME/configs/enterprise-gateway.json"

echo "✅ Enterprise configuration applied"
```

---

## Department Expansion (After Migration)

### Adding New Departments

```bash
#!/bin/bash
# add-department.sh

DEPARTMENT="$1"
ENTERPRISE_HOME="/Applications/ClawdBot"

if [ -z "$DEPARTMENT" ]; then
    echo "Usage: $0 <department-name>"
    echo "Example: $0 hr"
    exit 1
fi

echo "🏢 Adding $DEPARTMENT department..."

# Create department structure
mkdir -p "$ENTERPRISE_HOME/knowledge/$DEPARTMENT"
mkdir -p "$ENTERPRISE_HOME/agents/$DEPARTMENT"

# Create department agent config
cat > "$ENTERPRISE_HOME/agents/$DEPARTMENT/config.json" << EOF
{
    "agent_id": "${DEPARTMENT}_assistant", 
    "display_name": "${DEPARTMENT^} Department Assistant",
    "personality": {
        "role": "${DEPARTMENT^} Specialist",
        "communication_style": "Professional, department-focused",
        "expertise_domains": ["${DEPARTMENT}_operations", "department_policies"]
    },
    "capabilities": {
        "knowledge_domains": ["shared", "$DEPARTMENT"],
        "tools": ["document_search", "${DEPARTMENT}_tools"]
    },
    "security": {
        "access_level": "${DEPARTMENT}_staff",
        "audit_logging": true
    }
}
EOF

# Initialize department agent
clawdbot agents init "$DEPARTMENT" \
    --workspace="$ENTERPRISE_HOME/agents/$DEPARTMENT" \
    --knowledge="$ENTERPRISE_HOME/knowledge/shared,$ENTERPRISE_HOME/knowledge/$DEPARTMENT"

echo "✅ $DEPARTMENT department added successfully"
echo "📝 Upload department-specific documents to: $ENTERPRISE_HOME/knowledge/$DEPARTMENT"
```

---

## Migration Verification Checklist

### ✅ **Immediate Testing**
- [ ] `clawdbot doctor` runs without errors
- [ ] Enterprise gateway starts successfully
- [ ] All migrated agents respond to test queries
- [ ] Knowledge search returns results
- [ ] Channel integrations work (Discord/Telegram)

### ✅ **Configuration Verification**  
- [ ] API keys migrated correctly
- [ ] Knowledge base accessible
- [ ] Agent routing works
- [ ] Conversation history preserved (if applicable)

### ✅ **Functionality Testing**
- [ ] Multi-agent routing
- [ ] Department-specific responses  
- [ ] Shared knowledge access
- [ ] Cross-agent communication
- [ ] Channel messaging works

---

## Rollback Procedure (If Needed)

If you need to rollback to your personal setup:

```bash
#!/bin/bash
# rollback-to-personal.sh

BACKUP_DIR="$1"

if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup-directory>"
    exit 1
fi

echo "🔄 Rolling back to personal ClawdBot setup..."

# Stop enterprise setup
clawdbot stop

# Restore personal configuration
rm -rf ~/.clawdbot
cp -r "$BACKUP_DIR/config" ~/.clawdbot

# Restore personal knowledge
if [ -d "$BACKUP_DIR/knowledge" ]; then
    mkdir -p ~/Documents/ClawdBot
    cp -r "$BACKUP_DIR/knowledge/"* ~/Documents/ClawdBot/
fi

# Restart personal ClawdBot
clawdbot start

echo "✅ Rollback completed. Personal setup restored."
```

---

## Cost Considerations

### **Personal Setup**
- Single agent instance
- Lower API usage
- Minimal infrastructure needs

### **Enterprise Setup**  
- Multiple agent instances
- Higher API usage (multiple agents)
- More infrastructure resources needed
- Enterprise features and monitoring

### **Gradual Scaling**
Start with 2-3 departments and expand based on:
- Usage patterns
- API costs
- User feedback
- Resource utilization

---

## Timeline Recommendations

### **Week 1**: Personal Setup
- Install and configure basic ClawdBot
- Upload initial knowledge base
- Test with small user group

### **Week 2-3**: Migration Planning
- Assess department needs
- Plan knowledge organization
- Prepare migration scripts

### **Week 4**: Migration Execution  
- Backup personal setup
- Execute migration
- Test enterprise functionality

### **Week 5+**: Department Expansion
- Add departments one by one
- Train users on new features
- Monitor and optimize

---

This migration approach lets you:
✅ Start simple and learn the system
✅ Migrate without losing your work  
✅ Scale gradually based on needs
✅ Minimize downtime and disruption