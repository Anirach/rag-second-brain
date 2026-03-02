const pptxgen = require("pptxgenjs");
const BG = "0D1229", CARD_BG = "1E2341", BLUE = "60A5FA", PURPLE = "A78BFA", GREEN = "4ADE80", ORANGE = "FBBF24", CYAN = "22D3EE", PINK = "F472B6", RED = "F87171", GOLD = "FBBF24", GRAY = "8B95B0", LIGHT = "B0B8D0", WHITE = "FFFFFF";

function addBadge(s,t,x,y,c){c=c||"22C55E";s.addShape("rect",{x,y,w:1.8,h:0.32,fill:{color:c},rectRadius:0.05});s.addText(t,{x,y,w:1.8,h:0.32,fontSize:9,fontFace:"Arial",bold:true,color:WHITE,align:"center",valign:"middle",letterSpacing:1.5});}
function addTagline(s,t,y){y=y||4.95;s.addShape("rect",{x:0.4,y,w:9.2,h:0.4,fill:{color:"141E32"},line:{color:"2A3560",width:0.5},rectRadius:0.05});s.addText(t,{x:0.4,y,w:9.2,h:0.4,fontSize:11,fontFace:"Arial",bold:true,color:GOLD,align:"center",valign:"middle"});}
function addCard(s,title,body,x,y,w,h,ac){s.addShape("rect",{x,y,w,h,fill:{color:CARD_BG},rectRadius:0.08});s.addShape("rect",{x,y,w:0.06,h,fill:{color:ac||BLUE}});s.addText(title,{x:x+0.15,y,w:w-0.2,h:0.35,fontSize:13,fontFace:"Arial",bold:true,color:ac||BLUE,valign:"top",margin:[4,0,0,0]});s.addText(body,{x:x+0.15,y:y+0.32,w:w-0.2,h:h-0.36,fontSize:10,fontFace:"Arial",color:LIGHT,valign:"top",lineSpacingMultiple:1.3});}
function addNumberedItem(s,n,title,desc,x,y,w,cc,tc){s.addShape("rect",{x,y,w,h:0.55,fill:{color:CARD_BG},rectRadius:0.06});s.addShape("oval",{x:x+0.1,y:y+0.1,w:0.35,h:0.35,fill:{color:cc}});s.addText(String(n),{x:x+0.1,y:y+0.1,w:0.35,h:0.35,fontSize:11,fontFace:"Arial",bold:true,color:WHITE,align:"center",valign:"middle"});s.addText(title,{x:x+0.55,y:y+0.05,w:w-0.65,h:0.22,fontSize:11,fontFace:"Arial",bold:true,color:tc||BLUE});s.addText(desc,{x:x+0.55,y:y+0.27,w:w-0.65,h:0.23,fontSize:9,fontFace:"Arial",color:LIGHT});}

function titleSlide(p,t,sub,wk){let s=p.addSlide();s.background={color:BG};s.addShape("oval",{x:2.5,y:-0.5,w:3,h:3,fill:{color:"8B5CF6",transparency:88}});s.addShape("oval",{x:6,y:3.5,w:2.5,h:2.5,fill:{color:"3B82F6",transparency:88}});s.addText("DEVOPS WITH VIBECODING",{x:0,y:1.2,w:10,h:0.35,fontSize:11,fontFace:"Arial",color:"8B5CF6",align:"center",charSpacing:4});s.addText(t,{x:0.5,y:1.7,w:9,h:0.6,fontSize:28,fontFace:"Arial",bold:true,color:WHITE,align:"center"});s.addText(sub,{x:1,y:2.4,w:8,h:0.4,fontSize:16,fontFace:"Arial",color:BLUE,align:"center"});s.addText(wk,{x:3.5,y:3.1,w:3,h:0.35,fontSize:12,fontFace:"Arial",color:GRAY,align:"center"});s.addText("Anirach Mingkhwan",{x:2,y:3.6,w:6,h:0.3,fontSize:11,fontFace:"Arial",color:GRAY,align:"center"});s.addText("FITM, KMUTNB",{x:2,y:3.9,w:6,h:0.3,fontSize:10,fontFace:"Arial",color:GRAY,align:"center"});return s;}
function sectionSlide(p,num,t,sub){let s=p.addSlide();s.background={color:BG};s.addShape("rect",{x:0,y:0,w:10,h:5.63,fill:{color:"111936"}});s.addShape("oval",{x:-1,y:1,w:4,h:4,fill:{color:"8B5CF6",transparency:92}});s.addShape("oval",{x:7,y:-0.5,w:3,h:3,fill:{color:"3B82F6",transparency:92}});s.addText(num,{x:3.5,y:1.5,w:3,h:0.5,fontSize:14,fontFace:"Arial",color:PURPLE,align:"center",charSpacing:3});s.addText(t,{x:1,y:2.1,w:8,h:0.6,fontSize:28,fontFace:"Arial",bold:true,color:WHITE,align:"center"});s.addText(sub||"",{x:1.5,y:2.8,w:7,h:0.4,fontSize:14,fontFace:"Arial",color:BLUE,align:"center"});return s;}
function contentSlide(p,t){let s=p.addSlide();s.background={color:BG};s.addShape("rect",{x:0,y:0,w:10,h:0.9,fill:{color:"111936"}});s.addShape("rect",{x:0,y:0.88,w:10,h:0.03,fill:{color:PURPLE,transparency:50}});s.addText(t,{x:0.5,y:0.15,w:9,h:0.6,fontSize:20,fontFace:"Arial",bold:true,color:WHITE});return s;}

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";

// 1: Title
titleSlide(pres, "DevSecOps +\nAI Code Review", "Security-First Development Pipeline", "Week 8");

// 2: Learning Objectives
let s = contentSlide(pres, "Learning Objectives");
addNumberedItem(s, 1, "Shift Left Security", "Integrate security throughout the entire pipeline", 0.5, 1.1, 9, RED, RED);
addNumberedItem(s, 2, "Security Scanning", "Implement SAST, DAST, SCA, and secrets scanning in CI/CD", 0.5, 1.75, 9, PURPLE, PURPLE);
addNumberedItem(s, 3, "AI Code Review", "Use AI for automated security vulnerability detection", 0.5, 2.4, 9, BLUE, BLUE);
addNumberedItem(s, 4, "Supply Chain Security", "Implement SBOM generation and image signing", 0.5, 3.05, 9, GREEN, GREEN);
addNumberedItem(s, 5, "AI Code Safety", "Develop security-first mindset for AI-generated code", 0.5, 3.7, 9, ORANGE, ORANGE);
addTagline(s, '"Security is not a feature, it\'s a process" - DevSecOps Principle');

// 3: Agenda
s = contentSlide(pres, "Today's Agenda");
addCard(s, "Part 1: Shift Left", "DevSecOps Philosophy\nSecurity at Every Stage\nTool Landscape", 0.5, 1.1, 4.3, 1.5, RED);
addCard(s, "Part 2: Scanning Tools", "SAST, DAST, SCA\nSecrets Detection\nContainer Scanning", 5.2, 1.1, 4.3, 1.5, PURPLE);
addCard(s, "Part 3: AI Review", "AI Code Review Workflow\nStrengths & Limitations\nAI-Generated Code Risks", 0.5, 2.8, 4.3, 1.5, BLUE);
addCard(s, "Part 4: Supply Chain", "SBOM, Image Signing\nHands-on Lab\nSecurity Pipeline", 5.2, 2.8, 4.3, 1.5, GREEN);

// === SECTION 1: SHIFT LEFT ===
sectionSlide(pres, "SECTION 01", "Shifting Security Left", "From Afterthought to Built-In");

// 5: DevSecOps Pipeline
s = contentSlide(pres, "Security at Every Stage");
addNumberedItem(s, 1, "Planning", "Threat modeling, security requirements, risk assessment", 0.5, 1.1, 9, PURPLE, PURPLE);
addNumberedItem(s, 2, "Coding", "Secure coding standards, IDE plugins, pre-commit hooks", 0.5, 1.75, 9, BLUE, BLUE);
addNumberedItem(s, 3, "Building", "SAST, SCA, secrets scanning in CI pipeline", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "Testing", "DAST, fuzzing, penetration testing", 0.5, 3.05, 9, ORANGE, ORANGE);
addNumberedItem(s, 5, "Deploy & Operate", "Image scanning, runtime protection, WAF, RASP", 0.5, 3.7, 9, RED, RED);

// 6: Traditional vs DevSecOps
s = contentSlide(pres, "Traditional Security vs DevSecOps");
addCard(s, "Traditional", "- Security team reviews at the end\n- Bottleneck before release\n- Expensive to fix (found late)\n- Adversarial relationship\n- Manual, slow, inconsistent", 0.5, 1.1, 4.3, 2.3, RED);
addCard(s, "DevSecOps", "- Security embedded in every stage\n- Automated in CI/CD pipeline\n- Cheap to fix (found early)\n- Collaborative culture\n- Fast, consistent, scalable", 5.2, 1.1, 4.3, 2.3, GREEN);
addTagline(s, "Finding a bug in production costs 100x more than in development");

// 7: Cost of Late Detection
s = contentSlide(pres, "The Cost of Finding Bugs Late");
addCard(s, "Cost Multiplier by Stage", "Design:        1x\nCoding:         5x\nTesting:       10x\nStaging:       50x\nProduction:   100x\nPost-breach: 1000x+", 0.5, 1.1, 4.3, 2.5, RED);
addCard(s, "Real-World Impact", "Equifax (2017): $1.4B - unpatched Apache Struts\nSolarWinds (2020): $100M+ - supply chain\nLog4Shell (2021): Global impact - dependency\nMOVEit (2023): 2500+ orgs - SQL injection\n\nAll preventable with DevSecOps practices", 5.2, 1.1, 4.3, 2.5, ORANGE);
addTagline(s, "Shift left = shift savings");

// === SECTION 2: SCANNING TOOLS ===
sectionSlide(pres, "SECTION 02", "Security Scanning Tools", "SAST, DAST, SCA & More");

// 9: Tool Landscape
s = contentSlide(pres, "Security Scanning Landscape");
addCard(s, "SAST", "Static Application Security Testing\nAnalyzes source code without running\nSemgrep, CodeQL, Bandit\nFinds: injection, XSS, hardcoded secrets", 0.5, 1.1, 4.3, 1.7, PURPLE);
addCard(s, "DAST", "Dynamic Application Security Testing\nTests running application\nOWASP ZAP, Burp Suite\nFinds: runtime vulns, misconfigs", 5.2, 1.1, 4.3, 1.7, BLUE);
addCard(s, "SCA", "Software Composition Analysis\nScans dependencies for CVEs\nSnyk, pip-audit, npm audit\nFinds: known vulnerabilities in libs", 0.5, 3.0, 4.3, 1.4, GREEN);
addCard(s, "Secrets & Container", "gitleaks, detect-secrets\nTrivy, Grype for containers\nCheckov, tfsec for IaC\nFinds: leaked creds, image vulns", 5.2, 3.0, 4.3, 1.4, ORANGE);

// 10: Semgrep SAST
s = contentSlide(pres, "Semgrep: Modern SAST");
addCard(s, "What is Semgrep?", "Open-source static analysis tool\nPattern-based: write rules like code\nSupports 30+ languages\n10,000+ community rules\nFast: scans large repos in seconds", 0.5, 1.1, 4.3, 2.0, PURPLE);
addCard(s, "Example Rule", "rules:\n- id: sql-injection\n  pattern: |\n    cursor.execute(f\"...{$VAR}...\")\n  message: Possible SQL injection\n  severity: ERROR\n  languages: [python]", 5.2, 1.1, 4.3, 2.0, BLUE);
addCard(s, "CI Integration", "# GitHub Actions\n- uses: returntocorp/semgrep-action@v1\n  with:\n    config: p/python p/security-audit\n    generateSarif: true", 0.5, 3.3, 9, 1.2, GREEN);

// 11: CodeQL
s = contentSlide(pres, "GitHub CodeQL");
addCard(s, "Deep Semantic Analysis", "- Treats code as a database\n- Complex taint tracking queries\n- Data flow analysis across functions\n- GitHub-native integration\n- Free for open source projects", 0.5, 1.1, 4.3, 2.0, BLUE);
addCard(s, "Semgrep vs CodeQL", "Semgrep: Fast, pattern-based, easy rules\n  Best for: quick scans, custom rules\n\nCodeQL: Deep analysis, data flow tracking\n  Best for: complex vulnerabilities\n\nUse both for defense in depth", 5.2, 1.1, 4.3, 2.0, PURPLE);
addTagline(s, "Semgrep for speed, CodeQL for depth - use both in your pipeline");

// 12: SCA - Dependency Scanning
s = contentSlide(pres, "Dependency Scanning (SCA)");
addCard(s, "Why It Matters", "70-90% of code is open-source dependencies\nNew CVEs discovered daily\nTransitive dependencies are hidden risks\nLog4Shell proved the danger", 0.5, 1.1, 4.3, 1.7, RED);
addCard(s, "Tools", "pip-audit: Python packages\nnpm audit: Node.js packages\nSnyk: Multi-language, fix PRs\nDependabot: GitHub-native updates\nGrype: Container + filesystem scan", 5.2, 1.1, 4.3, 1.7, GREEN);
addCard(s, "Best Practices", "- Pin dependency versions (lock files)\n- Audit regularly (weekly CI job)\n- Monitor CVE databases\n- Evaluate dependency health (maintenance, popularity)\n- Have a remediation SLA: critical=24h, high=1wk", 0.5, 3.0, 9, 1.4, BLUE);

// 13: Secrets Detection
s = contentSlide(pres, "Secrets Detection");
addCard(s, "The Problem", "GitHub found 10M+ secrets in public repos (2023)\nAPI keys, passwords, tokens, certificates\nBots scan GitHub in real-time\nOne leaked AWS key = $1000s in minutes", 0.5, 1.1, 4.3, 1.7, RED);
addCard(s, "Tools & Prevention", "gitleaks: Git history scanning\ndetect-secrets: Baseline approach\nGitHub Secret Scanning: Built-in\npre-commit hooks: Prevent commit\nGit-secrets (AWS): AWS-specific", 5.2, 1.1, 4.3, 1.7, PURPLE);
addCard(s, "Pre-commit Hook", "# .pre-commit-config.yaml\nrepos:\n- repo: https://github.com/gitleaks/gitleaks\n  hooks:\n  - id: gitleaks", 0.5, 3.0, 9, 1.3, GREEN);
addTagline(s, "If a secret touches Git history, consider it compromised");

// 14: Container Scanning
s = contentSlide(pres, "Container Security with Trivy");
addCard(s, "What Trivy Scans", "- OS packages (CVEs)\n- Language dependencies\n- IaC misconfigurations\n- Kubernetes manifests\n- Dockerfile best practices\n- Secrets in images", 0.5, 1.1, 4.3, 2.3, BLUE);
addCard(s, "CI Pipeline Example", "# Scan and fail on critical\ntrivy image --severity CRITICAL \\\n  --exit-code 1 myapp:latest\n\n# Full report\ntrivy image --format json \\\n  --output trivy-report.json \\\n  myapp:latest", 5.2, 1.1, 4.3, 2.3, PURPLE);
addTagline(s, "Never deploy an image you haven't scanned");

// === SECTION 3: AI CODE REVIEW ===
sectionSlide(pres, "SECTION 03", "AI-Powered Code Review", "Augmenting Human Reviewers");

// 16: AI Review Workflow
s = contentSlide(pres, "AI Code Review Workflow");
addNumberedItem(s, 1, "Developer Commits", "Code pushed to feature branch, PR created", 0.5, 1.1, 9, BLUE, BLUE);
addNumberedItem(s, 2, "AI First Pass (CI)", "Automated security scan: patterns, anti-patterns, vulns", 0.5, 1.75, 9, PURPLE, PURPLE);
addNumberedItem(s, 3, "Developer Fixes", "Address AI findings, re-push", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "Human Review", "Architecture, logic, business rules, edge cases", 0.5, 3.05, 9, ORANGE, ORANGE);
addNumberedItem(s, 5, "Merge & Deploy", "All checks pass, code merged to main", 0.5, 3.7, 9, CYAN, CYAN);

// 17: What AI Catches
s = contentSlide(pres, "AI Security Review: Strengths");
addCard(s, "AI Catches Well", "- SQL injection patterns\n- Cross-site scripting (XSS)\n- Insecure deserialization\n- Hardcoded credentials\n- Weak cryptography (MD5, SHA1)\n- Path traversal\n- Command injection\n- Missing input validation", 0.5, 1.1, 4.3, 2.8, GREEN);
addCard(s, "Why AI Excels Here", "- Consistent: never tired or distracted\n- Fast: scans entire PR in seconds\n- Pattern library: trained on millions of vulns\n- Always up-to-date vulnerability DB\n- Instant feedback loop\n- Scales to any team size", 5.2, 1.1, 4.3, 2.8, BLUE);
addTagline(s, "AI handles the known patterns so humans can focus on logic");

// 18: What AI Misses
s = contentSlide(pres, "AI Security Review: Limitations");
addCard(s, "AI Misses", "- Business logic flaws\n- Authorization & access control issues\n- Novel/zero-day vulnerability patterns\n- Complex multi-step attacks\n- Race conditions\n- Timing side channels\n- Social engineering vectors", 0.5, 1.1, 4.3, 2.3, RED);
addCard(s, "Human Reviewers Add", "- Domain context understanding\n- Threat modeling perspective\n- Architecture-level security\n- Edge case identification\n- Security design patterns\n- Risk assessment judgment", 5.2, 1.1, 4.3, 2.3, PURPLE);
addTagline(s, "AI + Human = defense in depth for code review");

// 19: AI-Generated Code Risks
s = contentSlide(pres, "Security Risks of AI-Generated Code");
addCard(s, "Common AI Code Vulnerabilities", "- SQL injection (string concatenation)\n- Path traversal (unsanitized input)\n- Insecure defaults (debug=True)\n- Weak cryptography (MD5 for passwords)\n- Missing input validation\n- Hardcoded test credentials\n- Overly permissive CORS/permissions", 0.5, 1.1, 4.3, 2.5, RED);
addCard(s, "Mitigations", "- Always run SAST on AI-generated code\n- Provide security context in prompts\n- Maintain a security checklist\n- Train team on common AI patterns\n- Never trust AI output blindly\n- Review ALL generated code\n- Use secure coding templates", 5.2, 1.1, 4.3, 2.5, GREEN);
addTagline(s, '"AI Security Debt" - the growing attack surface from unreviewed AI code');

// 20: Prompting for Secure Code
s = contentSlide(pres, "VibeCoding: Prompting for Security");
addCard(s, "Bad Prompt", '"Write a login endpoint in Flask"\n\nResult: No rate limiting, plain text\npasswords, SQL injection, no CSRF,\nno input validation, debug mode on', 0.5, 1.1, 4.3, 1.7, RED);
addCard(s, "Good Prompt", '"Write a secure login endpoint in Flask\nwith bcrypt password hashing, rate\nlimiting (5/min), CSRF protection,\ninput validation, parameterized queries,\nand proper error handling. Follow\nOWASP Top 10 guidelines."', 5.2, 1.1, 4.3, 1.7, GREEN);
addCard(s, "Security-Aware Prompting Tips", "- Specify security requirements explicitly\n- Reference OWASP guidelines\n- Ask for input validation and error handling\n- Request parameterized queries, not string concat\n- Ask AI to explain its security choices", 0.5, 3.0, 9, 1.4, BLUE);

// === SECTION 4: SUPPLY CHAIN ===
sectionSlide(pres, "SECTION 04", "Supply Chain Security", "SBOM, Signing & Verification");

// 22: SBOM
s = contentSlide(pres, "Software Bill of Materials (SBOM)");
addCard(s, "What is SBOM?", "Complete inventory of software components\nLike ingredient list for software\nFormats: SPDX (Linux Foundation)\n         CycloneDX (OWASP)\nRequired by US Executive Order 14028", 0.5, 1.1, 4.3, 2.0, BLUE);
addCard(s, "Why SBOM Matters", "- Know what's in your software\n- Rapid vulnerability response\n- License compliance\n- Supply chain transparency\n- Regulatory requirements\n- Customer trust", 5.2, 1.1, 4.3, 2.0, PURPLE);
addCard(s, "Generate with syft", "# Generate SBOM for container image\nsyft myapp:latest -o spdx-json > sbom.json\n\n# Generate for source directory\nsyft dir:./src -o cyclonedx-json > sbom.json", 0.5, 3.3, 9, 1.2, GREEN);

// 23: Image Signing
s = contentSlide(pres, "Container Image Signing (cosign)");
addCard(s, "Sigstore / cosign", "Keyless signing with OIDC identity\nSign container images cryptographically\nVerify image authenticity before deploy\nTransparency log (Rekor) for audit", 0.5, 1.1, 4.3, 1.7, BLUE);
addCard(s, "Usage", "# Sign an image\ncosign sign myregistry/myapp:latest\n\n# Verify before deploy\ncosign verify myregistry/myapp:latest \\\n  --certificate-identity=dev@company.com \\\n  --certificate-oidc-issuer=https://github.com/login/oauth", 5.2, 1.1, 4.3, 1.7, PURPLE);
addCard(s, "Policy Enforcement", "Kyverno / OPA Gatekeeper:\n- Only allow signed images in cluster\n- Verify SBOM attestations\n- Enforce minimum scan results\n- Block images with critical CVEs", 0.5, 3.0, 9, 1.4, GREEN);
addTagline(s, "Sign everything, verify everything, trust nothing");

// 24: Dependency Management
s = contentSlide(pres, "Dependency Management Best Practices");
addNumberedItem(s, 1, "Pin Versions", "Use lock files (poetry.lock, package-lock.json) - reproducible builds", 0.5, 1.1, 9, BLUE, BLUE);
addNumberedItem(s, 2, "Audit Regularly", "Weekly CI job: pip-audit, npm audit - catch new CVEs early", 0.5, 1.75, 9, GREEN, GREEN);
addNumberedItem(s, 3, "Monitor CVEs", "Subscribe to security advisories - know before attackers do", 0.5, 2.4, 9, PURPLE, PURPLE);
addNumberedItem(s, 4, "Evaluate Health", "Check maintenance activity, bus factor, popularity - avoid abandoned libs", 0.5, 3.05, 9, ORANGE, ORANGE);
addNumberedItem(s, 5, "Remediation SLA", "Critical: 24h, High: 1 week, Medium: 1 month, Low: next release", 0.5, 3.7, 9, RED, RED);

// === SECTION 5: FULL PIPELINE ===
sectionSlide(pres, "SECTION 05", "Complete Security Pipeline", "End-to-End Automation");

// 26: Full Pipeline
s = contentSlide(pres, "DevSecOps Pipeline Architecture");
addCard(s, "Pre-Commit", "Secrets scanning (gitleaks)\nLinting & formatting\nLocal SAST quick scan", 0.5, 1.1, 2.8, 1.4, PURPLE);
addCard(s, "CI Pipeline", "SAST (Semgrep + CodeQL)\nSCA (pip-audit / npm audit)\nContainer scan (Trivy)\nIaC scan (Checkov / tfsec)", 3.6, 1.1, 2.8, 1.4, BLUE);
addCard(s, "Deploy Gate", "Image signature verification\nPolicy enforcement (Kyverno)\nSBOM attestation check\nZero critical vulns policy", 6.7, 1.1, 2.8, 1.4, GREEN);
addCard(s, "Runtime & Continuous", "WAF (ModSecurity / Cloudflare)\nRASP (runtime protection)\nScheduled scans & pen testing\nIncident response automation", 0.5, 2.7, 4.3, 1.6, ORANGE);
addCard(s, "Compliance Evidence", "CI/CD logs = audit trail\nSOC 2, ISO 27001, HIPAA, PCI-DSS\nAutomated compliance reports\nContinuous assurance", 5.2, 2.7, 4.3, 1.6, RED);

// 27: GitHub Actions Example
s = contentSlide(pres, "Full Security Pipeline: GitHub Actions");
addCard(s, "security-scan.yml", "name: Security Pipeline\non: [push, pull_request]\njobs:\n  sast:\n    steps:\n    - uses: returntocorp/semgrep-action@v1\n  sca:\n    steps:\n    - run: pip-audit --strict\n  secrets:\n    steps:\n    - uses: gitleaks/gitleaks-action@v2\n  container:\n    steps:\n    - run: trivy image --exit-code 1 $IMAGE\n  sign:\n    needs: [sast, sca, secrets, container]\n    steps:\n    - run: cosign sign $IMAGE", 0.5, 1.1, 9, 3.5, BLUE);

// 28: Thresholds & Gates
s = contentSlide(pres, "Security Gate Thresholds");
addCard(s, "Block (Fail Pipeline)", "- Any critical vulnerability\n- Hardcoded secrets detected\n- Known exploited CVEs (CISA KEV)\n- Unsigned container images\n- Failed policy checks", 0.5, 1.1, 4.3, 2.0, RED);
addCard(s, "Warn (Allow with Review)", "- High severity vulnerabilities\n- Medium SAST findings\n- Outdated dependencies (no CVE)\n- Missing SBOM fields\n- Non-standard configurations", 5.2, 1.1, 4.3, 2.0, ORANGE);
addCard(s, "Inform Only", "- Low severity findings\n- Code style suggestions\n- Performance hints\n- Informational advisories", 0.5, 3.3, 9, 1.2, BLUE);

// === SECTION 6: HANDS-ON LAB ===
sectionSlide(pres, "SECTION 06", "Hands-on Lab", "Implementing a DevSecOps Pipeline (90 min)");

// 30: Lab Overview
s = contentSlide(pres, "Lab: DevSecOps Pipeline");
addCard(s, "Part 1: Scanning (35 min)", "1. Semgrep SAST in GitHub Actions\n2. pip-audit for dependencies\n3. gitleaks for secrets scanning\n4. Trivy for container images\n5. Set thresholds: block critical, warn medium", 0.5, 1.1, 2.8, 2.2, PURPLE);
addCard(s, "Part 2: AI Review (30 min)", "1. Submit intentionally vulnerable PR\n2. Run Claude Code security review\n3. Document: what AI caught vs missed\n4. Fix all vulnerabilities\n5. Re-scan to verify fixes", 3.6, 1.1, 2.8, 2.2, BLUE);
addCard(s, "Part 3: Supply Chain (25 min)", "1. Generate SBOM with syft\n2. Sign image with cosign\n3. Add signing to CI/CD pipeline\n4. Create security audit report\n5. Document compliance evidence", 6.7, 1.1, 2.8, 2.2, GREEN);

// 31: Lab Part 1 Detail
s = contentSlide(pres, "Lab Part 1: Security Scanning Setup");
addCard(s, "Semgrep in CI", "# .github/workflows/security.yml\n- name: Semgrep SAST\n  uses: returntocorp/semgrep-action@v1\n  with:\n    config: >\n      p/python\n      p/security-audit\n      p/owasp-top-ten\n    generateSarif: true", 0.5, 1.1, 4.3, 2.2, PURPLE);
addCard(s, "Dependency + Secret Scan", "# SCA\n- name: pip-audit\n  run: |\n    pip install pip-audit\n    pip-audit --strict --desc\n\n# Secrets\n- name: gitleaks\n  uses: gitleaks/gitleaks-action@v2\n  env:\n    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}", 5.2, 1.1, 4.3, 2.2, GREEN);
addTagline(s, "All scans run in parallel for fast feedback");

// 32: Lab Part 2 Detail
s = contentSlide(pres, "Lab Part 2: AI Security Review");
addCard(s, "Vulnerable Code Sample", "# Intentionally vulnerable!\n@app.route('/search')\ndef search():\n    q = request.args.get('q')\n    cursor.execute(f\"SELECT * FROM users WHERE name='{q}'\")\n    return render_template_string(f'<h1>{q}</h1>')\n\n# Has: SQL injection, XSS, no validation", 0.5, 1.1, 4.3, 2.3, RED);
addCard(s, "AI Review Process", "1. Push vulnerable code as PR\n2. Claude Code reviews automatically\n3. Compare findings:\n   - SQL injection found?\n   - XSS detected?\n   - Missing validation noted?\n4. Document false positives/negatives\n5. Fix and re-scan", 5.2, 1.1, 4.3, 2.3, BLUE);
addTagline(s, "Learn what AI catches and what it misses - essential knowledge");

// 33: Lab Part 3 Detail
s = contentSlide(pres, "Lab Part 3: Supply Chain Security");
addCard(s, "SBOM Generation", "# Install syft\ncurl -sSfL https://raw.githubusercontent.com/\n  anchore/syft/main/install.sh | sh\n\n# Generate SBOM\nsyft myapp:latest -o spdx-json > sbom.json\nsyft myapp:latest -o cyclonedx-json > sbom-cdx.json", 0.5, 1.1, 4.3, 2.0, BLUE);
addCard(s, "Image Signing", "# Sign with cosign (keyless)\ncosign sign myregistry/myapp:latest\n\n# Attach SBOM as attestation\ncosign attest --predicate sbom.json \\\n  --type spdxjson \\\n  myregistry/myapp:latest", 5.2, 1.1, 4.3, 2.0, PURPLE);
addCard(s, "Security Audit Report", "Document: scans run, findings, fixes applied,\nSBOM contents, signed images, compliance mapping", 0.5, 3.3, 9, 1.1, GREEN);

// === SECTION 7: ASSESSMENT ===
sectionSlide(pres, "SECTION 07", "Assessment & Wrap-Up", "Deliverables & Next Steps");

// 35: Assessment
s = contentSlide(pres, "Week 8 Assessment: Security Audit Report");
addCard(s, "Deliverables", "1. CI/CD pipeline with all security scans\n2. AI review comparison report\n   (what AI caught vs missed)\n3. SBOM + signed container image\n4. Security audit report document\n5. Zero critical vulnerabilities", 0.5, 1.1, 4.3, 2.3, BLUE);
addCard(s, "Grading Criteria", "- Pipeline completeness (all scan types)\n- AI findings documentation quality\n- SBOM accuracy and completeness\n- Report professionalism\n- All critical findings resolved\n- Supply chain artifacts in place", 5.2, 1.1, 4.3, 2.3, PURPLE);
addTagline(s, "A secure pipeline is a professional pipeline");

// 36: OWASP Top 10 Quick Reference
s = contentSlide(pres, "OWASP Top 10 (2021) Quick Reference");
addNumberedItem(s, 1, "Broken Access Control", "Authorization failures - #1 most common", 0.5, 1.1, 4.3, PURPLE, PURPLE);
addNumberedItem(s, 2, "Cryptographic Failures", "Weak crypto, exposed data", 0.5, 1.75, 4.3, BLUE, BLUE);
addNumberedItem(s, 3, "Injection", "SQL, NoSQL, OS, LDAP injection", 0.5, 2.4, 4.3, RED, RED);
addNumberedItem(s, 4, "Insecure Design", "Missing security architecture", 0.5, 3.05, 4.3, ORANGE, ORANGE);
addNumberedItem(s, 5, "Security Misconfiguration", "Default configs, verbose errors", 0.5, 3.7, 4.3, GREEN, GREEN);
addNumberedItem(s, 6, "Vulnerable Components", "Known CVEs in dependencies", 5.2, 1.1, 4.3, CYAN, CYAN);
addNumberedItem(s, 7, "Auth Failures", "Broken authentication", 5.2, 1.75, 4.3, PINK, PINK);
addNumberedItem(s, 8, "Data Integrity", "Insecure deserialization, CI/CD", 5.2, 2.4, 4.3, PURPLE, PURPLE);
addNumberedItem(s, 9, "Logging Failures", "Insufficient monitoring", 5.2, 3.05, 4.3, BLUE, BLUE);
addNumberedItem(s, 10, "SSRF", "Server-Side Request Forgery", 5.2, 3.7, 4.3, RED, RED);

// 37: Recommended Reading
s = contentSlide(pres, "Recommended Reading");
addNumberedItem(s, 1, "OWASP DevSecOps Guideline", "owasp.org - Comprehensive reference for secure pipelines", 0.5, 1.1, 9, BLUE, BLUE);
addNumberedItem(s, 2, "Sigstore / Cosign Docs", "docs.sigstore.dev - Image signing and verification", 0.5, 1.75, 9, GREEN, GREEN);
addNumberedItem(s, 3, "Insecure Code with AI Assistants?", "Perry et al. (2023) ACM CCS - Research on AI code security", 0.5, 2.4, 9, PURPLE, PURPLE);
addNumberedItem(s, 4, "NIST SP 800-218", "Supply Chain Security Framework - Industry standard", 0.5, 3.05, 9, ORANGE, ORANGE);
addTagline(s, "Security knowledge compounds - invest in learning");

// 38: Key Takeaways
s = contentSlide(pres, "Key Takeaways");
addNumberedItem(s, 1, "Shift Left", "Security at every stage, not just before release", 0.5, 1.1, 9, RED, RED);
addNumberedItem(s, 2, "Automate Everything", "SAST + SCA + Secrets + Container scanning in CI/CD", 0.5, 1.75, 9, PURPLE, PURPLE);
addNumberedItem(s, 3, "AI + Human Review", "AI for patterns, humans for logic - both essential", 0.5, 2.4, 9, BLUE, BLUE);
addNumberedItem(s, 4, "Supply Chain Matters", "SBOM + signing + policy enforcement", 0.5, 3.05, 9, GREEN, GREEN);
addNumberedItem(s, 5, "Never Trust AI Code Blindly", "Always scan, always review, always verify", 0.5, 3.7, 9, ORANGE, ORANGE);

// 39: Next Week Preview
s = contentSlide(pres, "Next Week: Automated Testing with AI");
addCard(s, "Week 9 Preview", "- AI-generated test cases\n- Test pyramid with AI assistance\n- Property-based testing\n- Mutation testing\n- Test coverage strategies\n- Human validation of AI tests", 0.5, 1.1, 9, 2.0, PURPLE);
addTagline(s, "AI writes the tests, you verify they make sense");

// 40: Q&A
s = pres.addSlide();
s.background = { color: BG };
s.addShape("oval", { x: 2.5, y: -0.5, w: 3, h: 3, fill: { color: "8B5CF6", transparency: 88 } });
s.addShape("oval", { x: 6, y: 3.5, w: 2.5, h: 2.5, fill: { color: "3B82F6", transparency: 88 } });
s.addText("Questions?", { x: 0, y: 1.8, w: 10, h: 0.7, fontSize: 36, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
s.addText("Week 8: DevSecOps + AI Code Review", { x: 1, y: 2.7, w: 8, h: 0.4, fontSize: 14, fontFace: "Arial", color: BLUE, align: "center" });
s.addText("Anirach Mingkhwan | FITM, KMUTNB", { x: 2, y: 3.3, w: 6, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
addTagline(s, '"Security is a journey, not a destination"');

const outPath = "/home/clawdbot/clawd/tmp/Week08_raw.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`Saved ${outPath} (${pres.slides.length} slides)`);
}).catch(err => console.error("Error:", err));
