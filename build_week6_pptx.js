const pptxgen = require("pptxgenjs");

// Colors
const BG = "0D1229";
const CARD_BG = "1E2341";
const BLUE = "60A5FA";
const PURPLE = "A78BFA";
const GREEN = "4ADE80";
const ORANGE = "FBBF24";
const CYAN = "22D3EE";
const PINK = "F472B6";
const RED = "F87171";
const GOLD = "FBBF24";
const GRAY = "8B95B0";
const LIGHT = "B0B8D0";
const WHITE = "FFFFFF";

function addBadge(slide, text, x, y, colors) {
  const c1 = colors || "22C55E";
  slide.addShape("rect", { x, y, w: 1.8, h: 0.32, fill: { color: c1 }, rectRadius: 0.05 });
  slide.addText(text, { x, y, w: 1.8, h: 0.32, fontSize: 9, fontFace: "Arial", bold: true, color: WHITE, align: "center", valign: "middle", letterSpacing: 1.5 });
}

function addTagline(slide, text, y) {
  y = y || 4.95;
  slide.addShape("rect", { x: 0.4, y, w: 9.2, h: 0.4, fill: { color: "141E32" }, line: { color: "2A3560", width: 0.5 }, rectRadius: 0.05 });
  slide.addText(text, { x: 0.4, y, w: 9.2, h: 0.4, fontSize: 11, fontFace: "Arial", bold: true, color: GOLD, align: "center", valign: "middle" });
}

function addCard(slide, title, body, x, y, w, h, accentColor) {
  slide.addShape("rect", { x, y, w, h, fill: { color: CARD_BG }, rectRadius: 0.08 });
  slide.addShape("rect", { x, y, w: 0.06, h, fill: { color: accentColor || BLUE } });
  slide.addText(title, { x: x + 0.15, y, w: w - 0.2, h: 0.35, fontSize: 13, fontFace: "Arial", bold: true, color: accentColor || BLUE, valign: "top", margin: [4, 0, 0, 0] });
  slide.addText(body, { x: x + 0.15, y: y + 0.32, w: w - 0.2, h: h - 0.36, fontSize: 10, fontFace: "Arial", color: LIGHT, valign: "top", lineSpacingMultiple: 1.3 });
}

function addNumberedItem(slide, num, title, desc, x, y, w, circleColor, titleColor) {
  slide.addShape("rect", { x, y, w, h: 0.55, fill: { color: CARD_BG }, rectRadius: 0.06 });
  slide.addShape("oval", { x: x + 0.1, y: y + 0.1, w: 0.35, h: 0.35, fill: { color: circleColor } });
  slide.addText(String(num), { x: x + 0.1, y: y + 0.1, w: 0.35, h: 0.35, fontSize: 11, fontFace: "Arial", bold: true, color: WHITE, align: "center", valign: "middle" });
  slide.addText(title, { x: x + 0.55, y: y + 0.05, w: w - 0.65, h: 0.22, fontSize: 11, fontFace: "Arial", bold: true, color: titleColor || BLUE });
  slide.addText(desc, { x: x + 0.55, y: y + 0.27, w: w - 0.65, h: 0.23, fontSize: 9, fontFace: "Arial", color: LIGHT });
}

function titleSlide(pres, title, subtitle, week, author, meta) {
  let s = pres.addSlide();
  s.background = { color: BG };
  s.addShape("oval", { x: 2.5, y: -0.5, w: 3, h: 3, fill: { color: "8B5CF6", transparency: 88 } });
  s.addShape("oval", { x: 6, y: 3.5, w: 2.5, h: 2.5, fill: { color: "3B82F6", transparency: 88 } });
  s.addText("DEVOPS WITH VIBECODING", { x: 0, y: 1.2, w: 10, h: 0.35, fontSize: 11, fontFace: "Arial", color: "8B5CF6", align: "center", charSpacing: 4 });
  s.addText(title, { x: 0.5, y: 1.7, w: 9, h: 1.2, fontSize: 36, fontFace: "Arial", bold: true, color: WHITE, align: "center", valign: "middle", lineSpacingMultiple: 1.1 });
  s.addText(subtitle, { x: 0.5, y: 2.9, w: 9, h: 0.4, fontSize: 14, fontFace: "Arial", color: BLUE, align: "center" });
  s.addShape("rect", { x: 3, y: 3.7, w: 4, h: 0.8, fill: { color: CARD_BG }, line: { color: "2A3560", width: 0.5 }, rectRadius: 0.08 });
  s.addText(author, { x: 3, y: 3.72, w: 4, h: 0.4, fontSize: 13, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText(meta, { x: 3, y: 4.1, w: 4, h: 0.3, fontSize: 10, fontFace: "Arial", color: GRAY, align: "center" });
}

function sectionDivider(pres, part, title, subtitle) {
  let s = pres.addSlide();
  s.background = { color: BG };
  s.addShape("oval", { x: 2, y: -0.3, w: 3, h: 3, fill: { color: "8B5CF6", transparency: 88 } });
  s.addShape("oval", { x: 6, y: 3, w: 2.5, h: 2.5, fill: { color: "3B82F6", transparency: 88 } });
  s.addText(part, { x: 0, y: 1.5, w: 10, h: 0.35, fontSize: 11, fontFace: "Arial", color: "8B5CF6", align: "center", charSpacing: 4 });
  s.addText(title, { x: 0.5, y: 2, w: 9, h: 1, fontSize: 34, fontFace: "Arial", bold: true, color: WHITE, align: "center", valign: "middle" });
  s.addText(subtitle, { x: 0.5, y: 3.1, w: 9, h: 0.4, fontSize: 13, fontFace: "Arial", color: BLUE, align: "center" });
}

function practiceSlide(pres, time, title, subtitle) {
  let s = pres.addSlide();
  s.background = { color: "1A1520" };
  addBadge(s, "PRACTICE -- " + time, 0.4, 0.25, "E67E22");
  s.addText(title, { x: 0.4, y: 0.65, w: 9.2, h: 0.45, fontSize: 22, fontFace: "Arial", bold: true, color: GOLD, align: "center" });
  if (subtitle) s.addText(subtitle, { x: 0.4, y: 1.05, w: 9.2, h: 0.3, fontSize: 11, fontFace: "Arial", color: "D4A574", align: "center" });
  return s;
}

function addTable(slide, headers, rows, x, y, w, colW) {
  const h = [headers.map(h => ({ text: h, options: { fill: { color: "1B3A5C" }, color: WHITE, bold: true, fontSize: 9 } }))];
  const r = rows.map(row => row.map(c => ({ text: c, options: { color: LIGHT, fontSize: 9, fill: { color: CARD_BG } } })));
  slide.addTable([...h, ...r], { x, y, w, colW, border: { pt: 0.5, color: "2A3560" } });
}

function addCodeBlock(slide, code, x, y, w, h, borderColor) {
  slide.addShape("rect", { x, y, w, h, fill: { color: "1A1E2E" }, rectRadius: 0.06 });
  slide.addShape("rect", { x, y, w: 0.06, h, fill: { color: borderColor || "22C55E" } });
  slide.addText(code, { x: x + 0.15, y: y + 0.05, w: w - 0.2, h: h - 0.1, fontSize: 9, fontFace: "Courier New", color: "D4D4D4", lineSpacingMultiple: 1.3, valign: "top" });
}

const colors = ["3B82F6", "8B5CF6", "22C55E", "F59E0B", "06B6D4", "EC4899"];
const tcolors = [BLUE, PURPLE, GREEN, ORANGE, CYAN, PINK];

async function build() {
  let pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "Anirach Mingkhwan";
  pres.title = "Week 6: Infrastructure as Code with AI Prompting";

  // ===== SLIDE 1: Title =====
  titleSlide(pres,
    "Infrastructure as Code\nwith AI Prompting",
    "Week 6 — Provision Cloud Resources with VibeCoding",
    "Week 6",
    "Anirach Mingkhwan",
    "3-Hour Lecture | Week 6 of 15"
  );

  // ===== SLIDE 2: Agenda =====
  let s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "AGENDA", 0.4, 0.25, "3B82F6");
  s.addText("Today's Roadmap", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("3 hours: IaC fundamentals, Terraform, Ansible, and AI-driven provisioning", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addNumberedItem(s, 1, "Part 1 -- IaC Concepts", "What is IaC, why it matters, Terraform core concepts (60 min)", 0.4, 1.4, 9.2, "3B82F6", BLUE);
  addNumberedItem(s, 2, "Part 2 -- Config Management", "Ansible basics, Pulumi, GitOps & version control (50 min)", 0.4, 2.05, 9.2, "8B5CF6", PURPLE);
  addNumberedItem(s, 3, "Part 3 -- Production IaC", "Modules, testing, CI/CD integration, real-world cases (60 min)", 0.4, 2.7, 9.2, "22C55E", GREEN);
  addNumberedItem(s, 4, "Practice Labs", "3 hands-on sessions: Terraform + AI, Ansible, Full pipeline (30 min)", 0.4, 3.35, 9.2, "F59E0B", ORANGE);
  addTagline(s, "By end of class: you will provision real cloud infrastructure using AI-generated code");

  // ===== SLIDE 3: What is IaC =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "IaC FUNDAMENTALS", 0.4, 0.25, "3B82F6");
  s.addText("What is Infrastructure as Code?", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Managing infrastructure through machine-readable config files instead of manual processes", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCard(s, "Traditional (ClickOps)", "• Log into AWS Console\n• Click through menus manually\n• Configure settings by hand\n• No record of what was done\n• Hard to reproduce or scale\n• Error-prone and slow", 0.4, 1.4, 4.3, 1.9, RED);
  addCard(s, "Infrastructure as Code", "• Describe infra in config files\n• Version control everything in Git\n• Automated provisioning\n• Full audit trail of changes\n• Reproduce environments instantly\n• Peer review before applying", 5.3, 1.4, 4.3, 1.9, GREEN);
  addCard(s, "Core Principle", "If it is not in code, it does not exist. Infrastructure should be defined, versioned, tested, and deployed just like application code.", 0.4, 3.5, 9.2, 0.9, BLUE);
  addTagline(s, "IaC = treating your servers like software. Version it. Test it. Review it.");

  // ===== SLIDE 4: Why IaC =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "WHY IAC", 0.4, 0.25, "3B82F6");
  s.addText("Why Infrastructure as Code?", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  const benefits = [
    ["Speed", "Provision in minutes vs. hours of manual work", "3B82F6", BLUE],
    ["Consistency", "Identical envs: dev, staging, prod every time", "8B5CF6", PURPLE],
    ["Reproducibility", "Recreate any environment from scratch on demand", "22C55E", GREEN],
    ["Auditability", "Git history = full infrastructure change log", "F59E0B", ORANGE],
    ["Disaster Recovery", "Rebuild entire infrastructure from code in minutes", "06B6D4", CYAN],
    ["Cost Control", "Destroy dev environments after hours, save $$$", "EC4899", PINK],
  ];
  benefits.forEach((b, i) => {
    const bx = i < 3 ? 0.4 : 5.15;
    const by = 1.3 + (i % 3) * 1.1;
    addCard(s, b[0], b[1], bx, by, 4.5, 0.9, b[2]);
  });
  addTagline(s, "Companies using IaC deploy 200x more frequently with 24x faster recovery times (DORA)");

  // ===== SLIDE 5: IaC Categories =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "IAC LANDSCAPE", 0.4, 0.25, "3B82F6");
  s.addText("The IaC Tool Landscape", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addTable(s, ["Category", "Tools", "What It Does", "When to Use"],
    [
      ["Provisioning", "Terraform, Pulumi, CloudFormation", "Create cloud resources (VMs, networks, DBs)", "Setting up infrastructure"],
      ["Config Management", "Ansible, Chef, Puppet, SaltStack", "Configure servers after provisioning", "Software install, settings"],
      ["Container Orchestration", "Kubernetes, Docker Compose", "Manage containerized workloads", "Microservices, scaling"],
      ["Immutable Infra", "Packer, Docker", "Build pre-configured images", "Reproducible environments"],
      ["Policy as Code", "Open Policy Agent, Sentinel", "Enforce security & compliance rules", "Enterprise governance"],
    ],
    0.4, 1.15, 9.2, [1.5, 2.5, 2.6, 2.6]);
  addCard(s, "This Week's Focus", "Terraform (provisioning) + Ansible (config management) + Pulumi (code-first) + GitOps", 0.4, 3.95, 9.2, 0.6, CYAN);

  // ===== SLIDE 6: Section Divider - Part 1 =====
  sectionDivider(pres, "PART 1 -- IAC CONCEPTS", "Terraform Core", "Providers, Resources, State, and the Plan/Apply Cycle");

  // ===== SLIDE 7: Terraform Overview =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "TERRAFORM", 0.4, 0.25, "7B3FE4");
  s.addText("Terraform: The IaC Standard", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCard(s, "What is Terraform?", "Open-source tool by HashiCorp. Uses HCL (HashiCorp Configuration Language) to define infrastructure. Works with 1000+ providers: AWS, GCP, Azure, Kubernetes, GitHub, and more.", 0.4, 1.2, 9.2, 0.85, PURPLE);
  const tfstats = [["1000+", "Providers"], [">3M", "Downloads/Week"], ["2014", "First Release"], ["HCL", "Language"]];
  tfstats.forEach((st, i) => {
    const sx = 0.5 + i * 2.35;
    s.addShape("rect", { x: sx, y: 2.25, w: 2.1, h: 0.9, fill: { color: CARD_BG }, rectRadius: 0.06 });
    s.addText(st[0], { x: sx, y: 2.3, w: 2.1, h: 0.42, fontSize: 22, fontFace: "Arial", bold: true, color: PURPLE, align: "center" });
    s.addText(st[1], { x: sx, y: 2.75, w: 2.1, h: 0.3, fontSize: 9, fontFace: "Arial", color: GRAY, align: "center" });
  });
  addCodeBlock(s, "# Example: Launch an EC2 instance on AWS\nresource \"aws_instance\" \"web\" {\n  ami           = \"ami-0c55b159cbfafe1f0\"\n  instance_type = \"t3.micro\"\n  tags = { Name = \"MyWebServer\" }\n}", 0.4, 3.35, 9.2, 1.2, PURPLE);

  // ===== SLIDE 8: Terraform Providers =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "PROVIDERS", 0.4, 0.25, "7B3FE4");
  s.addText("Terraform Providers", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Providers are plugins that let Terraform talk to external services", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCodeBlock(s, "terraform {\n  required_providers {\n    aws = {\n      source  = \"hashicorp/aws\"\n      version = \"~> 5.0\"\n    }\n  }\n}\n\nprovider \"aws\" {\n  region = \"ap-southeast-1\"  # Singapore\n}", 0.4, 1.4, 4.5, 2.4, PURPLE);
  addCard(s, "Popular Providers", "AWS (amazon web services)\nGoogle Cloud Platform (GCP)\nMicrosoft Azure\nKubernetes\nGitHub, Cloudflare, Datadog\nLocal files, Random, TLS", 5.1, 1.4, 4.5, 2.4, BLUE);
  addCard(s, "How Providers Work", "1. Declare in required_providers block\n2. Run terraform init to download\n3. Configure with credentials/region\n4. Use provider resources in config", 0.4, 4.0, 9.2, 0.75, CYAN);

  // ===== SLIDE 9: Terraform Resources =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "RESOURCES", 0.4, 0.25, "7B3FE4");
  s.addText("Terraform Resources", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Resources are the building blocks -- each represents one infrastructure object", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCodeBlock(s, "# Syntax: resource \"<TYPE>\" \"<NAME>\" { ... }\n\n# VPC\nresource \"aws_vpc\" \"main\" {\n  cidr_block = \"10.0.0.0/16\"\n}\n\n# Subnet (references VPC above)\nresource \"aws_subnet\" \"public\" {\n  vpc_id     = aws_vpc.main.id  # dependency!\n  cidr_block = \"10.0.1.0/24\"\n}", 0.4, 1.4, 5.3, 2.6, PURPLE);
  addCard(s, "Resource Types", "aws_instance -- EC2 virtual machine\naws_s3_bucket -- S3 storage bucket\naws_db_instance -- RDS database\naws_security_group -- Firewall rules\ngoogle_compute_instance -- GCP VM\nazurerm_virtual_machine -- Azure VM", 5.9, 1.4, 3.7, 2.6, GREEN);
  addCard(s, "Key Concepts", "Resources can reference each other using dot notation: resource_type.name.attribute. Terraform builds a dependency graph automatically.", 0.4, 4.15, 9.2, 0.6, ORANGE);

  // ===== SLIDE 10: Terraform State =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "STATE", 0.4, 0.25, "7B3FE4");
  s.addText("Terraform State", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("State is how Terraform knows what it has already created", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCard(s, "terraform.tfstate", "A JSON file that maps your HCL config to real cloud resources. Tracks IDs, attributes, and dependencies. NEVER edit manually!", 0.4, 1.4, 4.3, 1.0, PURPLE);
  addCard(s, "Remote State (Best Practice)", "Store state in S3 + DynamoDB (AWS) or Terraform Cloud. Team collaboration, locking, encryption, versioning.", 5.3, 1.4, 4.3, 1.0, BLUE);
  addCodeBlock(s, "# Remote state backend (AWS S3)\nterraform {\n  backend \"s3\" {\n    bucket         = \"my-tf-state-bucket\"\n    key            = \"prod/terraform.tfstate\"\n    region         = \"ap-southeast-1\"\n    dynamodb_table = \"terraform-locks\"  # prevent conflicts\n    encrypt        = true\n  }\n}", 0.4, 2.6, 9.2, 2.1, GREEN);

  // ===== SLIDE 11: Plan/Apply Cycle =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "PLAN/APPLY", 0.4, 0.25, "7B3FE4");
  s.addText("The Terraform Workflow", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  const tfsteps = [
    { num: "1", title: "terraform init", desc: "Download providers & modules, set up backend", color: "3B82F6" },
    { num: "2", title: "terraform plan", desc: "Preview changes: what will be created/changed/destroyed", color: "8B5CF6" },
    { num: "3", title: "terraform apply", desc: "Execute the plan, provision real infrastructure", color: "22C55E" },
    { num: "4", title: "terraform destroy", desc: "Tear down all managed resources (use carefully!)", color: "F87171" },
  ];
  tfsteps.forEach((st, i) => addNumberedItem(s, st.num, st.title, st.desc, 0.4, 1.3 + i * 0.72, 9.2, st.color, tcolors[i]));
  addCodeBlock(s, "$ terraform plan\n  + aws_instance.web  (will be CREATED)\n  ~ aws_security_group.main  (will be MODIFIED)\n  - aws_s3_bucket.old  (will be DESTROYED)\nPlan: 1 to add, 1 to change, 1 to destroy.", 0.4, 4.3, 9.2, 0.8, PURPLE);

  // ===== SLIDE 12: VibeCoding Terraform - Intro =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "VIBECODING + IAC", 0.4, 0.25, "EC4899");
  s.addText("VibeCoding with Terraform", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Use AI to generate, explain, and optimize Terraform configurations", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCard(s, "Why AI + Terraform is Powerful", "Terraform HCL has a large surface area: 1000+ providers, complex syntax, subtle state issues. AI knows ALL of it and can generate correct configs instantly. You provide the intent, AI writes the code.", 0.4, 1.4, 9.2, 1.0, PINK);
  addCodeBlock(s, "# Prompt to AI:\n\"Create a Terraform config for AWS: VPC with public/private subnets,\n Internet Gateway, NAT Gateway, security groups allowing HTTP/HTTPS,\n and an EC2 t3.micro in the public subnet with an Elastic IP.\"\n\n# AI generates: ~80 lines of valid HCL in seconds", 0.4, 2.6, 9.2, 1.4, CYAN);
  addCard(s, "VibeCoding Workflow", "Describe intent in plain English -> AI generates HCL -> You review & understand -> terraform plan -> iterate with AI if needed -> terraform apply", 0.4, 4.2, 9.2, 0.55, GREEN);

  // ===== SLIDE 13: AI Prompting Strategies for IaC =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "PROMPT PATTERNS", 0.4, 0.25, "EC4899");
  s.addText("AI Prompting Strategies for IaC", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addTable(s, ["Prompt Pattern", "Example", "Use Case"],
    [
      ["Generate from scratch", "\"Create Terraform for an RDS PostgreSQL instance with read replica\"", "New resources"],
      ["Explain existing code", "\"Explain what this Terraform module does line by line\"", "Understanding code"],
      ["Debug errors", "\"Fix this Terraform error: Error creating Instance: InvalidParameterValue\"", "Troubleshooting"],
      ["Optimize & refactor", "\"Refactor this config to use variables and locals instead of hardcoded values\"", "Code quality"],
      ["Security review", "\"Review this Terraform for security issues and suggest fixes\"", "Security audit"],
      ["Convert ClickOps", "\"Convert these AWS Console steps to Terraform code: [screenshot]\"", "Migrate from manual"],
    ],
    0.4, 1.15, 9.2, [2.1, 4.9, 2.2]);
  addTagline(s, "Pro tip: Always ask AI to explain the code it generates -- understand before you apply!");

  // ===== SLIDE 14: Terraform Variables & Outputs =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "VARIABLES", 0.4, 0.25, "7B3FE4");
  s.addText("Variables, Locals & Outputs", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCodeBlock(s, "# variables.tf -- Input variables\nvariable \"instance_type\" {\n  type    = string\n  default = \"t3.micro\"\n  description = \"EC2 instance type\"\n}\n\n# locals.tf -- Computed values\nlocals {\n  env_prefix = \"${var.project}-${var.environment}\"\n}", 0.4, 1.3, 4.6, 2.4, PURPLE);
  addCodeBlock(s, "# outputs.tf -- Export values\noutput \"instance_ip\" {\n  value       = aws_instance.web.public_ip\n  description = \"Public IP of web server\"\n}\n\n# Use variables in resources\nresource \"aws_instance\" \"web\" {\n  instance_type = var.instance_type\n  tags = { Name = local.env_prefix }\n}", 5.2, 1.3, 4.4, 2.4, GREEN);
  addCard(s, "Variable Precedence (highest to lowest)", "1. CLI flags: -var=\"key=value\"\n2. .tfvars files: terraform.tfvars\n3. Environment: TF_VAR_name\n4. Default values in variable blocks", 0.4, 3.9, 9.2, 0.85, CYAN);

  // ===== SLIDE 15: Practice 1 =====
  s = practiceSlide(pres, "15 MIN", "Practice 1: Write Terraform with AI", "Use AI to provision a web server stack on AWS");
  addCard(s, "Task", "Use AI (ChatGPT/Claude/Copilot) to generate Terraform code that provisions: VPC, public subnet, Internet Gateway, Security Group (HTTP/HTTPS/SSH), EC2 t3.micro with user_data to install nginx. Then run terraform plan.", 0.4, 1.5, 9.2, 1.0, ORANGE);
  addNumberedItem(s, 1, "Prompt AI", "Describe the full architecture in plain English", 0.4, 2.65, 4.3, "3B82F6", BLUE);
  addNumberedItem(s, 2, "Review the code", "Read every line -- ask AI to explain anything unclear", 5.3, 2.65, 4.3, "8B5CF6", PURPLE);
  addNumberedItem(s, 3, "terraform init && plan", "Check the execution plan -- what will be created?", 0.4, 3.3, 4.3, "22C55E", GREEN);
  addNumberedItem(s, 4, "Iterate with AI", "Ask AI to add variables, outputs, tags", 5.3, 3.3, 4.3, "F59E0B", ORANGE);
  addTagline(s, "Remember: Do NOT run terraform apply unless you have an AWS account set up!");

  // ===== SLIDE 16: Section Divider - Part 2 =====
  sectionDivider(pres, "PART 2 -- CONFIG MANAGEMENT", "Ansible & Configuration\nManagement", "Configure servers after they're provisioned");

  // ===== SLIDE 17: Ansible Overview =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "ANSIBLE", 0.4, 0.25, "06B6D4");
  s.addText("Ansible: Agentless Config Management", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCard(s, "What is Ansible?", "Open-source automation tool for configuration management, app deployment, and task automation. Uses YAML playbooks. No agent required on managed nodes -- connects via SSH.", 0.4, 1.2, 9.2, 0.85, CYAN);
  addCard(s, "Terraform vs Ansible", "Terraform: WHAT to provision (create VM, network, DB)\nAnsible: HOW to configure it (install software, set configs)\nUsed together: Terraform creates infra, Ansible configures it", 0.4, 2.2, 9.2, 0.85, BLUE);
  addCodeBlock(s, "# Ansible inventory (hosts.ini)\n[webservers]\n192.168.1.10\n192.168.1.11\n\n[databases]\ndb.example.com\n\n# Run a playbook\nansible-playbook -i hosts.ini site.yml", 0.4, 3.25, 4.5, 1.55, CYAN);
  addCard(s, "Key Concepts", "Inventory: list of managed servers\nPlaybook: YAML file with automation steps\nTask: a single action (install, copy, service)\nRole: reusable collection of tasks", 5.1, 3.25, 4.7, 1.55, GREEN);

  // ===== SLIDE 18: Ansible Playbooks =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "PLAYBOOKS", 0.4, 0.25, "06B6D4");
  s.addText("Writing Ansible Playbooks", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCodeBlock(s, "# site.yml -- Install and configure Nginx\n---\n- name: Configure web servers\n  hosts: webservers\n  become: yes  # sudo\n  vars:\n    nginx_port: 80\n  tasks:\n    - name: Install nginx\n      apt:\n        name: nginx\n        state: present\n        update_cache: yes\n    - name: Start nginx\n      service:\n        name: nginx\n        state: started\n        enabled: yes\n    - name: Copy config\n      template:\n        src: nginx.conf.j2\n        dest: /etc/nginx/nginx.conf\n      notify: Restart nginx\n  handlers:\n    - name: Restart nginx\n      service: name=nginx state=restarted", 0.4, 1.15, 5.3, 3.6, CYAN);
  addCard(s, "Ansible Modules", "apt/yum -- package management\nservice -- start/stop services\ncopy/template -- file management\nuser -- manage system users\nshell/command -- run commands\nfire -- manage iptables rules\ncron -- schedule tasks\ndocker_container -- manage Docker", 5.9, 1.15, 3.7, 3.6, BLUE);

  // ===== SLIDE 19: Ansible Roles & Galaxy =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "ROLES", 0.4, 0.25, "06B6D4");
  s.addText("Ansible Roles & Galaxy", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCard(s, "Roles = Reusable Ansible Packages", "A role organizes tasks, templates, variables into a standard directory structure. Share roles via Ansible Galaxy (community hub with 10,000+ roles).", 0.4, 1.15, 9.2, 0.75, CYAN);
  addCodeBlock(s, "# Role directory structure\nroles/\n  nginx/\n    tasks/main.yml    # Main task list\n    handlers/main.yml # Notification handlers\n    templates/        # Jinja2 templates (.j2)\n    vars/main.yml     # Role variables\n    defaults/main.yml # Default variable values\n    meta/main.yml     # Role metadata", 0.4, 2.1, 4.5, 2.0, GREEN);
  addCodeBlock(s, "# Use a community role from Galaxy\nansible-galaxy install geerlingguy.nginx\n\n# site.yml using roles\n---\n- hosts: webservers\n  roles:\n    - geerlingguy.nginx\n    - geerlingguy.mysql\n    - mycompany.app_deploy", 5.1, 2.1, 4.5, 2.0, PURPLE);
  addTagline(s, "Ansible Galaxy: ansible.galaxy.com -- don't reinvent the wheel!");

  // ===== SLIDE 20: Pulumi =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "PULUMI", 0.4, 0.25, "7B3FE4");
  s.addText("Pulumi: IaC with Real Code", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Write infrastructure in Python, TypeScript, Go, or C# -- no new language to learn", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCodeBlock(s, "# Pulumi in Python (compare to Terraform HCL)\nimport pulumi\nimport pulumi_aws as aws\n\nbucket = aws.s3.Bucket('my-bucket',\n    acl='private',\n    tags={'Environment': 'production'}\n)\n\npulumi.export('bucket_name', bucket.id)", 0.4, 1.4, 4.7, 2.0, PURPLE);
  addCard(s, "Pulumi vs Terraform", "Terraform: DSL (HCL) -- learn new syntax\nPulumi: Real languages -- use loops, functions, classes\n\nPulumi advantages:\n- Full programming power (loops, conditions)\n- Reuse existing libraries\n- Better IDE support (autocomplete)\n- Great for developers", 5.3, 1.4, 4.3, 2.0, BLUE);
  addCard(s, "When to Choose Pulumi", "Team is developers (not ops-focused)\nComplex logic needed in infra code\nExisting Python/TS expertise\nNeed advanced abstractions", 0.4, 3.6, 4.3, 0.9, GREEN);
  addCard(s, "When to Choose Terraform", "Ops-focused team or mixed team\nLarge ecosystem & community needed\nHCL is readable and declarative\nIndustry standard for most shops", 5.3, 3.6, 4.3, 0.9, CYAN);

  // ===== SLIDE 21: GitOps =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "GITOPS", 0.4, 0.25, "4ADE80");
  s.addText("GitOps: Git as the Source of Truth", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("The infrastructure reflects exactly what is in Git -- always", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCard(s, "GitOps Principles", "1. Declarative: Entire system state described in Git\n2. Versioned: Git history = complete audit trail\n3. Automated: Changes applied automatically via CD\n4. Continuous reconciliation: System auto-corrects drift", 0.4, 1.4, 4.5, 1.8, GREEN);
  addCard(s, "GitOps Workflow", "Developer submits PR to infra repo\nPeer review + automated checks\nMerge to main triggers CI/CD\nCD pipeline runs terraform apply\nSystem matches Git state exactly\nAuto-alerts on configuration drift", 5.1, 1.4, 4.5, 1.8, BLUE);
  addCodeBlock(s, "# .github/workflows/terraform.yml\non:\n  push:\n    branches: [main]\njobs:\n  terraform:\n    steps:\n      - uses: actions/checkout@v3\n      - run: terraform init\n      - run: terraform plan -out=tfplan\n      - run: terraform apply tfplan", 0.4, 3.4, 9.2, 1.3, CYAN);

  // ===== SLIDE 22: IaC Version Control Best Practices =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "VERSION CONTROL", 0.4, 0.25, "4ADE80");
  s.addText("IaC Version Control Best Practices", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCard(s, "Repository Structure", "infra/\n  modules/        # Reusable modules\n  environments/\n    dev/            # Dev environment\n    staging/        # Staging environment\n    prod/           # Production environment\n  .github/workflows/ # CI/CD pipelines", 0.4, 1.2, 4.5, 1.8, BLUE);
  addCard(s, "What to Commit", "All .tf files (providers, resources, variables)\n.gitignore: *.tfstate, .terraform/, *.tfvars\nREADME.md with architecture diagrams\nCI/CD pipeline configs", 5.1, 1.2, 4.5, 1.8, GREEN);
  addCard(s, "Branch Strategy for IaC", "feature/* -- develop new infra changes\nPR required -- peer review all changes\nmain -- triggers auto plan in CI\nTag releases -- v1.0.0 for milestones\nNever commit tfstate to Git!", 0.4, 3.2, 4.5, 1.55, PURPLE);
  addCard(s, "Pre-commit Hooks", "terraform fmt -- auto-format HCL\nterraform validate -- syntax check\ntflint -- additional linting\ncheckov -- security scanning\nterraform-docs -- auto-generate docs", 5.1, 3.2, 4.5, 1.55, ORANGE);

  // ===== SLIDE 23: Practice 2 =====
  s = practiceSlide(pres, "15 MIN", "Practice 2: Ansible Playbook with AI", "Use AI to write a configuration management playbook");
  addCard(s, "Task", "Use AI to generate an Ansible playbook that: installs Docker on Ubuntu servers, starts the Docker service, pulls a specified image, and runs a container with health checks.", 0.4, 1.5, 9.2, 0.9, ORANGE);
  addNumberedItem(s, 1, "Prompt AI", '"Write an Ansible playbook to install Docker on Ubuntu and run nginx container"', 0.4, 2.55, 9.2, "3B82F6", BLUE);
  addNumberedItem(s, 2, "Add variables", "Ask AI to add vars for image name, port, container name", 0.4, 3.2, 4.3, "8B5CF6", PURPLE);
  addNumberedItem(s, 3, "Add idempotency", "Ask AI: \"make this idempotent -- safe to run multiple times\"", 5.3, 3.2, 4.3, "22C55E", GREEN);
  addNumberedItem(s, 4, "Test with --check", "ansible-playbook --check (dry run, no changes)", 0.4, 3.85, 4.3, "F59E0B", ORANGE);
  addTagline(s, "Idempotency means running the playbook 10 times = same result as running it once");

  // ===== SLIDE 24: Section Divider - Part 3 =====
  sectionDivider(pres, "PART 3 -- PRODUCTION IAC", "Advanced Terraform &\nCI/CD Integration", "Modules, testing, and production-grade workflows");

  // ===== SLIDE 25: Terraform Modules =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "MODULES", 0.4, 0.25, "F59E0B");
  s.addText("Terraform Modules", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Reusable, shareable infrastructure components", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCodeBlock(s, "# Call a module (from local path or registry)\nmodule \"vpc\" {\n  source  = \"terraform-aws-modules/vpc/aws\"\n  version = \"5.0.0\"\n\n  name = \"my-vpc\"\n  cidr = \"10.0.0.0/16\"\n  azs  = [\"ap-southeast-1a\", \"ap-southeast-1b\"]\n  private_subnets = [\"10.0.1.0/24\", \"10.0.2.0/24\"]\n  public_subnets  = [\"10.0.101.0/24\", \"10.0.102.0/24\"]\n  enable_nat_gateway = true\n}", 0.4, 1.4, 5.4, 2.6, ORANGE);
  addCard(s, "Module Benefits", "Encapsulation: hide complex logic\nReusability: use across projects\nStandardization: enforce patterns\nRegistry: 1000+ public modules\n  terraform.io/registry", 5.9, 1.4, 3.7, 2.6, BLUE);
  addCard(s, "Module Structure", "modules/my-module/\n  main.tf        # Resources\n  variables.tf   # Inputs\n  outputs.tf     # Outputs\n  README.md      # Docs", 0.4, 4.2, 9.2, 0.6, GREEN);

  // ===== SLIDE 26: Terraform Workspaces =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "WORKSPACES", 0.4, 0.25, "F59E0B");
  s.addText("Terraform Workspaces", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Multiple state files from one configuration -- manage dev/staging/prod", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCodeBlock(s, "# Create and switch workspaces\nterraform workspace new dev\nterraform workspace new staging\nterraform workspace new prod\nterraform workspace select prod\nterraform workspace list\n\n# Use workspace in config\nlocals {\n  env = terraform.workspace\n  instance_size = {\n    dev     = \"t3.micro\"\n    staging = \"t3.small\"\n    prod    = \"t3.large\"\n  }[local.env]\n}", 0.4, 1.4, 5.4, 2.9, ORANGE);
  addCard(s, "Workspace vs. Directories", "Workspaces: simple isolation, same config\nDirectory approach: separate configs per env\n\nRecommendation: Use directories for prod! Each env should have its own Terraform root to prevent accidental cross-env applies.", 5.9, 1.4, 3.7, 2.9, PURPLE);
  addTagline(s, "Never run terraform apply in prod from the same directory as dev!");

  // ===== SLIDE 27: Remote State & Secrets =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "SECRETS", 0.4, 0.25, "F59E0B");
  s.addText("Remote State & Secrets Management", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCard(s, "Remote State Data Sources", "Read output from another Terraform state file without hardcoding values between environments or modules.", 0.4, 1.2, 9.2, 0.65, BLUE);
  addCodeBlock(s, "# Read outputs from another state file\ndata \"terraform_remote_state\" \"vpc\" {\n  backend = \"s3\"\n  config = {\n    bucket = \"my-tf-state\"\n    key    = \"vpc/terraform.tfstate\"\n    region = \"ap-southeast-1\"\n  }\n}\n\n# Use the output\nresource \"aws_instance\" \"app\" {\n  subnet_id = data.terraform_remote_state.vpc.outputs.private_subnet_id\n}", 0.4, 2.05, 5.4, 2.6, ORANGE);
  addCard(s, "Secrets Best Practices", "NEVER put secrets in .tf files!\nUse: AWS Secrets Manager / Parameter Store\nOr: HashiCorp Vault\nOr: Environment variables (TF_VAR_*)\n\ndata \"aws_secretsmanager_secret_version\" source to read at runtime", 5.9, 2.05, 3.7, 2.6, RED);
  addTagline(s, "If secrets end up in state file -- encrypt state at rest! (S3 SSE or Vault)");

  // ===== SLIDE 28: Advanced Terraform - Data Sources =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "DATA SOURCES", 0.4, 0.25, "F59E0B");
  s.addText("Data Sources & Dynamic Blocks", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCodeBlock(s, "# Data source: read existing AWS resources\ndata \"aws_ami\" \"ubuntu\" {\n  most_recent = true\n  owners      = [\"099720109477\"]\n  filter {\n    name   = \"name\"\n    values = [\"ubuntu/images/hvm-ssd/ubuntu-*-22.04-amd64-server-*\"]\n  }\n}\n\nresource \"aws_instance\" \"web\" {\n  ami = data.aws_ami.ubuntu.id  # always latest Ubuntu\n}", 0.4, 1.3, 4.6, 2.5, ORANGE);
  addCodeBlock(s, "# Dynamic blocks -- avoid repetition\nresource \"aws_security_group\" \"web\" {\n  name = \"web-sg\"\n\n  dynamic \"ingress\" {\n    for_each = [80, 443, 8080]\n    content {\n      from_port   = ingress.value\n      to_port     = ingress.value\n      protocol    = \"tcp\"\n      cidr_blocks = [\"0.0.0.0/0\"]\n    }\n  }\n}", 5.1, 1.3, 4.5, 2.5, PURPLE);
  addCard(s, "Other Advanced Features", "count & for_each: create multiple resources\nconditional expressions: condition ? true : false\nlocal-exec/remote-exec: run scripts\nlifecycle blocks: control resource behavior\nterraform_data: trigger replacements", 0.4, 4.0, 9.2, 0.75, CYAN);

  // ===== SLIDE 29: IaC Testing =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "TESTING", 0.4, 0.25, "22C55E");
  s.addText("Testing Infrastructure Code", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Yes, you should test your Terraform like application code", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addTable(s, ["Testing Level", "Tool", "What It Checks", "Speed"],
    [
      ["Static Analysis", "terraform validate, tflint", "Syntax, type errors, best practices", "Seconds"],
      ["Security Scan", "checkov, tfsec, terrascan", "Security misconfigurations, CIS benchmarks", "Seconds"],
      ["Unit Testing", "terraform test (built-in)", "Module logic with mock providers", "Minutes"],
      ["Integration Test", "Terratest (Go)", "Actually provisions real infra, verifies", "10-30 min"],
      ["Policy Testing", "OPA/Sentinel", "Enforce org-wide compliance rules", "Seconds"],
      ["Drift Detection", "terraform plan -detailed-exitcode", "Detect manual changes vs. desired state", "Minutes"],
    ],
    0.4, 1.15, 9.2, [1.8, 2.2, 3.4, 1.8]);
  addTagline(s, "Minimum: validate + tflint + checkov on every PR. Terratest for critical modules.");

  // ===== SLIDE 30: CI/CD for IaC =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "CI/CD + IAC", 0.4, 0.25, "22C55E");
  s.addText("CI/CD Pipeline for Terraform", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCodeBlock(s, "# .github/workflows/terraform.yml\nname: Terraform CI/CD\non:\n  pull_request:\n    branches: [main]\n  push:\n    branches: [main]\n\njobs:\n  validate:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v3\n      - uses: hashicorp/setup-terraform@v3\n        with:\n          terraform_version: 1.7.0\n      - run: terraform init -backend=false\n      - run: terraform fmt -check\n      - run: terraform validate\n      - run: tflint\n      - run: checkov -d . --framework terraform", 0.4, 1.3, 5.3, 3.5, GREEN);
  addCodeBlock(s, "# Deploy job (on merge to main)\n  deploy:\n    needs: validate\n    if: github.ref == 'refs/heads/main'\n    environment: production\n    steps:\n      - run: terraform init\n      - run: terraform plan -out=tfplan\n      - name: Upload plan\n        uses: actions/upload-artifact@v3\n        with: {name: tfplan, path: tfplan}\n      - name: Apply\n        run: terraform apply tfplan", 5.5, 1.3, 4.1, 3.5, BLUE);
  addTagline(s, "Always plan on PR, always apply on merge -- never apply locally in production!");

  // ===== SLIDE 31: Atlantis & Terraform Cloud =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "AUTOMATION", 0.4, 0.25, "22C55E");
  s.addText("Terraform Automation Tools", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCard(s, "Atlantis (Open Source)", "Bot that runs terraform plan on PR comments and terraform apply on merge. Self-hosted. Posts plan output directly in PR for review.\n\nComment: 'atlantis plan' -> shows plan in PR\nComment: 'atlantis apply' -> applies on approval", 0.4, 1.2, 4.3, 2.0, CYAN);
  addCard(s, "Terraform Cloud / HCP", "HashiCorp's managed service:\n- Remote state management\n- Remote plan/apply execution\n- Team collaboration features\n- Sentinel policy enforcement\n- Cost estimation before apply\n- Free tier: up to 500 resources", 5.3, 1.2, 4.3, 2.0, PURPLE);
  addCard(s, "Spacelift / Scalr / env0", "Commercial alternatives with:\n- Multi-cloud support\n- Cost management\n- Drift detection\n- Audit logs\n- RBAC & approval workflows", 0.4, 3.35, 4.3, 1.4, ORANGE);
  addCard(s, "Recommendation", "Small team: GitHub Actions + remote state\nMid-size: Atlantis or Terraform Cloud Free\nEnterprise: Terraform Cloud Business or Spacelift", 5.3, 3.35, 4.3, 1.4, GREEN);

  // ===== SLIDE 32: VibeCoding Workflow for IaC =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "VIBECODING WORKFLOW", 0.4, 0.25, "EC4899");
  s.addText("VibeCoding Workflow for IaC", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addNumberedItem(s, 1, "Describe the architecture", "Tell AI the full picture: cloud, services, scale, region, security needs", 0.4, 1.2, 9.2, "3B82F6", BLUE);
  addNumberedItem(s, 2, "Generate scaffold", '"Create the Terraform directory structure for this architecture"', 0.4, 1.85, 9.2, "8B5CF6", PURPLE);
  addNumberedItem(s, 4, "Security review with AI", '"Review this Terraform for security vulnerabilities and apply best practices"', 0.4, 3.15, 9.2, "F59E0B", ORANGE);
  addNumberedItem(s, 5, "Test with AI-generated tests", '"Write Terratest tests for this module"', 0.4, 3.8, 9.2, "06B6D4", CYAN);
  addTagline(s, "VibeCoding IaC = Describe architecture, AI codes it, you understand and verify it");

  // ===== SLIDE 33: AI Tools for IaC =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "AI TOOLS FOR IAC", 0.4, 0.25, "EC4899");
  s.addText("AI Tools for IaC Development", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addTable(s, ["Tool", "Best For IaC", "How to Use"],
    [
      ["ChatGPT / Claude", "Full Terraform configs from description", "Describe architecture, get complete HCL"],
      ["GitHub Copilot", "Inline completion while writing .tf files", "Start typing resource block, tab complete"],
      ["Cursor", "Multi-file refactoring of Terraform modules", "Cmd+K to edit, Cmd+L to chat with codebase"],
      ["Claude Code", "Generate entire IaC projects with structure", "Describe infra, gets files, validates, plans"],
      ["checkov AI", "Auto-fix security misconfigurations", "checkov --compact with AI remediation hints"],
      ["Infracost AI", "Cost estimation + optimization suggestions", "Integrate with PR to show cost before apply"],
    ],
    0.4, 1.15, 9.2, [1.8, 3.2, 4.2]);
  addCard(s, "Pro Tip: Context is Everything", "When prompting AI for IaC, always specify: cloud provider, region, naming conventions, existing resources to integrate with, and security/compliance requirements.", 0.4, 4.05, 9.2, 0.65, PINK);

  // ===== SLIDE 34: Practice 3 =====
  s = practiceSlide(pres, "20 MIN", "Practice 3: Full IaC Pipeline", "Build a complete infrastructure pipeline with AI assistance");
  addCard(s, "Task", "Build a full IaC project: Terraform for infrastructure + Ansible for config + GitHub Actions CI/CD. Use AI at every step. Deploy a simple web app stack.", 0.4, 1.45, 9.2, 0.8, ORANGE);
  addNumberedItem(s, 1, "Generate Terraform", "AI creates: VPC, EC2, security groups with variables", 0.4, 2.4, 4.3, "3B82F6", BLUE);
  addNumberedItem(s, 2, "Generate Ansible", "AI creates playbook to install and configure nginx", 5.3, 2.4, 4.3, "8B5CF6", PURPLE);
  addNumberedItem(s, 3, "Generate CI/CD", "AI creates GitHub Actions workflow for plan + apply", 0.4, 3.05, 4.3, "22C55E", GREEN);
  addNumberedItem(s, 4, "Security scan", "Run checkov on Terraform, review findings with AI", 5.3, 3.05, 4.3, "F59E0B", ORANGE);
  addNumberedItem(s, 5, "Document everything", "AI generates README.md with architecture diagram", 0.4, 3.7, 9.2, "06B6D4", CYAN);
  addTagline(s, "This is the full DevOps IaC workflow -- you just built it in 20 minutes with AI");

  // ===== SLIDE 35: Real World: AWS Three-Tier =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "CASE STUDY 1", 0.4, 0.25, "F472B6");
  s.addText("Case Study: AWS Three-Tier Architecture", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCard(s, "Architecture", "Web tier: EC2 Auto Scaling + ALB\nApp tier: ECS Fargate containers\nData tier: RDS Aurora Multi-AZ\nAll provisioned and managed with Terraform", 0.4, 1.2, 4.3, 1.6, BLUE);
  addCard(s, "IaC Implementation", "1,200 lines of Terraform across 8 modules\nSeparate state for each tier\nCI/CD via GitHub Actions + Atlantis\nFull security group matrix\nCost: from 2 weeks manual -> 4 hours with AI", 5.3, 1.2, 4.3, 1.6, GREEN);
  addCodeBlock(s, "# AI Prompt that generated 80% of this:\n\"Create Terraform for AWS three-tier web app:\n- VPC with 3 AZs, public/private/data subnets\n- ALB in public subnets, EC2 ASG in private\n- RDS Aurora PostgreSQL in data subnets\n- All security groups with least-privilege rules\n- CloudWatch alarms for CPU, memory, DB connections\"", 0.4, 3.0, 9.2, 1.75, PINK);

  // ===== SLIDE 36: Real World: Multi-Cloud =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "CASE STUDY 2", 0.4, 0.25, "F472B6");
  s.addText("Case Study: Multi-Cloud with Terraform", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Managing AWS + GCP + Cloudflare from one Terraform codebase", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCard(s, "AWS Resources", "EKS cluster, RDS PostgreSQL\nS3 buckets, CloudFront CDN\nRoute53 DNS, WAF rules\nIAM roles and policies", 0.4, 1.4, 3.0, 1.8, BLUE);
  addCard(s, "GCP Resources", "GKE cluster (backup region)\nCloud SQL, Cloud Storage\nCloud Armor WAF\nGlobal Load Balancer", 3.6, 1.4, 3.0, 1.8, GREEN);
  addCard(s, "Cloudflare", "DNS records\nSSL/TLS management\nDDoS protection\nPage rules and caching", 6.7, 1.4, 2.9, 1.8, ORANGE);
  addCard(s, "Lessons Learned", "One Terraform workspace per cloud to avoid blast radius\nProvider aliases for multi-region in same cloud\nShared modules between clouds via abstraction layer\nAI dramatically reduced time to learn GCP provider\nTotal infra: 300 resources managed by 2 engineers", 0.4, 3.4, 9.2, 1.3, CYAN);

  // ===== SLIDE 37: Real World: Disaster Recovery =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "CASE STUDY 3", 0.4, 0.25, "F472B6");
  s.addText("Case Study: IaC-Enabled Disaster Recovery", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCard(s, "The Problem (Before IaC)", "Production outage: primary datacenter lost. Manual rebuild took 3 days. Documentation was incomplete. Config drift everywhere. $400,000 loss.", 0.4, 1.2, 4.3, 1.4, RED);
  addCard(s, "The Solution (With IaC)", "Terraform + Ansible for everything. DR site maintained as code. Full environment rebuilt in 47 minutes. RTO: 1 hour. RPO: 15 minutes.", 5.3, 1.2, 4.3, 1.4, GREEN);
  addCodeBlock(s, "# DR runbook became:\n# 1. git clone infra-repo\n# 2. cd environments/dr\n# 3. terraform apply -var-file=dr.tfvars\n# 4. ansible-playbook site.yml -i dr-hosts\n# 5. Verify health checks\n# Total time: 47 minutes vs 3 days", 0.4, 2.8, 9.2, 1.5, BLUE);
  addCard(s, "Key Takeaway", "IaC is not just about developer efficiency. It is your insurance policy. When disaster strikes, your infrastructure is a git clone away.", 0.4, 4.45, 9.2, 0.5, ORANGE);

  // ===== SLIDE 38: Real World: Cost Optimization =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "CASE STUDY 4", 0.4, 0.25, "F472B6");
  s.addText("Case Study: Cloud Cost Optimization with IaC", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  addCard(s, "The Challenge", "Startup spending $45,000/month on AWS. Most resources provisioned manually over 2 years. No clear ownership. Dev envs left running 24/7.", 0.4, 1.2, 4.3, 1.1, ORANGE);
  addCard(s, "IaC-Driven Solution", "Import all resources to Terraform. Tag everything with owner/env. Scheduled destroy of dev envs (18:00 weekdays). Right-sizing with AI analysis.", 5.3, 1.2, 4.3, 1.1, BLUE);
  addTable(s, ["Action", "Monthly Savings", "How IaC Helped"],
    [
      ["Dev env auto-shutdown", "$8,200", "terraform destroy on cron schedule"],
      ["Right-size instances", "$6,400", "AI analyzed usage, updated instance_type vars"],
      ["Remove zombie resources", "$4,100", "terraform state list revealed unused resources"],
      ["Reserved Instance planning", "$9,300", "Infracost showed 1yr RI savings before commit"],
      ["Total savings", "$28,000", "From $45K to $17K per month (62% reduction)"],
    ],
    0.4, 2.5, 9.2, [2.5, 1.8, 4.9]);

  // ===== SLIDE 39: Week Summary =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "SUMMARY", 0.4, 0.25, "60A5FA");
  s.addText("Week 6 Summary", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Infrastructure as Code with AI Prompting -- What We Covered", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCard(s, "Part 1: IaC & Terraform", "IaC fundamentals (what/why/how)\nTerraform: providers, resources, state\nPlan/Apply cycle\nVariables, locals, outputs", 0.4, 1.4, 4.3, 1.5, BLUE);
  addCard(s, "Part 2: Config Management", "Ansible playbooks and roles\nPulumi: code-first IaC\nGitOps principles\nIaC version control best practices", 5.3, 1.4, 4.3, 1.5, PURPLE);
  addCard(s, "Part 3: Production IaC", "Advanced Terraform (modules, workspaces)\nIaC testing strategy\nCI/CD for Terraform\nReal-world case studies", 0.4, 3.1, 4.3, 1.5, GREEN);
  addCard(s, "VibeCoding IaC", "AI generates 80% of boilerplate\nYou review, understand, and verify\nIterative prompting for complex infra\nSecurity reviews with AI assistance", 5.3, 3.1, 4.3, 1.5, PINK);
  addTagline(s, "IaC + VibeCoding = provision cloud infrastructure in hours, not days");

  // ===== SLIDE 40: Key Takeaways =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "KEY TAKEAWAYS", 0.4, 0.25, "60A5FA");
  s.addText("Key Takeaways", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  const takeaways = [
    ["IaC is non-negotiable", "Manual provisioning does not scale. Code everything."],
    ["Terraform is the standard", "Learn HCL -- it runs on every cloud provider"],
    ["State is sacred", "Remote state with locking. Never edit tfstate manually."],
    ["Plan before applying", "Always review terraform plan before terraform apply"],
    ["Ansible complements Terraform", "Provision with Terraform, configure with Ansible"],
    ["Security scan everything", "checkov and tflint on every PR. No exceptions."],
    ["AI accelerates IaC", "Let AI write the boilerplate, you validate the design"],
    ["GitOps is the goal", "Git is the source of truth. Automate everything else."],
  ];
  takeaways.forEach((t, i) => addNumberedItem(s, i+1, t[0], t[1], 0.4, 1.1 + i*0.48, 9.2, colors[i%6], tcolors[i%6]));

  // ===== SLIDE 41: Lab Assignment =====
  s = pres.addSlide(); s.background = { color: BG };
  addBadge(s, "LAB ASSIGNMENT", 0.4, 0.25, "F59E0B");
  s.addText("Lab 6: Build a Production-Ready IaC Stack", { x: 0, y: 0.6, w: 10, h: 0.45, fontSize: 24, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Due: Before Week 7 | Submit GitHub repo link + architecture diagram", { x: 0, y: 1.0, w: 10, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  addCard(s, "Lab 6A: Terraform Basics (40 points)", "Using AI assistance, create Terraform configs for:\n- VPC with public/private subnets in 2 AZs\n- EC2 instance with security group (SSH, HTTP)\n- S3 bucket for static website\n- Use variables for all configurable values\n- Run terraform plan and screenshot the output", 0.4, 1.4, 4.3, 2.1, BLUE);
  addCard(s, "Lab 6B: Ansible Config (30 points)", "Write an Ansible playbook (AI-assisted) that:\n- Installs nginx and Docker on Ubuntu\n- Deploys a static HTML page\n- Configures systemd service\n- Verifies with health check\n- Runs idempotently (test with --check)", 5.3, 1.4, 4.3, 2.1, CYAN);
  addCard(s, "Lab 6C: CI/CD Pipeline (30 points)", "GitHub Actions workflow that:\n- Runs terraform fmt --check\n- Runs terraform validate\n- Runs tflint\n- Runs checkov security scan\n- On main merge: terraform plan (no apply)\nBonus: Add Infracost cost estimation", 0.4, 3.7, 9.2, 1.1, GREEN);

  // ===== SLIDE 42: Closing =====
  s = pres.addSlide(); s.background = { color: BG };
  s.addShape("oval", { x: 2.5, y: -0.5, w: 3, h: 3, fill: { color: "8B5CF6", transparency: 88 } });
  s.addShape("oval", { x: 6, y: 3.5, w: 2.5, h: 2.5, fill: { color: "3B82F6", transparency: 88 } });
  s.addText("Week 6 Complete!", { x: 0.5, y: 1.2, w: 9, h: 0.8, fontSize: 32, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText("Infrastructure as Code with AI Prompting", { x: 0.5, y: 2.05, w: 9, h: 0.4, fontSize: 14, fontFace: "Arial", color: BLUE, align: "center" });
  s.addText("Questions & Discussion", { x: 0.5, y: 2.5, w: 9, h: 0.35, fontSize: 12, fontFace: "Arial", color: GRAY, align: "center" });
  const cboxes = [
    { t: "Next Steps", b: "Complete Lab 6A, 6B, 6C\nPush to GitHub\nSubmit repo link before Week 7" },
    { t: "Next Week", b: "Week 7: Kubernetes &\nContainer Orchestration" },
    { t: "Resources", b: "terraform.io/docs\nansible.com/docs\nregistry.terraform.io\ncheckov.io" }
  ];
  cboxes.forEach((b, i) => {
    const bx = 0.5 + i * 3.15;
    s.addShape("rect", { x: bx, y: 3.0, w: 2.9, h: 1.6, fill: { color: CARD_BG }, line: { color: "2A3560", width: 0.5 }, rectRadius: 0.08 });
    s.addText(b.t, { x: bx + 0.15, y: 3.1, w: 2.6, h: 0.35, fontSize: 12, fontFace: "Arial", bold: true, color: BLUE });
    s.addText(b.b, { x: bx + 0.15, y: 3.45, w: 2.6, h: 1.05, fontSize: 10, fontFace: "Arial", color: LIGHT, lineSpacingMultiple: 1.4 });
  });

  // Save
  const outPath = "/home/clawdbot/clawd/tmp/Week06_raw.pptx";
  await pres.writeFile({ fileName: outPath });
  console.log("Saved " + outPath + " (" + pres.slides.length + " slides)");
}

build().catch(e => { console.error(e); process.exit(1); });
