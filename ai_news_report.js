const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, LevelFormat, TableOfContents,
        HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak, 
        PageNumber, ExternalHyperlink } = require('docx');
const fs = require('fs');

// Helper functions
const heading1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 400, after: 200 },
  children: [new TextRun({ text, bold: true, size: 36, font: "Arial", color: "1A365D" })]
});

const heading2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 300, after: 150 },
  children: [new TextRun({ text, bold: true, size: 28, font: "Arial", color: "2B6CB0" })]
});

const para = (text, spacing = { after: 200 }) => new Paragraph({
  spacing,
  alignment: AlignmentType.JUSTIFIED,
  children: [new TextRun({ text, font: "Arial", size: 22 })]
});

const boldPara = (label, text) => new Paragraph({
  spacing: { after: 150 },
  children: [
    new TextRun({ text: label + ": ", bold: true, font: "Arial", size: 22 }),
    new TextRun({ text, font: "Arial", size: 22 })
  ]
});

const bulletItem = (text, ref = "bullets") => new Paragraph({
  numbering: { reference: ref, level: 0 },
  spacing: { after: 100 },
  children: [new TextRun({ text, font: "Arial", size: 22 })]
});

const linkPara = (text, url) => new Paragraph({
  spacing: { after: 100 },
  children: [
    new TextRun({ text: "🔗 ", font: "Arial", size: 22 }),
    new ExternalHyperlink({
      children: [new TextRun({ text, font: "Arial", size: 22, color: "2B6CB0", underline: {} })],
      link: url
    })
  ]
});

// Border style
const border = { style: BorderStyle.SINGLE, size: 8, color: "CBD5E0" };
const borders = { top: border, bottom: border, left: border, right: border };

// Create news item section
const newsItem = (title, description, source, url) => [
  new Paragraph({
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text: "▸ " + title, bold: true, font: "Arial", size: 24, color: "2D3748" })]
  }),
  para(description),
  linkPara(source, url),
  new Paragraph({ spacing: { after: 150 } })
];

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "Arial", size: 22, color: "2D3748" }
      }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1A365D" },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2B6CB0" },
        paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 1 }
      }
    ]
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  sections: [
    // Cover Page
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 }
        }
      },
      children: [
        new Paragraph({ spacing: { before: 3000 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "🤖 AI NEWS DIGEST", bold: true, size: 72, font: "Arial", color: "1A365D" })]
        }),
        new Paragraph({ spacing: { before: 400 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Daily Intelligence Report", size: 36, font: "Arial", color: "4A5568" })]
        }),
        new Paragraph({ spacing: { before: 800 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "January 29, 2026", size: 28, font: "Arial", color: "718096" })]
        }),
        new Paragraph({ spacing: { before: 200 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Compiled by Arthur 🐕", size: 24, font: "Arial", color: "718096", italics: true })]
        }),
        new Paragraph({ spacing: { before: 3000 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Prepared for Anirach", size: 22, font: "Arial", color: "A0AEC0" })]
        })
      ]
    },
    // Main Content
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [new TextRun({ text: "AI News Digest - January 29, 2026", italics: true, size: 20, font: "Arial", color: "718096" })]
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", size: 20, font: "Arial" }),
              new TextRun({ children: [PageNumber.CURRENT], size: 20, font: "Arial" }),
              new TextRun({ text: " of ", size: 20, font: "Arial" }),
              new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 20, font: "Arial" })
            ]
          })]
        })
      },
      children: [
        // Table of Contents
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
          children: [new TextRun({ text: "TABLE OF CONTENTS", bold: true, size: 32, font: "Arial", color: "1A365D" })]
        }),
        new TableOfContents("Table of Contents", {
          hyperlink: true,
          headingStyleRange: "1-2"
        }),
        new Paragraph({ children: [new PageBreak()] }),

        // Executive Summary
        heading1("Executive Summary"),
        para("This report provides a comprehensive overview of the most significant AI developments from the past 24 hours, covering new model releases, research breakthroughs, major industry moves, and regulatory updates. Key highlights include massive investment discussions around OpenAI, new model releases from Chinese AI companies, and evolving regulatory frameworks in the US and Europe."),
        new Paragraph({ children: [new PageBreak()] }),

        // Section 1: New AI Models and Releases
        heading1("1. New AI Models and Releases"),
        
        ...newsItem(
          "OpenAI Launches Prism Research Platform with GPT-5.2 Thinking",
          "OpenAI unveiled Prism, a cloud-based research platform built on acquired Crixet LaTeX infrastructure and powered by GPT-5.2 Thinking. Designed for scientific paper writing, it allows researchers to draft papers, search literature, and use AI to create, refactor, and reason over equations, citations, and figures with real-time collaboration.",
          "Gizmodo",
          "https://gizmodo.com/openais-new-product-helps-you-do-vibe-physics-like-travis-kalanick-2000714626"
        ),

        ...newsItem(
          "Anthropic Expands Model Context Protocol with New UI Framework",
          "Anthropic has extended its Model Context Protocol (MCP) with a new UI framework for developers to build sophisticated AI application interfaces. The expansion enables seamless interaction with Claude and other AI systems while maintaining enterprise security standards.",
          "The New Stack",
          "https://thenewstack.io/anthropic-extends-mcp-with-an-app-framework/"
        ),

        ...newsItem(
          "Chinese AI Companies Accelerate Model Releases - Qwen3-Max-Thinking & Ernie 5.0",
          "Alibaba announced Qwen3-Max-Thinking claiming to outperform major US rivals on 'Humanity's Last Exam' benchmark. Baidu released Ernie 5.0, driving shares to nearly three-year highs. Z.ai released free GLM 4.7 with overwhelming demand.",
          "CNBC",
          "https://www.cnbc.com/2026/01/28/chinese-tech-companies-accelerate-ai-model-rollouts-us-rivals-deepseek-moonshot-kimi.html"
        ),

        ...newsItem(
          "Moonshot AI Releases Open-Source Kimi K2.5 Model",
          "Chinese AI company Moonshot released Kimi K2.5, trained on 15 trillion mixed visual and text tokens, outperforming Gemini 3 Pro on coding benchmarks (SWE-Bench) and beating GPT 5.2 and Claude Opus 4.5 on video understanding (VideoMMMU). Also launched Kimi Code, rivaling Claude Code.",
          "TechCrunch",
          "https://techcrunch.com/2026/01/27/chinas-moonshot-releases-a-new-open-source-model-kimi-k2-5-and-a-coding-agent/"
        ),

        ...newsItem(
          "Google Rolls Out AI Plus Tier ($7.99/month) and NotebookLM Integration",
          "Google launched AI Plus at $7.99/month offering 90 daily Gemini 3 Flash Thinking prompts, 30 Gemini 3 Pro prompts, and 128K token context (up from 32K). NotebookLM now integrates with Gemini app on iOS for enhanced research capabilities.",
          "9to5Google",
          "https://9to5google.com/2026/01/28/gemini-app-google-ai-plus/"
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // Section 2: Research Breakthroughs
        heading1("2. Research Breakthroughs"),

        ...newsItem(
          "World Models: The Next AI Revolution",
          "Scientific American reports that 'world models' could unlock the next revolution in AI. Yann LeCun launched AMI Labs focusing on world models technology that mimics how humans learn through visual experience rather than text. Research shows AI agents using world models can improve behavior by 'imagining' future scenarios.",
          "Scientific American",
          "https://www.scientificamerican.com/article/world-models-could-unlock-the-next-revolution-in-artificial-intelligence/"
        ),

        ...newsItem(
          "Brain-AI Parallel Processing Discovery",
          "Research published in Nature Communications reveals that human brains and AI models build meaning incrementally over time through similar computational approaches. This finding provides new insights for developing more brain-like AI architectures.",
          "Nature Communications via HumAI Blog",
          "https://www.humai.blog/ai-news-trends-january-2026-complete-monthly-digest/"
        ),

        ...newsItem(
          "Jensen Huang at Davos 2026: The Five Key AI Breakthroughs",
          "Nvidia CEO Jensen Huang outlined major AI advances: (1) Models are significantly more grounded with reduced hallucinations, (2) Enhanced reasoning capabilities, (3) Better planning abilities, (4) Improved reliability for serious work, (5) Progress toward agentic AI systems.",
          "Forbes",
          "https://www.forbes.com/sites/bernardmarr/2026/01/22/davos-2026-jensen-huang-on-the-five-layer-ai-cake-the-ai-bubble-and-key-ai-breakthroughs/"
        ),

        ...newsItem(
          "AutoML 2.0: AI That Improves AI",
          "OpenAI is developing an 'automated AI researcher' system expected to match less experienced researchers by Fall 2026. This represents an evolution of Google's 2017 AutoML concept, now scaled for the modern AI era, potentially accelerating AI development.",
          "SF Examiner",
          "https://www.sfexaminer.com/silicon-valley-wants-to-build-ai-that-can-improve-ai-on-its-own/article_5307b04c-f5b3-59f7-90b3-2bff1aebe2ed.html"
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // Section 3: Industry News & Company Updates
        heading1("3. Industry News and Company Updates"),

        ...newsItem(
          "Big Tech Eyes $60B OpenAI Investment",
          "Nvidia, Microsoft, and Amazon are reportedly in talks to collectively invest up to $60 billion in OpenAI—one of the largest private capital infusions in technology history. This reflects the capital-intensive nature of generative AI and mounting competition for influence over foundational AI platforms.",
          "Startup News FYI",
          "https://startupnews.fyi/2026/01/29/big-tech-60b-openai-investment/"
        ),

        ...newsItem(
          "Microsoft Q2 2026 Earnings: OpenAI Accounts for 45% of Cloud Backlog",
          "Microsoft disclosed that OpenAI accounts for 45% of its cloud backlog during Q2 earnings. Anthropic announced plans to buy $30 billion in cloud services and contract up to a gigawatt of additional computing capacity from Microsoft.",
          "CNBC",
          "https://www.cnbc.com/2026/01/28/microsoft-msft-q2-earnings-report-2026.html"
        ),

        ...newsItem(
          "Anthropic Fundraising Exceeds $10B with Microsoft & Nvidia Participation",
          "Anthropic's latest funding round closed above $10 billion. Microsoft and Nvidia announced plans to invest up to $5 billion and $10 billion respectively. Anthropic also launched a new 'Labs' division with former CPO Mike Krieger.",
          "CNBC",
          "https://www.cnbc.com/2026/01/27/anthropic-fundraising-microsoft-nvidia.html"
        ),

        ...newsItem(
          "Anthropic Launches Interactive Claude Apps with Workplace Integration",
          "Anthropic introduced interactive Claude applications embedding Slack, Figma, and Asana directly in the AI chat interface. Claude Code continues viral success beyond developers—even Microsoft has widely adopted it internally despite competing with GitHub Copilot.",
          "VentureBeat",
          "https://venturebeat.com/ai/anthropic-embeds-slack-figma-and-asana-inside-claude-turning-ai-chat-into-a"
        ),

        ...newsItem(
          "Big Tech Earnings Show AI Spending Pressure",
          "Investors are scrutinizing $280 billion potentially at risk as OpenAI remains unprofitable while leading AI development. Analysts expect Google to spend over $115 billion in 2026, with deals inked with both OpenAI and Anthropic.",
          "Reuters",
          "https://www.reuters.com/business/autos-transportation/investors-punish-big-tech-ai-spending-that-delivers-slower-growth-2026-01-29/"
        ),

        ...newsItem(
          "Yann LeCun Leaves Meta, Founds AMI Labs",
          "Turing Award winner Yann LeCun departed Meta after professional disagreements, warning that current LLM approaches won't achieve AGI. Founded AMI Labs in Paris focusing on 'world models' as an alternative to language-only training.",
          "HumAI Blog",
          "https://www.humai.blog/ai-news-trends-january-2026-complete-monthly-digest/"
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // Section 4: AI Policy and Regulation
        heading1("4. AI Policy and Regulation Updates"),

        ...newsItem(
          "Texas TRAIGA Act Takes Effect January 1, 2026",
          "The Texas Responsible Artificial Intelligence Governance Act (TRAIGA, C.S.H.B. 149) took effect, regulating certain AI uses, providing civil penalties and Attorney General enforcement, and includes a regulatory sandbox for testing under defined conditions.",
          "Software Improvement Group",
          "https://www.softwareimprovementgroup.com/blog/us-ai-legislation-overview/"
        ),

        ...newsItem(
          "TRUMP AMERICA AI Act Proposed for Federal Preemption",
          "Senator Marsha Blackburn proposed the TRUMP AMERICA AI Act ('The Republic Unifying Meritocratic') representing the most ambitious congressional attempt to establish unified federal AI governance with comprehensive regulation across industries.",
          "Mondaq",
          "https://www.mondaq.com/unitedstates/new-technology/1736408/the-trump-america-ai-act-federal-preemption-meets-comprehensive-regulation"
        ),

        ...newsItem(
          "US Updates AI Chip Export Controls",
          "On January 13, 2026, the US Department of Commerce revised license review posture for NVIDIA H200 and AMD MI325X equivalent chips from 'presumption of denial' to 'case-by-case review', affecting global AI infrastructure access.",
          "Mayer Brown",
          "https://www.mayerbrown.com/en/insights/publications/2026/01/administration-policies-on-advanced-ai-chips-codified"
        ),

        ...newsItem(
          "EU EDPB/EDPS Issue Joint Opinion on Digital Omnibus and AI Act",
          "On January 20, 2026, the European Data Protection Board and European Data Protection Supervisor issued a joint opinion on the Digital Omnibus package related to AI, addressing the European Commission's December 2025 proposal.",
          "MediaLaws",
          "https://www.medialaws.eu/ai-act-and-digital-omnibus-the-edpb-and-edps-issued-their-joint-opinion/"
        ),

        ...newsItem(
          "2026 US Privacy Laws Update: Stricter AI Marketing Rules",
          "Multiple US states implementing stricter limits on targeted ads to minors, profiling, cross-site tracking, and geolocation-based marketing. Oregon bans sale of precise location data; multiple states require opt-out/opt-in for AI-driven ads.",
          "Ketch",
          "https://www.ketch.com/blog/posts/us-privacy-laws-2026"
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // Key Takeaways
        heading1("Key Takeaways"),
        
        bulletItem("Massive Capital Flows: $60B potential OpenAI investment, $10B+ Anthropic round, and $115B+ expected Google AI spending in 2026"),
        bulletItem("Chinese AI Surge: Alibaba, Baidu, Moonshot accelerating releases; models approaching or exceeding US capabilities on some benchmarks"),
        bulletItem("Agentic AI Focus: Workplace integrations (Slack, Figma, Asana), autonomous researchers, and self-improving systems"),
        bulletItem("Regulatory Momentum: Texas TRAIGA now active, federal TRUMP AMERICA AI Act proposed, EU AI Act enforcement ramping up"),
        bulletItem("World Models Emergence: Yann LeCun's AMI Labs and others exploring alternatives to pure LLM approaches"),
        bulletItem("Infrastructure is King: Compute, chips, and cloud capacity are the new competitive moats"),

        new Paragraph({ spacing: { before: 400 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 600 },
          children: [new TextRun({ text: "— End of Report —", italics: true, size: 20, font: "Arial", color: "718096" })]
        })
      ]
    }
  ]
});

// Generate and save
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('/home/clawdbot/clawd/AI_News_Digest_2026-01-29.docx', buffer);
  console.log('Report saved to /home/clawdbot/clawd/AI_News_Digest_2026-01-29.docx');
});
