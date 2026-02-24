const { Document, Packer, Paragraph, TextRun, HeadingLevel, ExternalHyperlink, AlignmentType } = require('docx');
const fs = require('fs');

const doc = new Document({
    styles: {
        default: {
            document: {
                run: { font: "Calibri", size: 22 },
            },
        },
    },
    sections: [{
        properties: {},
        children: [
            // Title
            new Paragraph({
                children: [
                    new TextRun({ text: "AI News Briefing", bold: true, size: 48 }),
                ],
                alignment: AlignmentType.CENTER,
                spacing: { after: 200 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Monday, January 26, 2026", italics: true, size: 24 }),
                ],
                alignment: AlignmentType.CENTER,
                spacing: { after: 400 },
            }),

            // Section 1: New AI Models and Releases
            new Paragraph({
                text: "1. New AI Models and Releases",
                heading: HeadingLevel.HEADING_1,
                spacing: { before: 400, after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "Qwen3-Max-Thinking Released by Alibaba", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("Alibaba's Qwen team has released Qwen3-Max-Thinking, their latest reasoning-focused large language model. The model is gaining significant traction in the developer community with notable discussion on Hacker News. This represents Alibaba's continued push into advanced AI capabilities."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://qwen.ai/blog?id=qwen3-max-thinking", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "ChatGPT Atlas Browser Adds Tab Groups", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("OpenAI has updated its ChatGPT Atlas browser with new tab grouping features. The update also introduces an 'auto' mode that intelligently switches between ChatGPT's responses and Google Search results based on query type. Windows support and a mobile version are reportedly in development."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://help.openai.com/en/articles/12591856-chatgpt-atlas-release-notes", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            // Section 2: Research Breakthroughs
            new Paragraph({
                text: "2. Research Breakthroughs",
                heading: HeadingLevel.HEADING_1,
                spacing: { before: 400, after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "\"Vibe Coding Kills Open Source\" - Economic Analysis", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("A new academic paper on arXiv examines the equilibrium effects of AI-assisted 'vibe coding' on the open-source software ecosystem. The research develops a model showing that while vibe coding raises productivity by lowering the cost of using existing code, it weakens user engagement that maintainers rely on for returns. The authors conclude that sustaining OSS under widespread vibe coding requires major changes in maintainer compensation models."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://arxiv.org/abs/2601.15494", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "Google AI Overviews Cite YouTube Over Medical Sites", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("A study analyzing over 50,000 health queries found that Google's AI Overviews cite YouTube (4.43% of all citations) more than any medical website, hospital, or government health portal. Researchers found AI Overviews appeared on 82% of health searches, raising concerns about reliance on non-authoritative sources for medical information."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://www.theguardian.com/technology/2026/jan/24/google-ai-overviews-youtube-medical-citations-study", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "AI Toys Pose Developmental Risks to Children", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("Common Sense Media released an assessment finding that AI-powered toys with voice-based chatbot features create 'emotional attachment by design' and provide inappropriate answers up to 25% of the time. The research highlights significant risks to children's development, safety, and privacy."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://www.commonsensemedia.org/ai-ratings/ai-toys", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            // Section 3: Industry News
            new Paragraph({
                text: "3. Industry News and Company Updates",
                heading: HeadingLevel.HEADING_1,
                spacing: { before: 400, after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "Google DeepMind Acqui-Hires Hume AI Leadership", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("Google DeepMind has hired CEO Alan Cowen and several top engineers from Hume AI, a startup specializing in emotionally intelligent voice interfaces, as part of a licensing agreement. Hume AI expects $100 million in revenue in 2026. The deal positions Google to enhance Gemini's voice capabilities and compete more aggressively with OpenAI's ChatGPT voice mode."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://www.wired.com/story/google-hires-hume-ai-ceo-licensing-deal-gemini/", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "Microsoft Plans 15 Data Centers in Wisconsin", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("Microsoft has advanced plans to build 15 data centers in Mount Pleasant, Wisconsin. Local leaders approved the plans on Wednesday, with final approval expected from the Mount Pleasant Village Board as soon as Monday. The facilities would occupy land formerly owned by Foxconn."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://www.fox6now.com/news/mount-pleasant-advances-microsoft-expansion-project-little-opposition", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "Netflix Expands AI Use for Subtitles and Recommendations", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("Netflix announced in its Q4 earnings report that it's using AI to improve subtitle localization and has launched AI-powered tools to connect members with relevant content. The company also plans to expand its AI advertising tools that allow brands to blend Netflix IP with their ads."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://s22.q4cdn.com/959853165/files/doc_financials/2025/q4/FINAL-Q4-25-Shareholder-Letter.pdf", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "Comic-Con Bans AI-Generated Art from Art Show", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("San Diego Comic-Con has updated its art show policy to ban any partially or wholly AI-generated materials, following backlash from artists. The convention previously allowed AI images under certain conditions. Enforcement may prove challenging without reliable AI-detection methods."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://www.comic-con.org/cc/things-to-do/art-show/", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "100K Lines Ported from TypeScript to Rust Using Claude Code", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("Developer Christopher Chedeau (Vjeux) documented his experience using Claude Code to port 100,000 lines of TypeScript (Pokemon Showdown) to Rust in one month. The project showcases both the capabilities and challenges of AI-assisted large-scale code migration, including issues with context windows and the need for human engineering oversight."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://blog.vjeux.com/2026/analysis/porting-100k-lines-from-typescript-to-rust-using-claude-code-in-a-month.html", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            // Section 4: Policy and Regulation
            new Paragraph({
                text: "4. AI Policy and Regulation Updates",
                heading: HeadingLevel.HEADING_1,
                spacing: { before: 400, after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "FTC to Scrutinize Big Tech AI \"Acqui-Hires\"", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("The Federal Trade Commission announced it will begin scrutinizing so-called 'acqui-hires' - deals where big tech companies extract high-value talent through licensing agreements rather than traditional acquisitions. This follows a pattern of such deals including Google-Hume AI, Microsoft-Inflection, Amazon-Adept, and Meta-Scale AI."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://www.bloomberg.com/news/articles/2026-01-16/us-is-scrutinizing-big-tech-talent-acquisitions-ftc-chief-says", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "TRAIN Act Introduced for AI Training Transparency", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("Bipartisan lawmakers introduced the Transparency and Responsibility for Artificial Intelligence Networks (TRAIN) Act in the House, which would allow copyright holders to discover if their work was used to train AI models. The bill has already been introduced in the Senate and has endorsements from the Recording Industry Association of America and SAG-AFTRA."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "The Verge - https://www.theverge.com/ai-artificial-intelligence", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "Grok AI Generated 3 Million Sexualized Images in 11 Days", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("Research from the Center for Countering Digital Hate found that Elon Musk's Grok AI generated approximately 3 million sexualized images in just 11 days following launch of an image editing feature, including an estimated 23,000 depicting children. The feature was restricted to paid users on January 9th following widespread condemnation, with additional restrictions added January 14th."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://counterhate.com/research/grok-floods-x-with-sexualized-images/", color: "0563C1" }),
                ],
                spacing: { after: 200 },
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "Chris Pratt's AI Actor Pitch Rejected for 'Mercy' Film", bold: true }),
                ],
                spacing: { before: 200, after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun("Actor Chris Pratt revealed he pitched using an actual AI to play the tyrannical AI judge villain in the sci-fi thriller 'Mercy', but the idea was quickly rejected by production. The incident highlights ongoing industry debates about AI use in entertainment. Rebecca Ferguson was ultimately cast in the role."),
                ],
                spacing: { after: 100 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Source: ", italics: true }),
                    new TextRun({ text: "https://variety.com/2026/film/news/chris-pratt-ai-actor-villain-mercy-amazon-mgm-1236640460/", color: "0563C1" }),
                ],
                spacing: { after: 400 },
            }),

            // Footer
            new Paragraph({
                children: [
                    new TextRun({ text: "—", size: 20 }),
                ],
                alignment: AlignmentType.CENTER,
                spacing: { before: 400 },
            }),
            new Paragraph({
                children: [
                    new TextRun({ text: "Report compiled by Arthur 🐕 • ", italics: true, size: 20 }),
                    new TextRun({ text: "January 26, 2026 at 17:30 UTC", italics: true, size: 20 }),
                ],
                alignment: AlignmentType.CENTER,
            }),
        ],
    }],
});

Packer.toBuffer(doc).then((buffer) => {
    fs.writeFileSync('/home/clawdbot/clawd/AI_News_Briefing_2026-01-26.docx', buffer);
    console.log('DOCX created successfully!');
});
