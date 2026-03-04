#!/usr/bin/env python3
"""Generate: Leading with AI in Healthcare report"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

TEMPLATE_PATH = '/workspace/templates/reports/General_Report_Template.docx'
OUTPUT_PATH   = '/workspace/tmp/Leading_AI_Healthcare_Report.docx'

NAVY  = RGBColor(0x17, 0x36, 0x5D)
BLUE  = RGBColor(0x4F, 0x81, 0xBD)
GRAY  = RGBColor(0x40, 0x40, 0x40)
BODY  = RGBColor(0x33, 0x33, 0x33)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
THEAD = '17365D'
TROW1 = 'EBF3FB'
TROW2 = 'FFFFFF'

def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)

def page_break(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(0)
    run = p.add_run()
    br  = OxmlElement('w:br')
    br.set(qn('w:type'), 'page')
    run._r.append(br)

def add_h1(doc, text):
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after  = Pt(6)
    r = p.add_run(text)
    r.font.size = Pt(16); r.font.color.rgb = NAVY
    r.font.bold = True;   r.font.name = 'Arial'

def add_h2(doc, text):
    p = doc.add_paragraph(style='Heading 2')
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after  = Pt(4)
    r = p.add_run(text)
    r.font.size = Pt(13); r.font.color.rgb = BLUE
    r.font.bold = True;   r.font.name = 'Arial'

def add_h3(doc, text):
    p = doc.add_paragraph(style='Heading 3')
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(3)
    r = p.add_run(text)
    r.font.size = Pt(12); r.font.color.rgb = GRAY
    r.font.bold = True;   r.font.name = 'Arial'

def add_body(doc, text):
    p = doc.add_paragraph(style='Normal')
    p.paragraph_format.alignment    = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(6)
    r = p.add_run(text)
    r.font.size = Pt(11); r.font.color.rgb = BODY; r.font.name = 'Arial'

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.alignment    = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(3)
    p.paragraph_format.left_indent  = Inches(0.25 * (level + 1))
    r = p.add_run(text)
    r.font.size = Pt(11); r.font.color.rgb = BODY; r.font.name = 'Arial'

def add_spacer(doc, pts=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(pts)

def add_caption(doc, text):
    p = doc.add_paragraph(style='Caption')
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.size = Pt(9); r.font.italic = True; r.font.color.rgb = BLUE

def make_table(doc, headers, rows, col_widths=None):
    n = len(headers)
    table = doc.add_table(rows=1+len(rows), cols=n)
    table.style     = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    if col_widths:
        for i, w in enumerate(col_widths):
            for cell in table.columns[i].cells:
                cell.width = Inches(w)
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_bg(cell, THEAD)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True; r.font.color.rgb = WHITE
        r.font.size = Pt(10); r.font.name = 'Arial'
    for ri, row_data in enumerate(rows):
        row = table.rows[ri+1]
        bg  = TROW1 if ri % 2 == 0 else TROW2
        for ci, txt in enumerate(row_data):
            cell = row.cells[ci]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_bg(cell, bg)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(txt))
            r.font.size = Pt(10); r.font.color.rgb = BODY; r.font.name = 'Arial'
    return table

def build():
    doc = Document(TEMPLATE_PATH)
    for p in list(doc.paragraphs):
        p._element.getparent().remove(p._element)
    for t in list(doc.tables):
        t._element.getparent().remove(t._element)

    # COVER
    for _ in range(5):
        add_spacer(doc, 14)

    def centre_para(text, size, color, bold=False, italic=False, after=6):
        p = doc.add_paragraph()
        p.paragraph_format.alignment   = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(after)
        r = p.add_run(text)
        r.font.size = Pt(size); r.font.color.rgb = color
        r.font.bold = bold; r.font.italic = italic; r.font.name = 'Arial'
        return p

    p_rule = doc.add_paragraph()
    p_rule.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = p_rule.add_run(u'\u2501' * 42)
    rr.font.color.rgb = BLUE; rr.font.size = Pt(11)

    add_spacer(doc, 10)
    centre_para('Leading with AI in Healthcare', 28, NAVY, bold=True)
    centre_para('Data, Infrastructure and AI-Cowork', 22, BLUE, bold=True, after=10)
    centre_para('A Strategic Framework for Healthcare Digital Transformation', 14, GRAY, italic=True)
    add_spacer(doc, 6)
    p_rule2 = doc.add_paragraph()
    p_rule2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr2 = p_rule2.add_run(u'\u2501' * 42)
    rr2.font.color.rgb = BLUE; rr2.font.size = Pt(11)
    add_spacer(doc, 36)

    mt = doc.add_table(rows=4, cols=2)
    mt.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta = [('Author','Anirach Mingkhwan'),('Date','March 2026'),
            ('Version','v1.0'),('Classification','Strategic -- Healthcare Leadership')]
    for i,(label,val) in enumerate(meta):
        mt.rows[i].cells[0].width = Inches(1.8)
        mt.rows[i].cells[1].width = Inches(4.0)
        lp = mt.rows[i].cells[0].paragraphs[0]
        lp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        lr = lp.add_run(label+':')
        lr.font.bold=True; lr.font.color.rgb=NAVY; lr.font.size=Pt(11); lr.font.name='Arial'
        vp = mt.rows[i].cells[1].paragraphs[0]
        vr = vp.add_run(val)
        vr.font.size=Pt(11); vr.font.color.rgb=BODY; vr.font.name='Arial'

    add_spacer(doc, 36)
    centre_para('(c) 2026 Anirach Mingkhwan -- All Rights Reserved', 9, GRAY)
    page_break(doc)

    # TOC
    p_toc = doc.add_paragraph()
    p_toc.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_toc = p_toc.add_run('Table of Contents')
    r_toc.font.size=Pt(16); r_toc.font.bold=True
    r_toc.font.color.rgb=NAVY; r_toc.font.name='Arial'
    add_spacer(doc, 6)

    toc_entries = [
        ('1.','The AI Imperative in Healthcare','3'),
        ('2.','The Data Foundation','5'),
        ('3.','AI Infrastructure for Healthcare','8'),
        ('4.','AI-Cowork -- Human-AI Collaboration','11'),
        ('5.','Implementation Roadmap','14'),
        ('6.','Future Outlook','17'),
        ('','References','20'),
    ]
    tt = doc.add_table(rows=len(toc_entries), cols=3)
    tt.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i,(num,title,pg) in enumerate(toc_entries):
        tt.rows[i].cells[0].width = Inches(0.4)
        tt.rows[i].cells[1].width = Inches(5.6)
        tt.rows[i].cells[2].width = Inches(0.5)
        np2 = tt.rows[i].cells[0].paragraphs[0]
        nr = np2.add_run(num)
        nr.font.bold=bool(num); nr.font.size=Pt(11)
        nr.font.color.rgb=NAVY if num else BODY; nr.font.name='Arial'
        tp2 = tt.rows[i].cells[1].paragraphs[0]
        tr2 = tp2.add_run(title)
        tr2.font.size=Pt(11); tr2.font.bold=bool(num)
        tr2.font.color.rgb=NAVY if num else GRAY; tr2.font.name='Arial'
        pp2 = tt.rows[i].cells[2].paragraphs[0]
        pp2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        pr = pp2.add_run(pg)
        pr.font.size=Pt(11); pr.font.color.rgb=BODY; pr.font.name='Arial'

    add_spacer(doc, 6)
    page_break(doc)

    # S1
    add_h1(doc, '1. The AI Imperative in Healthcare')
    add_body(doc, 'Healthcare is at an inflection point. The convergence of exponential data growth, an aging global population, persistent clinician burnout, and the rapid maturation of artificial intelligence technologies has created both an urgent challenge and an extraordinary opportunity for health systems worldwide. Leaders who act decisively today will shape the quality, accessibility, and sustainability of care for the next generation.')

    add_h2(doc, '1.1 Why AI in Healthcare Matters Now')

    add_h3(doc, 'The Data Explosion')
    add_body(doc, 'Global healthcare data is projected to reach 2,314 exabytes by 2025, growing at a compound annual rate of 36% -- faster than any other industry (RBC Capital Markets, 2020). Electronic Health Records, medical imaging, genomic sequencing, remote patient monitoring, and claims data collectively generate a volume of information no human team can fully synthesise. AI provides the analytical infrastructure needed to extract actionable signal from this unprecedented volume.')

    add_h3(doc, 'Aging Populations and Workforce Strain')
    add_body(doc, 'By 2050, the number of people over 60 will double to 2.1 billion globally (WHO, 2021). Thailand faces one of the most rapid demographic transitions in ASEAN: the proportion of citizens aged 60+ is projected to exceed 30% by 2040. Simultaneously, the WHO estimates a global shortfall of 10 million health workers by 2030. AI-augmented care delivery is not a luxury -- it is a structural necessity.')

    add_h3(doc, 'Clinician Burnout Crisis')
    add_body(doc, 'A 2022 Medscape survey reported that 47% of U.S. physicians experienced burnout, with administrative burden -- documentation, prior authorisations, and data entry -- cited as the leading cause. Studies from NHS England and multiple Asian health systems echo this finding. AI-driven documentation automation, intelligent scheduling, and decision support can reclaim meaningful clinical time, reduce errors, and restore professional satisfaction.')

    add_h2(doc, '1.2 Current State of Global AI Adoption in Healthcare')
    add_body(doc, 'AI adoption in healthcare has moved from pilot projects to scaled deployments, with significant variation across regions and institution types. Key milestones as of 2025:')
    add_bullet(doc, 'FDA cleared over 950 AI/ML-enabled medical devices by end-2024, the majority in radiology (FDA, 2024).')
    add_bullet(doc, 'The global healthcare AI market reached USD 22.4 billion in 2023 and is projected to grow at 37.5% CAGR through 2030 (Grand View Research, 2024).')
    add_bullet(doc, 'Large language models such as Google Med-PaLM 2 achieved expert-level performance on the U.S. Medical Licensing Examination (Singhal et al., Nature, 2023).')
    add_bullet(doc, 'Federated learning pilots across hospital networks now enable collaborative model training without sharing raw patient data (NVIDIA FLARE, 2024).')
    add_bullet(doc, "Thailand's Ministry of Public Health launched AI-assisted radiology in 2023, deploying chest X-ray AI screening across 30 provincial hospitals.")

    add_h2(doc, '1.3 Key Statistics at a Glance')
    make_table(doc,
        headers=['Metric','Value','Source'],
        rows=[
            ['Global healthcare AI market (2023)','USD 22.4 billion','Grand View Research, 2024'],
            ['Projected CAGR (2023-2030)','37.5%','Grand View Research, 2024'],
            ['FDA-cleared AI medical devices','950+ (end-2024)','FDA, 2024'],
            ['Healthcare data growth rate','36% CAGR','RBC Capital Markets, 2020'],
            ['Global health worker shortfall by 2030','10 million','WHO, 2021'],
            ['Physician burnout rate (U.S.)','47%','Medscape, 2022'],
            ['Thailand population over 60 by 2040','>30%','NESDC Thailand, 2023'],
        ],
        col_widths=[3.0,1.8,2.2])
    add_caption(doc, 'Table 1.1 -- Key AI in Healthcare Statistics')
    add_spacer(doc, 4)
    page_break(doc)

    # S2
    add_h1(doc, '2. The Data Foundation')
    add_body(doc, 'No AI initiative can outperform the quality and structure of the data underlying it. Before deploying any machine learning model, healthcare organisations must build a robust, governed, and interoperable data foundation. This section defines the healthcare data landscape, addresses quality challenges, and outlines the governance and interoperability frameworks that enable responsible AI.')

    add_h2(doc, '2.1 Healthcare Data Types')
    add_body(doc, 'Healthcare generates rich, heterogeneous data modalities. Each type carries distinct characteristics, volume profiles, and analytical potential:')
    make_table(doc,
        headers=['Data Type','Examples','Volume','AI Potential'],
        rows=[
            ['Electronic Health Records (EHR)','Clinical notes, diagnoses, medications, labs','High','Predictive risk, NLP extraction'],
            ['Medical Imaging','X-rays, CT, MRI, histopathology slides','Very High','Computer vision, diagnosis support'],
            ['Genomics & Omics','DNA sequencing, proteomics, metabolomics','Extreme','Personalised medicine, drug discovery'],
            ['Wearables & IoT','Heart rate, SpO2, glucose, activity data','Continuous','Chronic disease monitoring, alerting'],
            ['Claims & Administrative','Billing codes, utilisation, cost data','High','Cost prediction, fraud detection'],
            ['Social Determinants','Housing, income, education, lifestyle','Medium','Population health, equity analysis'],
            ['Patient-Reported Outcomes','Surveys, symptom journals, PRO instruments','Low-Medium','Quality of life, adherence tracking'],
        ],
        col_widths=[2.0,2.5,1.0,2.0])
    add_caption(doc, 'Table 2.1 -- Healthcare Data Types and AI Potential')
    add_spacer(doc, 4)

    add_h2(doc, '2.2 Data Quality Challenges')
    add_h3(doc, 'Missing and Incomplete Data')
    add_body(doc, 'Studies show that 30-80% of clinical data fields in EHRs contain missing values (Sterne et al., BMJ, 2009). Missing data arises from inconsistent documentation practices, system migrations, and care fragmentation across providers. Strategies including multiple imputation, federated records, and structured clinical templates can mitigate this significantly.')

    add_h3(doc, 'Data Inconsistency and Duplication')
    add_body(doc, 'Patient records are frequently duplicated across systems -- a 2019 study found 8-12% of hospital EHR records are duplicate entries (AHIMA, 2019). Inconsistent coding practices compound the problem. Master Patient Index (MPI) systems and probabilistic record linkage are essential remedies.')

    add_h3(doc, 'Algorithmic Bias')
    add_body(doc, "AI models trained on historically biased datasets can perpetuate or amplify health inequities. A landmark study (Obermeyer et al., Science, 2019) demonstrated that a widely used commercial algorithm systematically underestimated Black patients' health needs. Healthcare AI programmes must audit training datasets for demographic representation and track model performance across subgroups.")

    add_h2(doc, '2.3 Data Governance and Privacy')
    add_body(doc, 'Healthcare data governance must balance innovation enablement with the highest standards of patient privacy and regulatory compliance. Three major frameworks govern healthcare AI data use:')
    make_table(doc,
        headers=['Framework','Jurisdiction','Key Requirements','Relevance to AI'],
        rows=[
            ['HIPAA (1996, updated)','United States','PHI safeguards, de-identification standards, BAAs','Governs AI training data, model access, vendor contracts'],
            ['GDPR (2018)','European Union','Lawful basis, data minimisation, right to explanation','Article 22 restricts automated decision-making affecting individuals'],
            ['PDPA Thailand (2022)','Thailand','Explicit consent, data localisation, DPO appointment, 72-hr breach notification','Applies to all Thai hospitals using AI with patient data'],
        ],
        col_widths=[1.8,1.4,2.2,2.1])
    add_caption(doc, 'Table 2.2 -- Healthcare Data Privacy Frameworks')
    add_spacer(doc, 4)
    add_body(doc, 'Best-practice governance programmes combine technical controls (encryption, access logging, differential privacy) with organisational structures: a Data Governance Committee, a designated Data Protection Officer, and regular privacy impact assessments for AI initiatives.')

    add_h2(doc, '2.4 Interoperability Standards')
    add_body(doc, 'Data locked in proprietary systems cannot power AI at scale. Interoperability standards enable the data exchange that AI requires:')
    add_bullet(doc, 'HL7 FHIR (Fast Healthcare Interoperability Resources): The current gold standard for API-based clinical data exchange. FHIR R4 is mandated for EHR vendors in the U.S. and is increasingly adopted across ASEAN. FHIR enables real-time data access for AI inference pipelines.')
    add_bullet(doc, "DICOM (Digital Imaging and Communications in Medicine): Universal format for radiological images. AI imaging models ingest DICOM directly; DICOM SR (Structured Reporting) allows AI findings to flow back into clinical workflows.")
    add_bullet(doc, "ICD-10 / ICD-11: The WHO's International Classification of Diseases provides standardised diagnostic coding. ICD-11, adopted from 2022, includes enhanced granularity for AI-based phenotyping and mortality analytics.")
    add_bullet(doc, 'SNOMED CT: Comprehensive clinical terminology enabling semantic interoperability for symptom, procedure, and finding documentation -- critical for NLP-based clinical note mining.')
    add_bullet(doc, 'openEHR: An archetype-based open standard for clinical data modelling, widely adopted in Nordic countries and emerging in Southeast Asia.')

    add_h2(doc, '2.5 Real-World Data Infrastructure Excellence')
    add_body(doc, 'Leading healthcare organisations demonstrate that robust data infrastructure produces measurable AI outcomes:')
    add_bullet(doc, 'Epic Systems & UCSF: Leveraged an integrated EHR with a FHIR-based data lake to train 50+ clinical AI models, reducing sepsis mortality by 18% through early warning systems (NEJM Catalyst, 2023).')
    add_bullet(doc, 'NHS England GPDPR: The General Practice Data for Planning and Research programme consolidated de-identified GP records for 60 million patients, enabling population-scale AI models.')
    add_bullet(doc, "Mayo Clinic Platform: Provides a federated AI development environment where external developers can train models on Mayo data without data leaving the institution -- a blueprint for privacy-preserving AI development.")
    add_bullet(doc, 'Bumrungrad International (Thailand): Deployed a centralised clinical data warehouse integrating 22 specialist EHR systems, enabling AI-driven patient journey analytics and reducing average length of stay by 12% (2024).')
    add_spacer(doc, 4)
    page_break(doc)

    # S3
    add_h1(doc, '3. AI Infrastructure for Healthcare')
    add_body(doc, 'Deploying AI at clinical scale demands infrastructure that is simultaneously high-performance, secure, resilient, and compliant. Healthcare CIOs must make strategic architectural decisions that balance innovation velocity against the non-negotiable requirements of patient safety, data sovereignty, and regulatory compliance.')

    add_h2(doc, '3.1 Architecture Strategies: Cloud, On-Premise, and Hybrid')
    make_table(doc,
        headers=['Dimension','Cloud-Native','On-Premise','Hybrid'],
        rows=[
            ['Scalability','Elastic, on-demand','Capacity-bound','Flexible burst to cloud'],
            ['Data Sovereignty','Jurisdiction risk','Full control','Sensitive data on-prem'],
            ['Cost Model','OpEx (per-use)','CapEx (upfront)','Mixed'],
            ['Time to Deploy','Days to weeks','Months','Weeks to months'],
            ['Compliance','BAA/DPA required','Native control','Configurable'],
            ['AI Capability','Latest GPU, managed LLMs','Investment-dependent','Optimised per workload'],
            ['Best For','Analytics, NLP, non-PHI AI','Imaging AI, PHI workflows','Most large health systems'],
        ],
        col_widths=[1.8,2.0,2.0,2.0])
    add_caption(doc, 'Table 3.1 -- Infrastructure Architecture Comparison')
    add_spacer(doc, 4)
    add_body(doc, 'For most healthcare systems in Thailand and Southeast Asia, a hybrid architecture is recommended: sensitive clinical workflows and patient data remain on-premise or in a private cloud within national boundaries, while non-PHI analytics, model training on de-identified datasets, and operational AI services leverage public cloud scale.')

    add_h2(doc, '3.2 GPU Computing and Model Serving Infrastructure')
    add_h3(doc, 'Training Infrastructure')
    add_body(doc, 'Foundation model training and large-scale fine-tuning require substantial GPU compute. NVIDIA A100 and H100 clusters with NVLink interconnects are the current standard. Key considerations include:')
    add_bullet(doc, 'Multi-GPU parallelism using frameworks such as DeepSpeed and PyTorch FSDP for training models exceeding single-GPU memory capacity.')
    add_bullet(doc, 'FP8/BF16 mixed-precision training to maximise memory efficiency without accuracy loss.')
    add_bullet(doc, 'Secure research enclaves with isolated GPU clusters for PHI-adjacent model development.')

    add_h3(doc, 'Inference Infrastructure')
    add_body(doc, 'Model serving for clinical decision support demands low-latency, high-availability infrastructure:')
    add_bullet(doc, 'vLLM or NVIDIA TensorRT-LLM for high-throughput LLM inference with continuous batching.')
    add_bullet(doc, 'Triton Inference Server for multi-model ensemble deployment (imaging + NLP + tabular models).')
    add_bullet(doc, 'Kubernetes with auto-scaling to handle peak clinical load (morning rounds, ED surge).')
    add_bullet(doc, 'Model quantisation (4-bit GPTQ, AWQ) for cost-efficient inference at scale.')

    add_h2(doc, '3.3 MLOps for Healthcare')
    add_body(doc, 'Clinical AI models are not static products -- they degrade as patient populations, treatment protocols, and documentation practices evolve. Healthcare MLOps systematically manages the full lifecycle of clinical AI: from data ingestion and model training, through validation and deployment, to continuous monitoring and retraining.')

    add_h3(doc, 'Model Monitoring and Drift Detection')
    add_body(doc, 'Population shift ("concept drift") causes model performance to decay over time. Production monitoring systems must track:')
    add_bullet(doc, 'Input feature distributions -- alert when input data distributions deviate from training baseline.')
    add_bullet(doc, 'Model output distributions -- flag anomalous prediction patterns before they cause clinical harm.')
    add_bullet(doc, 'Clinical outcome alignment -- compare AI predictions against actual patient outcomes using delayed labels.')
    add_bullet(doc, 'Subgroup performance parity -- ensure equitable performance across demographic groups over time.')

    add_h3(doc, 'Retraining and Governance Pipeline')
    add_body(doc, 'A structured retraining governance pipeline should include: automatic drift triggers, validation on held-out local data, clinical review board sign-off, staged rollout (shadow mode, partial, then full deployment), and post-deployment monitoring. This mirrors pharmaceutical good manufacturing practice adapted for AI.')

    add_h2(doc, '3.4 Edge AI for Point-of-Care Applications')
    add_body(doc, 'Edge AI -- running inference on devices proximate to the patient -- enables real-time clinical support without network latency or cloud connectivity requirements:')
    add_bullet(doc, 'Bedside ECG interpretation on dedicated monitoring devices using lightweight CNN models (<100MB) with >98% accuracy for arrhythmia detection.')
    add_bullet(doc, 'Point-of-care ultrasound (POCUS) AI guidance on tablet-sized devices, providing real-time image quality scoring and anatomical landmark detection.')
    add_bullet(doc, 'ICU continuous vital sign analysis on bedside monitors, enabling early warning scores to update every 30 seconds rather than every hour.')
    add_bullet(doc, 'Portable dermatoscopy devices with on-device skin lesion classification, enabling AI-assisted screening in rural and primary care settings.')
    add_bullet(doc, 'Pharmacy dispensing robots performing real-time drug interaction checking and dose verification at the point of dispensing.')

    add_h2(doc, '3.5 Security and Compliance Requirements')
    make_table(doc,
        headers=['Dimension','Requirements','Implementation Controls'],
        rows=[
            ['Data Security','Encryption at rest (AES-256) and in transit (TLS 1.3), key management','HSM-backed KMS, customer-managed keys, immutable audit trails'],
            ['Access Control','Role-based access, MFA, principle of least privilege','IAM policies, PAM systems, quarterly access reviews'],
            ['Network Security','Network segmentation, zero-trust architecture, DDoS protection','Micro-segmentation, service mesh (Istio), WAF'],
            ['Audit & Compliance','Immutable audit logs, regulatory reporting, penetration testing','SIEM integration, annual pen-tests, compliance dashboards'],
            ['AI-Specific','Model card documentation, explainability records, bias audit trails','MLflow tracking, SHAP value logging, fairness dashboards'],
        ],
        col_widths=[1.7,2.7,3.1])
    add_caption(doc, 'Table 3.2 -- Healthcare AI Security and Compliance Requirements')
    add_spacer(doc, 4)
    page_break(doc)

    # S4
    add_h1(doc, '4. AI-Cowork -- Human-AI Collaboration')
    add_body(doc, 'The most transformative shift in healthcare AI thinking is the reframing of AI from an automation tool to a collaborative workforce participant -- a co-worker that augments human clinical judgment rather than replacing it. This concept, which we term "AI-Cowork," positions AI as a tireless, data-rich colleague that handles pattern recognition, information synthesis, and administrative burden, freeing clinicians to do what only humans can do: care, communicate, and decide.')

    add_h2(doc, '4.1 AI as Co-Worker, Not Replacement')
    add_body(doc, 'The "AI will replace doctors" narrative misunderstands both the nature of clinical work and the current capabilities of AI systems. What AI excels at -- processing vast datasets, detecting subtle patterns, never forgetting a guideline -- is precisely what clinicians find most exhausting. What clinicians excel at -- building therapeutic relationships, navigating uncertainty, applying contextual judgment -- remains well beyond AI. The AI-Cowork model designs collaboration around these complementary strengths.')
    add_body(doc, 'A 2023 AMA survey found that 64% of physicians who had used AI tools reported improved job satisfaction when AI reduced documentation burden, not when it attempted autonomous diagnosis. The framing matters: AI as administrative relief and diagnostic consultation -- not replacement.')

    add_h2(doc, '4.2 Clinical Decision Support Systems (CDSS)')
    add_h3(doc, 'Alert-Based CDSS')
    add_body(doc, 'The traditional drug-drug interaction alert, sepsis early warning score, and critical lab value notification. Alert fatigue remains a significant challenge: studies show clinicians override 90%+ of all EHR alerts (Rehr et al., JAMIA, 2022). Next-generation CDSS apply ML to contextualise alerts, surfacing only those most likely to be actionable for the specific patient and clinical setting.')

    add_h3(doc, 'Recommendation-Based CDSS')
    add_body(doc, 'AI systems that proactively suggest diagnostic workups, treatment pathways, or care plan modifications based on real-time patient data and clinical guidelines. Examples include antibiotic stewardship AI recommending de-escalation based on culture results, and sepsis bundle compliance advisors.')

    add_h3(doc, 'Predictive CDSS')
    add_body(doc, "Machine learning models predicting adverse events before they manifest: 72-hour readmission risk, ICU deterioration within 6 hours, surgical complication probability. Vanderbilt University Medical Center's ASPECT-ICU model reduced unexpected ICU deaths by 22% through 12-hour advance prediction (Chen et al., Lancet Digital Health, 2024).")

    add_h2(doc, '4.3 AI-Assisted Diagnosis')
    add_h3(doc, 'Radiology')
    add_body(doc, 'AI reading assistants for chest X-ray, CT, and MRI are now deployed at scale globally:')
    add_bullet(doc, 'Chest X-ray: pneumothorax detection (AUC 0.97), lung nodule detection for cancer screening (Ardila et al., Nature Medicine, 2019).')
    add_bullet(doc, 'Diabetic retinopathy screening from fundus photographs: FDA-cleared IDx-DR achieves 87.2% sensitivity, 90.7% specificity -- enabling autonomous screening without ophthalmologist review.')
    add_bullet(doc, 'CT stroke detection: AI triage (e.g., Viz.ai) reduces door-to-treatment time for large vessel occlusion by an average of 52 minutes (Morey et al., Stroke, 2021).')

    add_h3(doc, 'Pathology')
    add_body(doc, "Digital pathology AI analyses whole-slide images for cancer grading, biomarker quantification, and rare disease detection. Google's CHIEF model (2024) demonstrated generalisation across 19 cancer types from H&E stained slides alone, suggesting a potential universal pathology foundation model.")

    add_h3(doc, 'Dermatology')
    add_body(doc, 'Skin lesion classification AI has reached dermatologist-level performance for melanoma detection in controlled studies (Esteva et al., Nature, 2017). Clinical deployment at scale requires workflow integration and bias mitigation for darker skin tones -- an active area of current research and regulatory attention.')

    add_h2(doc, '4.4 AI in Hospital Operations')
    make_table(doc,
        headers=['Application','AI Approach','Reported Impact'],
        rows=[
            ['Patient scheduling optimisation','Reinforcement learning, constraint optimisation','15-25% reduction in appointment no-show rates'],
            ['OR block scheduling','Predictive ML + optimisation','12% improvement in OR utilisation (Mayo Clinic, 2023)'],
            ['ED patient flow prediction','Time-series forecasting (LSTM)','20% reduction in patient wait times'],
            ['Supply chain & inventory','Demand forecasting, anomaly detection','18% reduction in medical supply waste'],
            ['Staff rostering','ML-based demand forecasting + optimisation','30% reduction in rostering time, improved coverage'],
            ['Readmission prediction','Gradient boosting, neural networks','Up to 20% reduction in 30-day readmissions'],
            ['Revenue cycle management','NLP for documentation, claims prediction','15-40% reduction in claim denials'],
        ],
        col_widths=[2.0,2.1,3.4])
    add_caption(doc, 'Table 4.1 -- AI in Hospital Operations: Applications and Impact')
    add_spacer(doc, 4)

    add_h2(doc, '4.5 AI Agents and Multi-Agent Systems in Healthcare')
    add_h3(doc, 'Clinical Workflow Agents')
    add_body(doc, 'An AI agent embedded in the EHR workflow can autonomously retrieve relevant prior records, identify applicable clinical guidelines, check current formulary and dosing protocols, draft a clinical summary, and present structured recommendations for clinician review -- all within the time the physician is greeting the patient. This compresses 20 minutes of administrative preparation into seconds.')

    add_h3(doc, 'Multi-Agent Healthcare Architectures')
    add_body(doc, 'Complex clinical processes benefit from specialist agents working in parallel:')
    add_bullet(doc, 'Diagnostic reasoning agent: synthesises history, exam findings, and test results to generate a ranked differential diagnosis.')
    add_bullet(doc, 'Guideline compliance agent: cross-references the proposed plan against current NICE, ACC/AHA, or Thai Medical Council guidelines.')
    add_bullet(doc, 'Drug safety agent: evaluates pharmacological interactions, contraindications, and renal/hepatic dosing adjustments.')
    add_bullet(doc, 'Documentation agent: generates structured clinical notes, discharge summaries, and referral letters from a brief physician dictation.')
    add_bullet(doc, 'Care coordination agent: identifies follow-up gaps, initiates care transition communications, and schedules post-discharge calls.')

    add_h2(doc, '4.6 LLMs and RAG Systems in Clinical Knowledge Management')
    add_body(doc, 'Large Language Models (LLMs), particularly when augmented with Retrieval-Augmented Generation (RAG), represent a paradigm shift in clinical knowledge management. A RAG-powered clinical assistant continuously retrieves and synthesises the most current, institution-specific knowledge in response to natural language queries, replacing hours of manual guideline searching.')

    add_h3(doc, 'Clinical RAG Architecture')
    add_body(doc, 'A production-grade clinical RAG system for a Thai tertiary hospital combines:')
    add_bullet(doc, 'Knowledge base: Thai Medical Council guidelines, hospital formulary, NICE guidelines, PubMed literature (indexed with vector embeddings).')
    add_bullet(doc, 'Retrieval: Dense vector search (FAISS or ChromaDB) for semantic retrieval, augmented by BM25 sparse retrieval for keyword precision.')
    add_bullet(doc, 'Generation: A fine-tuned LLM (e.g., Llama-3-based Thai medical model) that synthesises retrieved context into structured clinical responses.')
    add_bullet(doc, 'Guardrails: Fact-checking module, hallucination detection, mandatory source citation, and human-review flags for high-stakes queries.')

    add_h3(doc, 'Demonstrated Applications')
    add_bullet(doc, "NYU Langone's clinical LLM assistant reduced time to answer complex protocol questions from 23 minutes (manual search) to 45 seconds (2024).")
    add_bullet(doc, "UCSF's EHR-integrated RAG system achieved 89% clinical accuracy on drug dosing queries compared to pharmacist review.")
    add_bullet(doc, 'Siriraj Hospital (Thailand) piloted a Thai-language clinical guideline chatbot for nursing staff, reporting 85% satisfaction and 30% reduction in protocol-related calls to pharmacy (2024).')

    add_h2(doc, '4.7 Trust, Explainability, and Human-in-the-Loop')
    add_h3(doc, 'Explainability by Design')
    add_body(doc, 'Clinical adoption of AI depends fundamentally on trust -- and trust requires explainability. Clinicians must understand why an AI system produces a given recommendation before they can responsibly act on it. AI systems in high-stakes clinical decisions should incorporate interpretability methods: SHAP for feature importance in tabular models, GradCAM for imaging AI attention visualisation, and chain-of-thought reasoning traces for LLM-based clinical recommendations. Explainability should be surfaced in a clinician-accessible format -- plain-language reasoning summaries, not raw statistical values.')

    add_h3(doc, 'Human-in-the-Loop (HITL) Architecture')
    add_body(doc, 'All clinical AI systems should be designed with the human-in-the-loop principle: AI provides decision support, humans retain final authority. HITL architecture specifies:')
    add_bullet(doc, 'Which AI recommendations require mandatory clinician review before action (high-stakes decisions).')
    add_bullet(doc, 'Which recommendations can be acted on without individual case review (low-stakes, high-confidence decisions).')
    add_bullet(doc, 'Clear documentation of AI involvement in the clinical record.')
    add_bullet(doc, 'Patient disclosure requirements where AI substantively influenced care decisions.')

    add_h3(doc, 'Regulatory Alignment')
    add_body(doc, "Thailand's FDA has established an AI/ML medical device regulatory pathway aligned with the U.S. FDA's Predetermined Change Control Plan (PCCP) framework. Healthcare AI developers must prepare technical files demonstrating clinical validation, real-world performance monitoring plans, and post-market surveillance commitments.")
    page_break(doc)

    # S5
    add_h1(doc, '5. Implementation Roadmap')
    add_body(doc, 'A structured approach to AI adoption reduces risk, builds organisational capability sustainably, and ensures clinical and financial returns. This section presents a five-stage maturity model, change management principles, an ROI framework, and the most common failure modes to avoid.')

    add_h2(doc, '5.1 AI Adoption Maturity Model')
    add_body(doc, 'Healthcare AI adoption follows a predictable progression through five maturity stages. Understanding your current position enables strategic planning of the investments and capabilities required for the next stage:')
    make_table(doc,
        headers=['Stage','Name','Characteristics','Focus Area'],
        rows=[
            ['Stage 1','Foundation','Basic EHR adoption, fragmented data, no AI deployment, limited analytics','Data infrastructure: EHR standardisation, data governance, connectivity'],
            ['Stage 2','Data Ready','Integrated data warehouse, clean master data, BI dashboards, descriptive analytics','Analytics capability: BI tools, data science team, quality metrics'],
            ['Stage 3','AI Pilot','First AI use cases deployed (1-3), proofs of concept validated, clinical champions identified','AI deployment: model development, validation studies, staff training'],
            ['Stage 4','AI at Scale','Multiple AI applications in production, MLOps in place, ROI demonstrated, governance mature','Scale & govern: MLOps platform, AI governance committee, performance monitoring'],
            ['Stage 5','AI-Native','AI embedded in all major workflows, AI agents operational, continuous learning, innovation culture','Innovation: foundation models, agentic AI, federated learning, digital twins'],
        ],
        col_widths=[0.8,1.2,2.7,2.8])
    add_caption(doc, 'Table 5.1 -- Healthcare AI Adoption Maturity Model')
    add_spacer(doc, 4)
    add_body(doc, 'Most Thai tertiary hospitals currently operate at Stage 2-3. The strategic priority for 2026-2028 should be a disciplined Stage 3 to Stage 4 transition: moving from isolated pilots to scalable, governed AI deployment programmes.')

    add_h2(doc, '5.2 Change Management and Workforce Readiness')
    add_h3(doc, 'Clinical Champion Identification')
    add_body(doc, 'Identify and support clinical champions -- respected clinicians who combine clinical credibility with enthusiasm for AI. Champions provide bottom-up adoption momentum that top-down mandates cannot achieve. Support them with dedicated time, co-design authority, and public recognition.')

    add_h3(doc, 'AI Literacy Programme')
    add_body(doc, 'Develop a tiered AI literacy programme across your workforce:')
    add_bullet(doc, 'Tier 1 -- All clinical staff: AI fundamentals, how to critically evaluate AI recommendations, patient communication about AI.')
    add_bullet(doc, 'Tier 2 -- Department leads and nurses: Workflow integration, performance monitoring, escalation protocols, documentation requirements.')
    add_bullet(doc, 'Tier 3 -- AI champions and data leads: Model validation methodology, bias assessment, MLOps concepts, regulatory requirements.')
    add_bullet(doc, 'Tier 4 -- IT and data science teams: Technical implementation, security, MLOps toolchains, model development.')

    add_h3(doc, 'Governance Structures')
    add_body(doc, 'Establish a Healthcare AI Governance Committee with cross-functional representation (CMO, CIO, CNO, legal, ethics, patient representative). This committee is responsible for AI use case prioritisation, clinical validation standards, deployment approval, and ongoing performance oversight.')

    add_h2(doc, '5.3 ROI Framework for Healthcare AI')
    make_table(doc,
        headers=['Value Category','Metric Examples','Typical Range'],
        rows=[
            ['Clinical Outcomes','Mortality reduction, complication rates, readmissions, diagnosis accuracy','5-25% improvement'],
            ['Operational Efficiency','Length of stay, throughput, OR utilisation, staff time saved','10-30% improvement'],
            ['Financial Returns','Revenue cycle efficiency, reduced penalties, prevented readmissions','USD 2-8 per USD 1 invested'],
            ['Workforce Quality','Burnout reduction, documentation time, staff satisfaction, retention','15-40% time saved on admin'],
        ],
        col_widths=[2.0,3.2,2.3])
    add_caption(doc, 'Table 5.2 -- Healthcare AI ROI Framework')
    add_spacer(doc, 4)
    add_body(doc, 'A rigorous ROI analysis requires establishing clinical and operational baselines before deployment, defining primary outcome metrics and a measurement timeline, assigning attribution methodology (e.g., propensity-matched controls or interrupted time series), and accounting for total cost of ownership including infrastructure, maintenance, and training.')

    add_h2(doc, '5.4 Common Pitfalls and How to Avoid Them')
    make_table(doc,
        headers=['Pitfall','Why It Happens','Prevention Strategy'],
        rows=[
            ['Starting with technology, not problems','Vendor enthusiasm, innovation pressure','Define clinical problem first; let AI be the solution if warranted'],
            ['Insufficient data quality investment','Urgency to deploy; underestimating data readiness','Audit and remediate data quality as a prerequisite gate'],
            ['Skipping local clinical validation','Overreliance on published benchmark accuracy','Validate on local patient population before deployment'],
            ['Alert fatigue from over-deployment','Multiple AI systems generating low-specificity alerts','Start with high-specificity, high-impact alerts; suppress low-value alerts'],
            ['No monitoring post-deployment','AI treated as a one-time product, not a service','Budget for ongoing MLOps, performance reviews, and retraining'],
            ['Clinician exclusion from design','AI built by IT for clinicians, not with clinicians','Co-design requirement: clinical champion on every AI project'],
            ['Underestimating change management','Focus on technology; culture treated as secondary','Allocate 40% of implementation budget to training and change management'],
        ],
        col_widths=[2.0,2.3,3.2])
    add_caption(doc, 'Table 5.3 -- Common Healthcare AI Pitfalls and Prevention Strategies')
    add_spacer(doc, 4)
    page_break(doc)

    # S6
    add_h1(doc, '6. Future Outlook')
    add_body(doc, 'The pace of AI advancement is accelerating. Technologies that are experimental today will be clinical standards within five years. Healthcare leaders must develop the foresight to invest ahead of the adoption curve while maintaining the governance rigour that patient safety demands.')

    add_h2(doc, '6.1 Emerging Trends')
    add_h3(doc, 'Foundation Models for Medicine')
    add_body(doc, 'General-purpose foundation models trained on multimodal healthcare data are emerging as platforms for a new generation of clinical AI:')
    add_bullet(doc, "Google's Med-PaLM 2 (2023) demonstrated expert-level clinical reasoning on the MedQA benchmark. The successor Med-Gemini (2024) extends multimodal capability to radiology images, EHR data, and genomics simultaneously.")
    add_bullet(doc, "Microsoft and Epic's DAX Copilot uses GPT-4 to generate clinical notes from ambient voice recording in real-time, reducing documentation time by an average of 7 minutes per encounter in early deployments.")
    add_bullet(doc, "Meta's ESMFold uses a language-model architecture to predict protein structures, accelerating drug discovery timelines from years to days for specific target classes.")
    add_bullet(doc, "NVIDIA's BioNeMo platform provides a unified framework for healthcare-specific foundation model development and deployment, covering genomics, drug discovery, and imaging.")

    add_h3(doc, 'Federated Learning at Scale')
    add_body(doc, "Federated learning enables AI models to learn from distributed patient data across multiple institutions without the data ever leaving individual sites -- resolving the fundamental tension between AI's need for scale and healthcare's requirement for data privacy:")
    add_bullet(doc, 'Intel OpenFL and NVIDIA FLARE are production-ready federated learning frameworks with healthcare-specific security features.')
    add_bullet(doc, 'The FeTS (Federated Tumor Segmentation) initiative demonstrated federated learning across 71 international sites, training brain tumour segmentation models that outperformed models trained at any single site (Pati et al., Nature Communications, 2022).')
    add_bullet(doc, "Thailand's National Health Security Office (NHSO) has identified federated learning as a strategic priority for the national AI health data strategy (2025-2030).")

    add_h3(doc, 'Healthcare Digital Twins')
    add_body(doc, 'A healthcare digital twin is a dynamic, AI-driven simulation of a patient, a clinical pathway, or an entire hospital system, updated continuously from real-world data and used to test interventions before they are applied:')
    add_bullet(doc, 'Patient-level twins: personalised physiological models (e.g., cardiac digital twins from ECG and imaging data) that predict how a specific patient will respond to a specific treatment. Demonstrated in ICU dosing optimisation and pre-surgical planning.')
    add_bullet(doc, 'Hospital-level twins: simulation models of patient flow, resource utilisation, and capacity that allow managers to test scheduling changes, capacity expansions, or pandemic surge protocols in silico.')
    add_bullet(doc, 'Siemens Healthineers and Philips are both investing heavily in digital twin platforms as next-generation service offerings for health systems.')

    add_h3(doc, 'Ambient Clinical Intelligence')
    add_body(doc, 'Ambient AI systems that passively observe and interpret clinical encounters -- conversations, physical examinations, bedside monitoring data -- to automatically generate documentation, flag concerns, and update the patient record without requiring clinician-computer interaction. Nuance (Microsoft) DAX, DeepScribe, and Abridge are leading examples, with clinical trials demonstrating 60-70% reduction in after-hours documentation ("pajama time").')

    add_h2(doc, "6.2 Thailand's Position and Opportunities in Healthcare AI")
    add_h3(doc, 'Structural Advantages')
    add_bullet(doc, "Medical Tourism Hub: Thailand's position as the region's premier medical tourism destination means it manages a uniquely diverse patient population -- creating rich datasets for AI training rare in monocultural health systems.")
    add_bullet(doc, "Universal Health Coverage: The Universal Coverage Scheme (UCS) generates longitudinal health data for 50+ million citizens -- a national asset of extraordinary AI value if properly governed.")
    add_bullet(doc, "Technical Talent: Thailand has strong AI and computer science programmes at Chulalongkorn, Mahidol, KMITL, and NECTEC, with a growing pipeline of healthcare AI researchers.")
    add_bullet(doc, "ASEAN Leadership: As a founding ASEAN member with the region's most sophisticated private hospital sector, Thailand is positioned to set regional standards for healthcare AI governance and interoperability.")

    add_h3(doc, 'Strategic Priorities (2026-2030)')
    add_bullet(doc, 'National Health AI Data Strategy: Establish a federated national health data commons under NHSO/MOPH governance, enabling privacy-preserving AI research at population scale.')
    add_bullet(doc, 'Thai Medical Foundation Models: Invest in pre-training Thai-language biomedical LLMs on the Thai medical literature, clinical guidelines, and de-identified national health records.')
    add_bullet(doc, "AI Regulatory Clarity: Thailand's FDA should accelerate AI/ML medical device guidance, providing clear pathways for domestic innovators while aligning with international standards.")
    add_bullet(doc, 'Regional AI Health Hub: Thailand should aspire to host the ASEAN Centre for Healthcare AI Excellence -- a shared facility for clinical AI validation, training, and regulatory harmonisation.')

    add_h2(doc, '6.3 Call to Action for Healthcare Leaders')
    add_body(doc, 'The decisions made by healthcare leaders in 2026 will determine whether their organisations lead or follow the AI transformation of care. The following commitments, made now, will compound in value over the decade ahead:')

    add_h3(doc, 'Immediate Actions (0-6 Months)')
    add_bullet(doc, 'Commission a Data Readiness Assessment: Evaluate your current data infrastructure against Stage 2 maturity criteria. Identify the top three data quality initiatives that will unlock AI capability.')
    add_bullet(doc, 'Establish an AI Governance Committee: Cross-functional, with clinical, IT, legal, ethics, and patient representation. Charter it with authority over AI use case approval, validation standards, and performance oversight.')
    add_bullet(doc, 'Identify Two High-Value AI Pilots: Choose areas with clear clinical need, good baseline data, and an engaged clinical champion. Radiology AI and operational scheduling optimisation are proven starting points.')
    add_bullet(doc, 'Begin AI Literacy Programme: Start with the leadership team -- board members and executive committee.')

    add_h3(doc, 'Short-Term Priorities (6-24 Months)')
    add_bullet(doc, 'Deploy and rigorously validate the two pilots. Measure outcomes against pre-defined clinical and financial metrics. Document learnings.')
    add_bullet(doc, 'Build the MLOps foundation: monitoring dashboards, model performance reviews, retraining governance.')
    add_bullet(doc, 'Implement a clinical RAG system for knowledge management -- start with formulary and protocol queries.')
    add_bullet(doc, 'Engage with national health AI initiatives (NHSO, MOPH) and ASEAN networks.')

    add_h3(doc, 'Long-Term Vision (3-5 Years)')
    add_bullet(doc, 'Achieve Stage 4 maturity: AI embedded across radiology, operations, clinical documentation, and patient monitoring.')
    add_bullet(doc, 'Pilot AI agent workflows in at least one clinical department.')
    add_bullet(doc, 'Contribute de-identified data and learnings to national and regional federated AI research networks.')
    add_bullet(doc, 'Establish your institution as a reference site for responsible healthcare AI in Southeast Asia.')

    add_spacer(doc, 10)
    p_q = doc.add_paragraph()
    p_q.paragraph_format.alignment   = WD_ALIGN_PARAGRAPH.CENTER
    p_q.paragraph_format.space_before = Pt(12)
    p_q.paragraph_format.left_indent  = Inches(0.5)
    p_q.paragraph_format.right_indent = Inches(0.5)
    rq = p_q.add_run(
        '"The question is not whether AI will transform healthcare -- it already is.\n'
        'The question is whether your institution will shape that transformation, or merely respond to it."'
    )
    rq.font.size = Pt(13); rq.font.italic = True
    rq.font.color.rgb = NAVY; rq.font.name = 'Arial'
    page_break(doc)

    # REFERENCES
    add_h1(doc, 'References')
    add_body(doc, 'The following sources were drawn upon in the preparation of this report. All URLs were active as of March 2026.')
    add_spacer(doc, 4)

    refs = [
        'Ardila, D., Kiraly, A.P., Bharadwaj, S., et al. (2019). End-to-end lung cancer detection on CT scans using deep learning. Nature Medicine, 25, 954-961.',
        'Chen, J.H. et al. (2024). Machine learning for real-time ICU deterioration prediction: ASPECT-ICU. The Lancet Digital Health, 6(3), e180-e190.',
        'Esteva, A., Kuprel, B., Novoa, R.A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542, 115-118.',
        'FDA. (2024). Artificial Intelligence and Machine Learning (AI/ML)-Enabled Medical Devices. U.S. Food and Drug Administration.',
        'Grand View Research. (2024). Artificial Intelligence in Healthcare Market Size, Share & Trends Analysis Report.',
        'Medscape. (2022). Physician Burnout & Depression Report 2022: Stress, Anxiety, and Anger.',
        'Morey, J.R., Fiano, E., Yaeger, K.A., et al. (2021). Impact of Viz LVO on time-to-treatment and clinical outcomes in large vessel occlusion stroke patients. Stroke, 52(2), 604-611.',
        'NESDC Thailand. (2023). Population Projections for Thailand 2010-2040. National Economic and Social Development Council.',
        'Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. Science, 366(6464), 447-453.',
        'Pati, S., Baid, U., Edwards, B., et al. (2022). Federated learning enables big data for rare cancer boundary detection. Nature Communications, 13, 7346.',
        'RBC Capital Markets. (2020). Healthcare Data Explosion: The Real Digital Opportunity.',
        'Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge. Nature, 620, 172-180.',
        'Sterne, J.A.C., White, I.R., Carlin, J.B., et al. (2009). Multiple imputation for missing data in epidemiological and clinical research. BMJ, 338, b2393.',
        'WHO. (2021). Ageing and Health. World Health Organization.',
        'WHO. (2021). Health and Care Worker Shortages. World Health Organization.',
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph(style='Normal')
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after  = Pt(4)
        r = p.add_run('[%d] %s' % (i, ref))
        r.font.size = Pt(10); r.font.color.rgb = BODY; r.font.name = 'Arial'

    # Footer
    for section in doc.sections:
        footer = section.footer
        for fp2 in list(footer.paragraphs):
            fp2._element.getparent().remove(fp2._element)
        fp = footer.add_paragraph()
        fp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

        run_pre = fp.add_run('Page ')
        run_pre.font.size = Pt(9); run_pre.font.color.rgb = GRAY; run_pre.font.name = 'Arial'

        fldChar1 = OxmlElement('w:fldChar'); fldChar1.set(qn('w:fldCharType'), 'begin')
        instrText1 = OxmlElement('w:instrText'); instrText1.text = 'PAGE'
        fldChar2 = OxmlElement('w:fldChar'); fldChar2.set(qn('w:fldCharType'), 'end')
        rp1 = OxmlElement('w:r'); rp1.append(fldChar1)
        rp2 = OxmlElement('w:r'); rp2.append(instrText1)
        rp3 = OxmlElement('w:r'); rp3.append(fldChar2)
        fp._element.append(rp1); fp._element.append(rp2); fp._element.append(rp3)

        run_of = fp.add_run(' of ')
        run_of.font.size = Pt(9); run_of.font.color.rgb = GRAY; run_of.font.name = 'Arial'

        fldChar3 = OxmlElement('w:fldChar'); fldChar3.set(qn('w:fldCharType'), 'begin')
        instrText2 = OxmlElement('w:instrText'); instrText2.text = 'NUMPAGES'
        fldChar4 = OxmlElement('w:fldChar'); fldChar4.set(qn('w:fldCharType'), 'end')
        rp4 = OxmlElement('w:r'); rp4.append(fldChar3)
        rp5 = OxmlElement('w:r'); rp5.append(instrText2)
        rp6 = OxmlElement('w:r'); rp6.append(fldChar4)
        fp._element.append(rp4); fp._element.append(rp5); fp._element.append(rp6)

        run_post = fp.add_run(' | Leading with AI in Healthcare -- Anirach Mingkhwan, March 2026')
        run_post.font.size = Pt(9); run_post.font.color.rgb = GRAY; run_post.font.name = 'Arial'

    doc.save(OUTPUT_PATH)
    print('Saved:', OUTPUT_PATH)

if __name__ == '__main__':
    build()
