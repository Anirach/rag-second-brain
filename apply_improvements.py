#!/usr/bin/env python3
"""Apply all 5 improvements to MGNA paper v2 -> v3."""

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import copy

doc = Document('Multi_Graph_LLM_Second_Brain_v2.docx')

# Helper functions
def add_mono_block(paragraphs_list, insert_idx, text_lines):
    """Insert monospace block paragraphs. Returns number of paragraphs added."""
    added = 0
    for line in text_lines:
        p = doc.add_paragraph()
        run = p.add_run(line if line else ' ')
        run.font.name = 'Courier New'
        run.font.size = Pt(8)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = Pt(10)
        # Move paragraph to correct position
        paragraphs_list[insert_idx]._element.addprevious(p._element)
        added += 1
    return added

def add_normal_para(anchor_element, text, bold_prefix=None):
    """Add a normal paragraph before anchor_element."""
    from docx.oxml.ns import qn
    from lxml import etree
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    anchor_element.addprevious(p._element)
    return p

def add_heading_para(anchor_element, text, level=2):
    """Add a heading paragraph before anchor_element."""
    p = doc.add_paragraph()
    p.style = doc.styles['Normal']
    run = p.add_run(text)
    run.bold = True
    if level == 2:
        run.font.size = Pt(12)
    elif level == 3:
        run.font.size = Pt(11)
    anchor_element.addprevious(p._element)
    return p

def add_mono_before(anchor_element, lines):
    """Add monospace lines before anchor_element."""
    for line in lines:
        p = doc.add_paragraph()
        run = p.add_run(line if line else ' ')
        run.font.name = 'Courier New'
        run.font.size = Pt(8)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = Pt(10)
        anchor_element.addprevious(p._element)

# Find paragraph indices by text prefix
def find_para(prefix):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith(prefix):
            return i
    return None

# ============================================================
# IMPROVEMENT 1: Add Block Schemes after Section 2.1
# ============================================================
# Insert after paragraph 29 (GraphRAG paragraph), before paragraph 30 (2.2)
idx_22 = find_para('2.2 Knowledge Graph Integration')
anchor = doc.paragraphs[idx_22]._element

add_heading_para(anchor, '2.1.1 Architectural Comparison: GraphRAG vs. MGNA', level=3)

add_normal_para(anchor, 'To clarify the fundamental differences between GraphRAG and the proposed MGNA framework, we present architectural block schemes for both systems followed by a detailed comparison.')

add_normal_para(anchor, 'Figure A: Standard GraphRAG Pipeline', bold_prefix='')
graphrag_lines = [
    '┌─────────────┐    ┌──────────────┐    ┌─────────────────┐',
    '│    Raw       │───▶│    Text      │───▶│    Entity       │',
    '│  Documents   │    │  Chunking    │    │  Extraction     │',
    '└─────────────┘    └──────────────┘    └────────┬────────┘',
    '                                                │',
    '                                                ▼',
    '                                     ┌─────────────────┐',
    '                                     │   Knowledge     │',
    '                                     │   Graph (KG)    │',
    '                                     └────────┬────────┘',
    '                                              │',
    '                                              ▼',
    '                                     ┌─────────────────┐',
    '                                     │   Community     │',
    '                                     │   Detection     │',
    '                                     │  (Leiden alg.)  │',
    '                                     └────────┬────────┘',
    '                                              │',
    '                              ┌───────────────┼───────────────┐',
    '                              ▼               │               ▼',
    '                    ┌──────────────┐          │     ┌──────────────┐',
    '                    │ Local Search │          │     │ Global Search│',
    '                    │ (entity      │          │     │ (community   │',
    '                    │  neighbors)  │          │     │  summaries)  │',
    '                    └──────┬───────┘          │     └──────┬───────┘',
    '                           └──────────────────┼────────────┘',
    '                                              ▼',
    '                                     ┌─────────────────┐',
    '                                     │   LLM Answer    │',
    '                                     └─────────────────┘',
]
add_mono_before(anchor, graphrag_lines)

add_normal_para(anchor, 'Figure B: MGNA Pipeline (Proposed)')
mgna_lines = [
    '┌─────────────┐',
    '│    Raw       │',
    '│  Documents   │',
    '└──────┬──────┘',
    '       │',
    '       ├──────────────────┬──────────────────┐',
    '       ▼                  ▼                  ▼',
    '┌─────────────┐   ┌─────────────┐   ┌─────────────┐',
    '│Co-occurrence│   │  Sequence   │   │  Knowledge  │',
    '│ Graph Gco   │   │ Graph Gseq  │   │  Graph Gkg  │',
    '│(PMI weights)│   │(dependency  │   │(entity-rel  │',
    '│             │   │ + position) │   │  triples)   │',
    '└──────┬──────┘   └──────┬──────┘   └──────┬──────┘',
    '       │                 │                  │',
    '       └────────┬────────┴────────┬─────────┘',
    '                ▼                 │',
    '       ┌─────────────────┐       │',
    '       │  Unified Multi- │       │',
    '       │  Graph M        │       │',
    '       └────────┬────────┘       │',
    '                ▼                │',
    '       ┌─────────────────┐       │',
    '       │  Multi-Edge     │       │',
    '       │  Message Passing│       │',
    '       │  (Def. 5)       │       │',
    '       └────────┬────────┘       │',
    '                ▼                │',
    '       ┌─────────────────┐       │',
    '       │  Cross-Graph    │       │',
    '       │  Attention      │       │',
    '       │  (Def. 6)       │       │',
    '       └───┬─────────┬───┘       │',
    '           │         │           │',
    '           ▼         ▼           │',
    '    ┌──────────┐ ┌──────────┐    │',
    '    │   Soft   │ │   Hard   │    │',
    '    │Prompting │ │Retrieval │    │',
    '    │ (P_g)    │ │ (C_k)    │    │',
    '    └────┬─────┘ └────┬─────┘    │',
    '         └──────┬─────┘          │',
    '                ▼                │',
    '       ┌─────────────────┐       │',
    '       │  LLM f_theta    │       │',
    '       │  Input: P_g +   │       │',
    '       │  C_k + x_q      │       │',
    '       └────────┬────────┘       │',
    '                ▼',
    '       ┌─────────────────┐',
    '       │   Output y      │',
    '       └─────────────────┘',
]
add_mono_before(anchor, mgna_lines)

# Written explanation paragraphs
add_normal_para(anchor, 
    'The most fundamental architectural difference between GraphRAG and MGNA lies in the number and diversity of graph types employed. GraphRAG constructs a single knowledge graph from extracted entities and relationships, then applies hierarchical community detection (typically the Leiden algorithm) to partition this graph into communities at multiple resolutions. All retrieval—whether local (entity-neighborhood) or global (community-summary)—operates over this single graph structure. In contrast, MGNA constructs three complementary graph types in parallel: a co-occurrence graph capturing statistical word associations via pointwise mutual information, a sequence graph encoding syntactic dependencies and positional relationships, and a knowledge graph representing explicit factual triples. This multi-graph design ensures that MGNA captures statistical, structural, and semantic dimensions of the input corpus simultaneously.')

add_normal_para(anchor,
    'A second key difference concerns the mechanism for information aggregation. GraphRAG relies on hierarchical community detection followed by query-time graph traversal: at inference, the system either traverses local neighborhoods around query-relevant entities or summarizes global community structures. This process is fundamentally a retrieval operation that does not involve learned parameters for cross-structure reasoning. MGNA, by contrast, employs heterogeneous message passing (Definition 5) with learnable edge-type attention weights, followed by cross-graph attention (Definition 6) that enables information to flow between the three graph types. This means MGNA learns how to combine co-occurrence statistics with sequential structure and factual knowledge through end-to-end training, rather than relying on heuristic traversal strategies.')

add_normal_para(anchor,
    'Third, the integration mechanism with the LLM differs substantially. GraphRAG follows a retrieval-only paradigm: retrieved context (entity descriptions, community summaries) is concatenated with the query and passed as text input to the LLM. MGNA implements a dual integration strategy combining soft prompting—where graph-derived embeddings are projected into the LLM\'s continuous input space as learnable prefix tokens—with hard retrieval of top-k relevant subgraph nodes. This dual pathway allows MGNA to influence the LLM\'s generation through both continuous latent representations (soft prompting) and discrete factual context (hard retrieval), providing richer and more nuanced augmentation than text-only retrieval.')

add_normal_para(anchor,
    'Finally, these architectural choices have implications for scalability and adaptability. GraphRAG\'s community-based approach provides efficient global summarization but cannot dynamically reweight the importance of different information types per query. MGNA\'s learned attention weights (both edge-type weights α_τ and cross-graph weights β_g) enable query-adaptive information routing, where factual queries can emphasize the knowledge graph while statistical or stylistic queries draw more heavily on co-occurrence and sequence graphs. This adaptability is reflected in the experimental results (Section 6), where MGNA demonstrates consistent improvements across diverse task types.')


# ============================================================
# IMPROVEMENT 2: Add Section 2.5 Research Gap Analysis
# ============================================================
idx_3 = find_para('3. MATHEMATICAL FRAMEWORK')
anchor3 = doc.paragraphs[idx_3]._element

add_heading_para(anchor3, '2.5 Research Gap Analysis and Derived Research Questions', level=2)

add_normal_para(anchor3,
    'Gap from Section 2.1 (RAG/GraphRAG): Current retrieval-augmented generation systems, including GraphRAG, rely on a single graph type—typically a community-structured knowledge graph—for organizing and retrieving external knowledge. This design omits two critical information dimensions: statistical co-occurrence patterns that capture implicit topical associations between terms, and sequential/syntactic structures that encode how concepts relate within discourse. The absence of these complementary views limits the system\'s ability to reason about implicit associations and discourse-level coherence, particularly for queries requiring integration of statistical, structural, and factual evidence.')

add_normal_para(anchor3,
    'Gap from Section 2.2 (KG Integration): Knowledge graph integration approaches such as QA-GNN, GreaseLM, and recent LLM-based KG construction methods treat knowledge graphs as static, pre-constructed resources. These methods do not incorporate dynamically computed co-occurrence statistics or learn unified representations across heterogeneous graph types. The lack of a multi-graph learning framework means that complementary information sources cannot be jointly optimized, limiting the expressiveness of the resulting representations.')

add_normal_para(anchor3,
    'Gap from Section 2.3 (GNN for Text): Graph neural network approaches for text classification, from TextGCN to TextGSL, have demonstrated the value of graph-based representations but typically operate on a single graph type (e.g., co-occurrence or dependency graphs) without mechanisms for cross-graph interaction. Furthermore, these approaches are designed for classification tasks and lack integration pathways for augmenting large language model generation, leaving a gap between graph-based text understanding and LLM-based text generation.')

add_normal_para(anchor3,
    'Gap from Section 2.4 (Memory-Augmented Architectures): Memory-augmented LLM architectures such as MemGPT, A-MEM, and G-Memory provide external storage mechanisms but typically use unstructured or flat memory representations without graph-based organization. Even graph-memory approaches like G-Memory employ single-graph hierarchical structures rather than multi-graph representations, and none combine graph-based retrieval with continuous soft prompting for LLM integration.')

add_normal_para(anchor3,
    'Based on these identified gaps, we derive four research questions that guide the design and evaluation of MGNA:')

add_normal_para(anchor3,
    'RQ1: How can co-occurrence, sequence, and knowledge graphs be unified within a single learnable framework that preserves the distinct information captured by each graph type while enabling joint optimization?',
    bold_prefix='')
add_normal_para(anchor3,
    'RQ2: What cross-graph attention mechanism enables effective information flow across heterogeneous graph types with different node sets, edge semantics, and structural properties?',
    bold_prefix='')
add_normal_para(anchor3,
    'RQ3: How does multi-graph integration reduce LLM hallucination compared to single-graph approaches, and which graph types contribute most to factual grounding?',
    bold_prefix='')
add_normal_para(anchor3,
    'RQ4: What is the computational trade-off between multi-graph expressiveness and inference efficiency, and can the overhead be bounded to practical levels?',
    bold_prefix='')

add_normal_para(anchor3,
    'These research questions are addressed systematically: RQ1 and RQ2 through the mathematical framework (Section 3) and architecture (Section 4), RQ3 through hallucination analysis (Section 6.4), and RQ4 through efficiency analysis (Section 6.6).')


# ============================================================
# IMPROVEMENT 3: Reduce/Focus Sections 3.1-3.3 + Add roadmap
# ============================================================
# Find Section 3 intro paragraph and modify it
idx_sec3 = find_para('This section formalizes the mathematical')
p3 = doc.paragraphs[idx_sec3]

# Replace the existing intro paragraph with a roadmap
for run in p3.runs:
    run.text = ''
p3.runs[0].text = (
    'This section presents the mathematical framework underlying MGNA as a progressive construction: '
    'Section 3.1 defines the three individual graph types (co-occurrence, sequence, knowledge) that capture '
    'complementary information from the corpus. Section 3.2 unifies these into a single multi-graph structure M '
    'that serves as the input to all subsequent operations. Section 3.3 defines the heterogeneous message-passing '
    'mechanism that learns node representations within M by aggregating information across edge types. Section 3.4 '
    'introduces the cross-graph attention mechanism that enables information flow between the three graph types, '
    'producing enriched representations. Finally, Section 3.5 specifies how these graph-derived representations '
    'are integrated with the LLM through soft prompting and hard retrieval. Each definition builds directly on '
    'the preceding ones, and together they constitute a complete, end-to-end specification of MGNA.'
)

# Add bridge sentences to sections 3.1, 3.2, 3.3
# After 3.1 (before 3.2)
idx_32 = find_para('3.2 Unified Multi-Graph')
anchor_32 = doc.paragraphs[idx_32]._element
add_normal_para(anchor_32,
    'These three graph definitions provide the building blocks for the unified multi-graph formalism defined in Section 3.2, which combines all three graph types into a single heterogeneous structure.')

# After 3.2 (before 3.3)
idx_33 = find_para('3.3 Heterogeneous Message Passing')
anchor_33 = doc.paragraphs[idx_33]._element
add_normal_para(anchor_33,
    'The multi-graph M defined above serves as the input structure for the heterogeneous message-passing mechanism in Section 3.3, which learns node representations by aggregating information across all edge types.')

# After 3.3 (before 3.4)
idx_34 = find_para('3.4 Cross-Graph Attention')
anchor_34 = doc.paragraphs[idx_34]._element
add_normal_para(anchor_34,
    'The node representations produced by message passing serve as inputs to the cross-graph attention mechanism in Section 3.4, which enables information exchange between the three graph types.')


# ============================================================
# IMPROVEMENT 4: Expand Section 3.4
# ============================================================
# Find the existing CrossAttn equation paragraph and add after it
idx_crossattn = find_para('CrossAttn(Ht, Hs) = softmax')
anchor_after_34 = doc.paragraphs[idx_crossattn]
# We'll insert after this paragraph, before 3.5
idx_35 = find_para('3.5 LLM Integration')
anchor_35 = doc.paragraphs[idx_35]._element

# Insert expanded content before 3.5
add_normal_para(anchor_35,
    'Gross Graph Attention. We now extend Definition 6 from pairwise cross-graph attention to a global mechanism operating over the entire multi-graph M. We term this "Gross Graph Attention" because it operates over the gross (complete) heterogeneous graph structure rather than individual within-graph neighborhoods. Formally, we define the full attention tensor A ∈ ℝ^(|V_M| × |V_M| × |O_E|) where each entry A[i,j,τ] represents the attention weight from node j to node i under edge type τ ∈ O_E. This tensor subsumes both within-graph attention (when nodes i and j belong to the same graph) and cross-graph attention (when they belong to different graphs).')

add_normal_para(anchor_35,
    'The pairwise cross-graph attention defined in Definition 6 is recovered as a special case by restricting indices: for source graph G_s and target graph G_t, CrossAttn(H_t, H_s) corresponds to the sub-tensor A[i,j,τ] where i ∈ V_t, j ∈ V_s, and τ is the cross-graph edge type connecting G_s to G_t. The complete cross-graph representation for node i aggregates information from all three graph types:')

add_normal_para(anchor_35,
    'h_i^(cross) = Σ_{g ∈ {co, seq, kg}} β_g · CrossAttn(H_i^target, H^g_source)')

add_normal_para(anchor_35,
    'where β_g ∈ ℝ are learned graph-level importance weights satisfying Σ_g β_g = 1, implemented via a softmax over learnable parameters. The weights β_g are query-adaptive: they are computed as β_g = softmax(w_g^T · h_query) where h_query is the query embedding and w_g are learnable vectors. This allows the model to dynamically emphasize different graph types depending on the nature of the query—factual queries increase β_kg while statistical or stylistic queries increase β_co or β_seq.')

add_normal_para(anchor_35,
    'Intuitively, the cross-graph attention enables the following information flows: from the co-occurrence graph to the knowledge graph, statistical association patterns provide contextual priors about which entity relationships are topically relevant; from the sequence graph to the knowledge graph, syntactic and positional structure disambiguates entity references and relation directions; from the knowledge graph to the co-occurrence and sequence graphs, factual constraints refine statistical associations by grounding them in verified relationships. These bidirectional flows are all captured by the attention tensor A and learned end-to-end. The architectural comparison in Section 2.1.1 (Figure B) illustrates how this cross-graph attention sits between message passing and LLM integration in the MGNA pipeline.')

add_normal_para(anchor_35,
    'The node representations produced by cross-graph attention feed directly into the LLM integration mechanism defined in Section 3.5.')


# ============================================================
# IMPROVEMENT 5: Expand Section 3.5
# ============================================================
# Find existing 3.5 content
idx_35_text = find_para('We integrate multi-graph representations')
p35 = doc.paragraphs[idx_35_text]

# Find Section 4 to insert before
idx_sec4 = find_para('4. MULTI-GRAPH NEURAL ARCHITECTURE')
anchor_sec4 = doc.paragraphs[idx_sec4]._element

# Add block scheme and descriptions before Section 4
add_normal_para(anchor_sec4, 'The complete LLM integration pipeline is illustrated in the following block scheme:')

integration_lines = [
    '┌─────────────────┐',
    '│  Multi-Graph M  │',
    '│  (Def. 4)       │',
    '└────────┬────────┘',
    '         │  Node features X',
    '         ▼',
    '┌─────────────────┐',
    '│  Message Passing │',
    '│  (Def. 5)       │',
    '│  L layers        │',
    '└────────┬────────┘',
    '         │  H_co^(L), H_seq^(L), H_kg^(L)',
    '         ▼',
    '┌─────────────────┐',
    '│  Cross-Graph    │',
    '│  Attention      │',
    '│  (Def. 6 +      │',
    '│   Gross Attn)   │',
    '└───┬─────────┬───┘',
    '    │         │',
    '    ▼         ▼',
    '┌────────┐ ┌────────────┐',
    '│  Hard  │ │    Soft    │',
    '│Retriev.│ │ Prompting  │',
    '│        │ │            │',
    '│Top-k   │ │ H_graph →  │',
    '│subgraph│ │ W_proj →   │',
    '│nodes   │ │ prefix P_g │',
    '│  C_k   │ │ (p tokens) │',
    '└───┬────┘ └─────┬──────┘',
    '    │            │',
    '    └─────┬──────┘',
    '          ▼',
    '┌──────────────────────┐',
    '│ Augmented Input:     │',
    '│ [P_g ; C_k ; x_q]   │',
    '└──────────┬───────────┘',
    '           ▼',
    '┌──────────────────────┐',
    '│     LLM f_theta      │',
    '└──────────┬───────────┘',
    '           ▼',
    '┌──────────────────────┐',
    '│     Output y         │',
    '└──────────────────────┘',
]
add_mono_before(anchor_sec4, integration_lines)

add_normal_para(anchor_sec4, 'Component-wise specification of the integration pipeline:')

add_normal_para(anchor_sec4,
    'Multi-Graph M (Definition 4). Input: Document corpus D. Operation: Parallel construction of G_co, G_seq, G_kg with initial node features X ∈ ℝ^(|V_M| × d) from pre-trained embeddings. Output: Multi-graph M = (V_M, E_M, φ, ψ, X).')

add_normal_para(anchor_sec4,
    'Message Passing (Definition 5). Input: Multi-graph M with features X. Operation: L iterations of multi-edge message passing with edge-type attention weights α_τ. Output: Updated representations H_co^(L), H_seq^(L), H_kg^(L) ∈ ℝ^(n_g × d) for each graph type.')

add_normal_para(anchor_sec4,
    'Cross-Graph Attention (Definition 6 + Gross Attention). Input: Per-graph representations {H_g^(L)}. Operation: Pairwise cross-graph attention with learned β_g weights. Output: Cross-graph enriched representations h_i^(cross) for all nodes i ∈ V_M.')

add_normal_para(anchor_sec4,
    'Soft Prompting. Input: Graph-level representations obtained by pooling h_i^(cross) per graph type. Operation: Linear projection W_proj ∈ ℝ^(d_graph × d_LLM) followed by reshaping into p prefix tokens P_g ∈ ℝ^(p × d_LLM). Output: Prefix token sequence P_g that prepends to the LLM input, steering generation through continuous representations. This corresponds to the soft prompting mechanism described above.')

add_normal_para(anchor_sec4,
    'Hard Retrieval. Input: Query embedding h_query and cross-graph node representations {h_i^(cross)}. Operation: Compute relevance scores s_i = h_query^T · h_i^(cross), select top-k nodes, extract associated text spans. Output: Context tokens C_k concatenated as discrete text input. This corresponds to the hard retrieval mechanism described above.')

add_normal_para(anchor_sec4,
    'LLM Generation. Input: Concatenated sequence [P_g; C_k; x_q] where x_q is the query. Operation: Standard autoregressive generation by f_theta. Output: Response y = f_theta(P_g, C_k, x_q).')

add_normal_para(anchor_sec4,
    'Together, Definitions 1–6 and the dual integration mechanisms (soft prompting and hard retrieval) constitute a complete specification of the Multi-Graph Neural Architecture (MGNA). Every stage has well-defined inputs and outputs: raw documents enter the graph construction pipeline (Definitions 1–3), are unified into a multi-graph (Definition 4), processed through message passing (Definition 5) and cross-graph attention (Definition 6), and finally integrated with the LLM through complementary soft and hard pathways. This end-to-end formalization ensures that MGNA is fully reproducible and that each component\'s contribution can be isolated through ablation (Section 6.3).')


# Save
doc.save('Multi_Graph_LLM_Second_Brain_v3.docx')
print("Saved v3 successfully!")
