from docx import Document
from copy import deepcopy

doc = Document('Multi_Graph_LLM_Second_Brain_v3.docx')

# Helper: find paragraph index by text prefix
def find_para(prefix, start=0):
    for i, p in enumerate(doc.paragraphs):
        if i >= start and p.text.strip().startswith(prefix):
            return i
    return None

# Helper: replace paragraph text preserving style
def set_text(idx, text):
    p = doc.paragraphs[idx]
    for run in p.runs:
        run.text = ""
    if p.runs:
        p.runs[0].text = text
    else:
        p.add_run(text)

def insert_para_after(idx, text, style='Normal'):
    """Insert a new paragraph after the given index"""
    p = doc.paragraphs[idx]
    new_p = doc.add_paragraph(text, style=style)
    # Move it after p
    p._element.addnext(new_p._element)
    return new_p

# ============================================================
# FIX 1: Rewrite Abstract
# ============================================================
abstract_idx = find_para("ABSTRACT")
# Current abstract is paragraphs 10-15 (Background, Objective, Methods, Results, Conclusions, Keywords)

new_abstract = """Large Language Models (LLMs) achieve remarkable fluency in natural language generation but remain prone to hallucination—producing plausible yet factually incorrect outputs—due to their sole reliance on parametric knowledge acquired during pre-training. Graph-augmented retrieval methods such as GraphRAG mitigate this by grounding generation in structured external knowledge, yet they employ only a single knowledge graph, capturing entity-relation triples while neglecting the statistical and sequential regularities that also underpin language understanding.

We propose the Multi-Graph Neural Architecture (MGNA), a framework that jointly leverages three complementary graph types—co-occurrence graphs encoding distributional word associations via PMI-weighted edges, sequence graphs preserving syntactic and positional token dependencies, and knowledge graphs supplying entity-relation triples—within a unified heterogeneous structure. A multi-edge message-passing mechanism with learnable edge-type attention propagates information within each graph, while a cross-graph attention module enables information flow across graph types at different levels of linguistic granularity. Graph representations are integrated with the LLM through dual pathways: soft prompting (projecting pooled graph embeddings into the LLM prefix space) and hard retrieval (selecting top-k relevant subgraph nodes as textual context).

Extensive experiments on seven benchmarks demonstrate that MGNA achieves 91.67% accuracy on domain-specific QA (compared with 33.33% for GPT-4 alone), a 23.4% average improvement on multi-hop reasoning tasks, and a 31.2% reduction in hallucination rate relative to GraphRAG, while incurring only 5.4% additional inference latency."""

# Replace paragraphs 10-14 with the new abstract
# First, clear existing abstract paragraphs
for idx in [10, 11, 12, 13, 14]:
    set_text(idx, "")

# Set new abstract in paragraph 10
set_text(10, new_abstract)

# ============================================================
# FIX 2: Figure Numbering - rename Figure A/B references
# ============================================================
# Change "Figure A:" to "Figure 1:" and "Figure B:" to be merged into Figure 1
for i, p in enumerate(doc.paragraphs):
    txt = p.text
    if "Figure A:" in txt:
        set_text(i, txt.replace("Figure A:", "Figure 1:"))
    if "Figure B:" in txt:
        set_text(i, txt.replace("Figure B: MGNA Pipeline (Proposed)", 
                                "Figure 1 (continued): MGNA Pipeline with Query-Time Inference"))
    # Add query flow note to MGNA figure caption area
    # Update any in-text figure references
    if "Figure A" in p.text and i not in [32]:
        set_text(i, p.text.replace("Figure A", "Figure 1"))
    if "Figure B" in p.text and i not in [63]:
        set_text(i, p.text.replace("Figure B", "Figure 1"))

# ============================================================
# FIX 3: Rewrite Definition 1 — Co-occurrence Graph
# ============================================================
def1_idx = find_para("Definition 1")
new_def1 = (
    "Definition 1 (Co-occurrence Graph). Prior to graph construction, each document d ∈ D is tokenized "
    "using a subword tokenizer (e.g., BPE or WordPiece) producing token sequence T = [t₁, ..., t_L]. "
    "Tokens with corpus frequency below f_min are excluded from the vocabulary V to reduce noise. "
    "The co-occurrence graph G_co = (V, E_co, W_co) is a weighted undirected graph where V is the filtered "
    "token vocabulary. A sliding window of size w is applied over the token sequence of each document; "
    "sentence boundaries DO interrupt co-occurrence windows—the window does not span across sentence "
    "boundaries, preserving sentence-level semantic coherence. Two tokens t_i and t_j co-occur if they "
    "appear within the same window, yielding an edge (t_i, t_j) ∈ E_co. "
    "The edge weight is the Pointwise Mutual Information (PMI) [Church & Hanks, 1990]:\n"
    "PMI(t_i, t_j) = log[ p(t_i, t_j) / (p(t_i) · p(t_j)) ]\n"
    "where p(t_i, t_j) is the joint probability estimated by co-occurrence counts within windows, and "
    "p(t_i), p(t_j) are unigram probabilities over the corpus. Only positive PMI values are retained (PPMI), "
    "i.e., E_co = {(t_i, t_j) : PMI(t_i, t_j) > 0}."
)
set_text(def1_idx, new_def1)

# Also update the PMI line (paragraph 143) to be empty since it's now inline
set_text(143, "")

# ============================================================
# FIX 4: Rewrite Definition 2 — Sequence Graph
# ============================================================
def2_idx = find_para("Definition 2")
new_def2 = (
    "Definition 2 (Sequence Graph). The sequence graph G_seq = (V_seq, E_seq, τ) is a directed graph "
    "encoding syntactic and positional dependencies between tokens. An edge (t_i, t_j) ∈ E_seq exists if "
    "t_j directly follows t_i in the token sequence (first-order adjacency) OR if a dependency arc from "
    "t_i to t_j exists in the dependency parse tree of the sentence. Formally, E_seq = E_adj ∪ E_dep where "
    "E_adj = {(t_i, t_{i+1})} captures sequential adjacency and E_dep contains typed dependency arcs "
    "(e.g., nsubj, dobj, amod) extracted via dependency parsing. The edge type function τ: E_seq → O_seq "
    "assigns each edge its syntactic relation label.\n\n"
    "Definition 2 as stated yields a first-order Markov model where each token depends only on its immediate "
    "predecessor. Higher-order sequence graphs (k-th order: edges spanning k positions) would capture "
    "longer-range dependencies but face state explosion: a k-th order graph over vocabulary V has O(|V|^k) "
    "potential edges, making it computationally intractable for large vocabularies. We address this limitation "
    "through two mechanisms: (1) the dependency parse arcs provide non-local structural connections without "
    "the exponential blowup; (2) we employ a Graph Structure Learner (GSL) [Zhu et al., 2021] that adaptively "
    "learns edge weights beyond first-order neighbors, effectively implementing a soft higher-order model "
    "within the message-passing framework."
)
set_text(def2_idx, new_def2)

# ============================================================
# FIX 5: Fix Definition 3 — Knowledge Graph (symbol overloading)
# ============================================================
def3_idx = find_para("Definition 3")
new_def3 = (
    "Definition 3 (Knowledge Graph). The knowledge graph G_kg = (ℰ, ℛ, 𝒯) is a directed labeled multigraph where:\n"
    "• ℰ is the set of entities (named or otherwise grounded concepts)\n"
    "• ℛ is the set of semantic relation types (e.g., is-a, part-of, causes, treats, located-in)—relation "
    "types are drawn from a predefined ontology or extracted by the relation extractor R\n"
    "• 𝒯 ⊆ ℰ × ℛ × ℰ is the set of triples (head entity, relation type, tail entity)\n\n"
    "Relations in ℛ are typed semantic predicates extracted by a fine-tuned relation extraction model. "
    "Examples include taxonomic (is-a, subclass-of), meronymic (part-of, contains), causal (causes, leads-to), "
    "and domain-specific relations depending on the corpus."
)
set_text(def3_idx, new_def3)

# Add Node Set Unification paragraph after Definition 3
# Paragraph 146 is "These three graph definitions..."
node_unification = (
    "Node Set Unification. An important question is whether the vocabulary V (co-occurrence graph), "
    "the token nodes V_seq (sequence graph), and the entities ℰ (knowledge graph) represent the same objects. "
    "The answer is NO—they operate at different levels of granularity:\n"
    "• V contains subword tokens from BPE/WordPiece tokenization (e.g., 'treat', '##ment')\n"
    "• V_seq also contains subword tokens and their positional instances—nodes in G_seq may represent "
    "token occurrences (position-sensitive) rather than token types\n"
    "• ℰ contains named entities and concepts at a higher semantic level (e.g., 'Treatment', 'Disease'), "
    "which may span multiple tokens\n"
    "The multi-graph node set V_M = V ∪ V_seq ∪ ℰ is therefore a union of heterogeneous node types. "
    "Cross-graph edges in the multi-graph M connect token nodes to entity nodes via a mention-to-entity "
    "alignment (entity linking), allowing information to flow across granularity levels. This heterogeneity "
    "is precisely what motivates the cross-graph attention mechanism in Section 3.4."
)
insert_para_after(def3_idx, node_unification)

# ============================================================
# FIX 5e: Fix E → ℰ notation globally
# ============================================================
# Fix paragraph 148 (multi-graph definition) and others
for i, p in enumerate(doc.paragraphs):
    txt = p.text
    changed = False
    # Fix "VM = V ∪ Vseq ∪ E" -> "VM = V ∪ Vseq ∪ ℰ"
    if "V ∪ Vseq ∪ E" in txt:
        txt = txt.replace("V ∪ Vseq ∪ E", "V ∪ Vseq ∪ ℰ")
        changed = True
    if "V_M ← V ∪ V_seq ∪ Entities" in txt:
        txt = txt.replace("V_M ← V ∪ V_seq ∪ Entities", "V_M ← V ∪ V_seq ∪ ℰ")
        changed = True
    # Fix "Gkg = (E, R, T)" patterns that aren't already fixed
    if "G_kg = (E, R, T)" in txt:
        txt = txt.replace("G_kg = (E, R, T)", "G_kg = (ℰ, ℛ, 𝒯)")
        changed = True
    if "Gkg = (E, R, T)" in txt:
        txt = txt.replace("Gkg = (E, R, T)", "Gkg = (ℰ, ℛ, 𝒯)")
        changed = True
    if changed:
        set_text(i, txt)

# ============================================================
# Add new references
# ============================================================
# Find last reference and add new ones
last_ref_idx = find_para("[65]")
church_ref = "[66] Church, K.W. & Hanks, P. (1990). Word association norms, mutual information, and lexicography. Computational Linguistics, 16(1), 22–29."
zhu_ref = "[67] Zhu, Y., Xu, W., Zhang, J., et al. (2021). Deep graph structure learning for robust representations: A survey. IJCAI. https://arxiv.org/abs/2103.03036"

insert_para_after(last_ref_idx, zhu_ref)
insert_para_after(last_ref_idx, church_ref)

# Update header counts
for i, p in enumerate(doc.paragraphs):
    if "References: 65" in p.text:
        set_text(i, p.text.replace("References: 65", "References: 67"))

# Save
doc.save('Multi_Graph_LLM_Second_Brain_v4.docx')
print("DONE: saved v4")
