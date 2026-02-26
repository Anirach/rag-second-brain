#!/usr/bin/env python3
"""Apply fixes 6 & 7 to MGNA v3."""
from docx import Document
from docx.shared import Pt

doc = Document('Multi_Graph_LLM_Second_Brain_v3.docx')

def find_para(prefix):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith(prefix):
            return i
    return None

def add_text(anchor_el, text, bold=False, font_size=None, font_name=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    if bold: r.bold = True
    if font_size: r.font.size = Pt(font_size)
    if font_name: r.font.name = font_name
    anchor_el.addprevious(p._element)
    return p

def add_mono(anchor_el, lines):
    for line in lines:
        p = doc.add_paragraph()
        r = p.add_run(line if line else ' ')
        r.font.name = 'Courier New'
        r.font.size = Pt(9)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = Pt(11)
        anchor_el.addprevious(p._element)

# ============================================================
# Step 1: Add 3.6, 3.7, 3.8 before Section 4
# ============================================================
idx_sec4 = find_para('4. MULTI-GRAPH NEURAL ARCHITECTURE')
anchor4 = doc.paragraphs[idx_sec4]._element

# --- 3.6 Algorithm 1 ---
add_text(anchor4, '3.6 Graph Construction Algorithm', bold=True, font_size=12)

add_mono(anchor4, [
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    'Algorithm 1: Multi-Graph Construction',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    'INPUT:  Corpus D = {d₁, ..., dₙ}, frequency threshold f_min,',
    '        sliding window size w, NER extractor E, relation',
    '        extractor R, pre-trained embedding model emb(·)',
    'OUTPUT: Multi-graph M = (V_M, E_M, φ, ψ, X) containing',
    '        G_co, G_seq, G_kg with shared node features',
    '──────────────────────────────────────────────────────────────',
    ' 1  V ← ExtractVocabulary(D, f_min)',
    ' 2  // ── Co-occurrence Graph (Definition 1) ──',
    ' 3  FOR each document d ∈ D:',
    ' 4      FOR each window of size w in d:',
    ' 5          Record co-occurrence counts for word pairs',
    ' 6  Compute PMI: W_co(v_i,v_j) = log[P(v_i,v_j)/(P(v_i)P(v_j))]',
    ' 7  E_co ← {(v_i,v_j) : PMI(v_i,v_j) > 0}',
    ' 8  G_co ← (V, E_co, W_co)',
    ' 9  // ── Sequence Graph (Definition 2) ──',
    '10  FOR each document d ∈ D:',
    '11      Parse dependency tree T_d using Stanza/spaCy',
    '12      FOR each dependency arc (head, dep, relation) in T_d:',
    '13          Add directed edge with relation label τ',
    '14      Add positional edges for adjacent tokens',
    '15  V_seq ← all tokens with positional IDs',
    '16  G_seq ← (V_seq, E_seq, τ)',
    '17  // ── Knowledge Graph (Definition 3) ──',
    '18  FOR each document d ∈ D:',
    '19      entities ← E(d)          // NER extraction',
    '20      Link entities to KB (Wikidata/UMLS)',
    '21      triples ← R(d, entities) // Relation extraction',
    '22      Filter triples by confidence threshold',
    '23  G_kg ← (Entities, Relations, Triples)',
    '24  // ── Unification (Definition 4) ──',
    '25  V_M ← V ∪ V_seq ∪ Entities  // align shared nodes',
    '26  E_M ← E_co ∪ E_seq ∪ Triples',
    '27  X ← emb(V_M) ∈ ℝ^(|V_M| × d_in)  // pre-trained init',
    '28  RETURN M = (V_M, E_M, φ, ψ, X)',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
])

add_text(anchor4,
    'Algorithm 1 implements Definitions 1–4 from Sections 3.1–3.2. The co-occurrence graph (line 6) uses PMI as specified in Definition 1, the sequence graph (lines 11–14) follows Definition 2 with dependency-based directed edges, and the knowledge graph (lines 18–23) instantiates Definition 3 via NER and relation extraction. The unification step (lines 25–27) constructs the multi-graph M of Definition 4 by merging node sets with alignment of shared nodes across graph types. Node features X are initialized from pre-trained language model embeddings, providing a semantically informed starting point for message passing.')

# --- 3.7 Algorithm 2 ---
add_text(anchor4, '3.7 MGNA Forward Pass', bold=True, font_size=12)

add_mono(anchor4, [
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    'Algorithm 2: MGNA Forward Pass (Encoder + Integration)',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    'INPUT:  Multi-graph M = (V_M, E_M, φ, ψ, X) from Algorithm 1',
    '        where X ∈ ℝ^(|V_M| × d_in) is the node feature matrix,',
    '        adjacency structure encoded in E_M with edge types O_E,',
    '        query x_q, number of layers L, top-k for retrieval',
    'OUTPUT: LLM response y',
    'PARAMS: W_msg^(τ,l), W_upd^(l) (message passing per layer/type)',
    '        α_τ (edge-type attention), W_Q, W_K, W_V (cross-attn)',
    '        β_g (graph importance), W_proj (soft prompt projection)',
    '──────────────────────────────────────────────────────────────',
    ' 1  // ── Initialize ──',
    ' 2  h_i^(0) ← X[i]  for all i ∈ V_M',
    ' 3  // ── Heterogeneous Message Passing (Definition 5) ──',
    ' 4  FOR l = 1 TO L:',
    ' 5      FOR each node i ∈ V_M:',
    ' 6          FOR each edge type τ ∈ O_E:',
    ' 7              msg_τ ← AGG({W_msg^(τ,l) · [h_i^(l-1)||h_j^(l-1)||e_ij^τ]',
    ' 8                          : j ∈ N_τ(i)})',
    ' 9          m_i^(l) ← Σ_τ α_τ · msg_τ    // α_τ learnable, Σα=1',
    '10          h_i^(l) ← UPDATE(h_i^(l-1), m_i^(l))',
    '11                  = LayerNorm(h_i^(l-1) + ReLU(W_upd^(l) · m_i^(l)))',
    '12  // ── Cross-Graph Attention (Definition 6 + Gross Attn) ──',
    '13  Partition nodes: V_co, V_seq, V_kg from V_M',
    '14  H_co^(L), H_seq^(L), H_kg^(L) ← node reps per graph type',
    '15  FOR each graph pair (source g_s, target g_t):',
    '16      Q ← H_{g_t}^(L) · W_Q,  K ← H_{g_s}^(L) · W_K,  V ← H_{g_s}^(L) · W_V',
    '17      CrossAttn(g_t, g_s) ← softmax(QK^T / √d) · V',
    '18  FOR each node i:',
    '19      h_i^(cross) ← Σ_{g∈{co,seq,kg}} β_g · CrossAttn(target_of(i), g)',
    '20          where β_g = softmax(w_g^T · h_query)  // query-adaptive',
    '21  // ── Readout: pool per graph type ──',
    '22  H_co ← MeanPool({h_i^(cross) : i ∈ V_co}) ∈ ℝ^d',
    '23  H_seq ← MeanPool({h_i^(cross) : i ∈ V_seq}) ∈ ℝ^d',
    '24  H_kg ← MeanPool({h_i^(cross) : i ∈ V_kg}) ∈ ℝ^d',
    '25  // ── Soft Prompting (Section 3.5) ──',
    '26  H_graph ← [H_co; H_seq; H_kg] ∈ ℝ^(3d)',
    '27  P_g ← Reshape(W_proj · H_graph, [p, d_LLM])  // p prefix tokens',
    '28  // ── Hard Retrieval (Section 3.5) ──',
    '29  h_query ← emb(x_q)',
    '30  s_i ← h_query^T · h_i^(cross)  for all i ∈ V_M',
    '31  C_k ← TextSpans(TopK({s_i}, k))  // top-k node text',
    '32  // ── LLM Generation ──',
    '33  input ← [P_g ; C_k ; x_q]   // concat prefix + context + query',
    '34  y ← LLM_f_theta(input)',
    '35  RETURN y',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
])

add_text(anchor4,
    'Algorithm 2 implements the complete MGNA inference pipeline, realizing Definitions 4, 5, and 6 as a concrete computational procedure. Lines 4–11 execute the heterogeneous message passing of Definition 5, where each layer aggregates typed messages from neighbors with learnable edge-type attention weights α_τ. Lines 12–20 implement the cross-graph attention of Definition 6 and its gross attention extension (Section 3.4), enabling information flow between co-occurrence, sequence, and knowledge graph representations. Lines 21–24 perform graph-level readout by mean-pooling cross-graph-enriched node representations per graph type, producing the three graph-level vectors H_co, H_seq, H_kg that serve as input to the dual LLM integration pathway of Section 3.5. The soft prompting path (lines 26–27) projects these vectors into the LLM\'s continuous embedding space as p learnable prefix tokens, while the hard retrieval path (lines 28–31) selects the top-k most query-relevant nodes and extracts their associated text spans as discrete context.')

# --- 3.8 Algorithm 3 ---
add_text(anchor4, '3.8 Training Procedure', bold=True, font_size=12)

add_mono(anchor4, [
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    'Algorithm 3: MGNA Training Procedure',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    'INPUT:  Corpus D, entity/relation annotations for KG,',
    '        QA pairs {(x_q, y*)} for fine-tuning,',
    '        hyperparameters: L, d, p, k, λ₁, λ₂, λ₃, lr, epochs',
    'OUTPUT: Trained parameters θ*',
    'PARAMS θ: {W_msg^(τ,l)}  — message passing weights (per type/layer)',
    '          {W_upd^(l)}    — update weights (per layer)',
    '          {α_τ}          — edge-type attention weights',
    '          {W_Q, W_K, W_V}— cross-graph attention projections',
    '          {β_g, w_g}     — graph importance weights',
    '          {W_proj}       — soft prompt projection matrix',
    '          {W_readout}    — readout layer weights',
    '──────────────────────────────────────────────────────────────',
    '  // ══ PHASE 1: Self-Supervised Pre-Training ══',
    ' 1  Construct M from D using Algorithm 1',
    ' 2  Initialize θ randomly (Xavier uniform)',
    ' 3  FOR epoch = 1 TO pre_train_epochs:',
    ' 4      // ── Loss 1: Masked Node Prediction (MNP) ──',
    ' 5      // Purpose: learn local graph structure',
    ' 6      V_mask ← randomly mask 15% of node features in X',
    ' 7      H ← MGNA_encode(M with masked X, θ)  // Alg.2 lines 1-24',
    ' 8      L_MNP ← -Σ_{v∈V_mask} log P(x_v | h_v^(L))',
    ' 9          // predict original features from neighborhood context',
    '10',
    '11      // ── Loss 2: Link Prediction (LP) ──',
    '12      // Purpose: learn cross-graph relational structure',
    '13      Sample positive edges E⁺ from E_M',
    '14      Sample negative edges E⁻ (random non-edges)',
    '15      L_LP ← -Σ_{(i,j)∈E⁺} log σ(h_i^T h_j)',
    '16             -Σ_{(i,j)∈E⁻} log(1 - σ(h_i^T h_j))',
    '17          // predicts edges across all three graph types',
    '18',
    '19      // ── Loss 3: Graph-Text Alignment (GTA) ──',
    '20      // Purpose: align graph reps with LLM embedding space',
    '21      T ← LLM token embeddings for node text labels',
    '22      L_GTA ← Σ_i ‖h_i^(cross) · W_align - T[i]‖²',
    '23          // ensures graph embeddings are LLM-compatible',
    '24',
    '25      // ── Combined pre-training loss ──',
    '26      L_total ← λ₁·L_MNP + λ₂·L_LP + λ₃·L_GTA',
    '27      θ ← θ - lr · ∇_θ L_total',
    '28',
    '  // ══ PHASE 2: Task-Specific Fine-Tuning ══',
    '29  FREEZE LLM parameters θ_LLM',
    '30  FOR epoch = 1 TO warmup_epochs:',
    '31      FOR each (x_q, y*) in QA pairs:',
    '32          y_pred ← Algorithm 2 full pipeline (M, x_q, θ)',
    '33          L_QA ← CrossEntropy(y_pred, y*)',
    '34          Update θ \\ θ_LLM  // graph params only',
    '35  UNFREEZE θ_LLM',
    '36  FOR epoch = 1 TO finetune_epochs:',
    '37      FOR each (x_q, y*) in QA pairs:',
    '38          y_pred ← Algorithm 2 full pipeline (M, x_q, θ)',
    '39          L_QA ← CrossEntropy(y_pred, y*)',
    '40          Update all θ (including θ_LLM) with reduced lr',
    '41  RETURN θ* ← θ',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
])

add_text(anchor4,
    'Algorithm 3 specifies the complete training procedure in two phases. Phase 1 (pre-training) optimizes the multi-graph encoder using three complementary self-supervised objectives without requiring labeled QA data: Masked Node Prediction (L_MNP, lines 6–9) learns local graph structure by reconstructing randomly masked node features from their neighborhood context; Link Prediction (L_LP, lines 13–17) learns cross-graph relational structure by distinguishing true edges from random negative samples across all three graph types; and Graph-Text Alignment (L_GTA, lines 21–23) ensures that the graph encoder produces representations compatible with the LLM\'s embedding space by minimizing the distance between graph node embeddings and corresponding LLM token embeddings. The combined loss (line 26) balances these objectives via hyperparameters λ₁, λ₂, λ₃.')

add_text(anchor4,
    'Phase 2 (fine-tuning, lines 29–40) adapts the pre-trained encoder to downstream QA tasks using a two-stage strategy: first, the LLM parameters are frozen and only graph encoder parameters are updated (warmup stage, lines 30–34), allowing the graph representations to adapt to the LLM\'s input distribution; then, all parameters including the LLM are jointly fine-tuned with a reduced learning rate (lines 35–40). This freeze-then-unfreeze strategy prevents catastrophic forgetting of the LLM\'s pre-trained knowledge while enabling end-to-end optimization. The complete parameter set θ includes all message passing weights, attention projections, graph importance weights, and the soft prompt projection matrix as enumerated in the algorithm header.')

# ============================================================
# Step 2: Replace 4.1-4.3 content with redirect
# ============================================================
# Remove paragraphs 223-234 (4.1 through 4.3 content) and replace with redirect
# We need indices for: "4.1 Graph Construction Pipeline" through end of 4.3
# Find fresh indices after insertions
idx_41 = None
idx_sec5 = None
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if t == '4.1 Graph Construction Pipeline':
        idx_41 = i
    if t.startswith('5. THEORETICAL ANALYSIS'):
        idx_sec5 = i

# Remove all paragraphs from 4.1 through just before Section 5
# by clearing their text and making them empty
if idx_41 and idx_sec5:
    # Clear 4.1 header through 4.3 content (everything between "4. MGNA" heading and "5.")
    # First find the "4. MULTI-GRAPH" heading
    idx_sec4_head = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == '4. MULTI-GRAPH NEURAL ARCHITECTURE (MGNA)':
            idx_sec4_head = i
            break
    
    # Remove paragraphs from idx_41 to idx_sec5-1
    to_remove = []
    for i in range(idx_41, idx_sec5):
        t = doc.paragraphs[i].text.strip()
        if t:  # only non-empty
            to_remove.append(i)
    
    # Clear content of those paragraphs
    for i in to_remove:
        p = doc.paragraphs[i]
        for run in p.runs:
            run.text = ''
    
    # Add redirect text after "4. MULTI-GRAPH..." heading
    # Insert before the now-cleared 4.1
    anchor_after_4head = doc.paragraphs[idx_41]._element
    add_text(anchor_after_4head,
        'The graph construction pipeline, neural network encoder forward pass, and training procedure are formalized as Algorithms 1–3 in Sections 3.6–3.8 respectively. These algorithms provide complete, self-contained specifications with explicit inputs, outputs, parameters, and step-by-step procedures that implement Definitions 1–6. The remainder of this section focuses on the theoretical properties of the resulting architecture.')

doc.save('Multi_Graph_LLM_Second_Brain_v3.docx')
print("Done!")
