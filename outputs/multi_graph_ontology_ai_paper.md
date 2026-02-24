# Using Multi-Graph and Ontology as a Knowledge Representation for AI-Systems: A Comprehensive Survey

**Authors:** Research Paper prepared by AI Research Assistant  
**Date:** February 2026  
**Keywords:** Knowledge Representation, Multi-Graph, Ontology, Artificial Intelligence, Graph Neural Networks, Neuro-Symbolic AI, Knowledge Graphs

---

## Abstract

Knowledge representation remains a fundamental challenge in artificial intelligence, requiring systems that can effectively capture, reason over, and leverage complex real-world information. This paper presents a comprehensive survey on the integration of multi-graph structures and ontologies as unified knowledge representation frameworks for AI systems. We examine how multi-relational graphs, including heterogeneous knowledge graphs and temporal knowledge graphs, provide flexible structures for encoding complex relationships, while ontologies offer formal semantic foundations through description logics and inference mechanisms. The survey covers recent advances in graph neural networks for knowledge representation, ontology-based reasoning systems, and emerging neuro-symbolic approaches that bridge the gap between sub-symbolic learning and symbolic reasoning. We analyze integration methodologies including knowledge graph embeddings, ontology-enhanced neural architectures, and retrieval-augmented generation systems. Applications across healthcare, scientific discovery, and enterprise knowledge management demonstrate the practical impact of these hybrid representations. Finally, we identify key challenges in scalability, reasoning complexity, and knowledge integration, and outline promising future research directions toward more robust, explainable, and contextually-aware AI systems. This survey synthesizes findings from over 30 recent publications (2024-2026) to provide researchers and practitioners with a comprehensive understanding of the current state and future potential of multi-graph and ontology-based knowledge representation in artificial intelligence.

---

## 1. Introduction

### 1.1 Motivation and Background

The rapid advancement of artificial intelligence systems has created an unprecedented demand for sophisticated knowledge representation mechanisms that can effectively capture the complexity, heterogeneity, and dynamism of real-world information [1, 2]. Traditional AI approaches often relied on either purely symbolic systems with handcrafted rules or purely statistical methods that learn patterns from data. However, neither approach alone has proven sufficient for building robust, generalizable, and explainable AI systems capable of complex reasoning tasks [3].

Knowledge graphs have emerged as a powerful paradigm for representing structured information, encoding entities as nodes and relationships as edges in graph structures [4]. Major technology companies including Google, Microsoft, and Meta have deployed large-scale knowledge graphs containing billions of facts to power search engines, recommendation systems, and conversational AI [5]. Yet traditional knowledge graphs, while effective for basic relationship representation, often struggle with the multi-faceted nature of real-world knowledge that requires capturing multiple types of relationships, temporal dynamics, and hierarchical semantic structures simultaneously.

Ontologies, rooted in formal logic and philosophical traditions of knowledge organization, provide complementary capabilities through explicit specification of conceptualizations [6]. Using description logics and Web Ontology Language (OWL), ontologies enable formal reasoning, consistency checking, and inference of implicit knowledge [7]. The integration of ontological frameworks with graph-based representations promises AI systems that combine the flexibility of data-driven learning with the rigor of logical reasoning.

### 1.2 Research Questions and Scope

This survey addresses the following fundamental research questions:

1. How can multi-graph structures effectively represent heterogeneous, multi-relational, and temporal knowledge for AI systems?
2. What role do ontologies play in providing semantic foundations and reasoning capabilities for knowledge-intensive AI applications?
3. What integration approaches successfully combine the strengths of multi-graph representations with ontological frameworks?
4. What are the current applications, challenges, and future directions for these hybrid knowledge representation systems?

### 1.3 Contributions

The main contributions of this paper are:

- A comprehensive taxonomy of multi-graph knowledge representation approaches including heterogeneous graphs, temporal knowledge graphs, and multi-modal knowledge graphs
- A systematic review of ontology-based systems and their integration with machine learning methods
- Analysis of neuro-symbolic integration approaches that combine graph neural networks with symbolic reasoning
- Survey of applications across healthcare, scientific discovery, and enterprise domains
- Identification of open challenges and future research directions based on analysis of over 30 recent publications

### 1.4 Paper Organization

The remainder of this paper is organized as follows. Section 2 provides background on knowledge representation and related work. Section 3 discusses multi-graph representations for AI systems. Section 4 examines ontology-based knowledge systems. Section 5 presents integration approaches combining multi-graph and ontological methods. Section 6 surveys applications and case studies. Section 7 discusses challenges and future directions. Section 8 concludes the paper.

---

## 2. Background and Related Work

### 2.1 Fundamentals of Knowledge Representation

Knowledge representation in artificial intelligence encompasses the methods and structures used to encode information about the world in a form that computer systems can utilize for reasoning and problem-solving [8]. Effective knowledge representation must balance expressiveness—the ability to capture complex relationships and nuances—with computational tractability for inference and learning tasks.

Historically, knowledge representation approaches can be categorized into several paradigms. Symbolic approaches, including logic-based systems and semantic networks, emphasize explicit representation of concepts and rules [9]. Connectionist approaches utilize distributed representations in neural networks where knowledge is implicitly encoded in network weights [10]. More recently, hybrid neuro-symbolic approaches seek to combine the strengths of both paradigms [3, 11].

### 2.2 Knowledge Graphs: Foundations and Evolution

A knowledge graph can be formally defined as a directed labeled graph G = (E, R, T) where E represents the set of entities, R the set of relation types, and T ⊆ E × R × E the set of triples (facts) connecting entities through relations [4]. This simple yet powerful representation has enabled numerous applications from question answering to recommendation systems.

The evolution of knowledge graphs has progressed through several generations. First-generation knowledge graphs focused primarily on simple factual relationships extracted from structured databases [12]. Second-generation systems incorporated natural language processing to extract knowledge from unstructured text and expanded relation types [13]. Current third-generation knowledge graphs increasingly integrate multi-modal information, temporal dynamics, and probabilistic reasoning capabilities [14].

Major knowledge graphs include Freebase, Wikidata, DBpedia, and proprietary systems like Google's Knowledge Graph and Microsoft's Satori [5]. These resources contain millions to billions of entities and facts, serving as foundation for various AI applications. However, knowledge graphs face inherent challenges including incompleteness, noise, and difficulty representing complex semantic constraints [15].

### 2.3 Ontologies in Artificial Intelligence

Ontologies provide formal, explicit specifications of shared conceptualizations for particular domains [6]. In the context of AI, ontologies serve multiple purposes: defining vocabularies and taxonomies, specifying constraints and rules, enabling interoperability across systems, and supporting automated reasoning [16].

The Web Ontology Language (OWL), built on description logics, has become the standard for representing ontologies on the Semantic Web [7]. OWL supports various expressivity levels, from OWL Lite for simple taxonomies to OWL Full for maximum expressiveness. Key constructs include class hierarchies, property definitions (object and datatype properties), cardinality restrictions, and logical axioms [17].

Ontological reasoning enables inference of implicit knowledge from explicit assertions. For example, given that "all mammals are warm-blooded" and "dogs are mammals," a reasoner can infer that "dogs are warm-blooded." Reasoners like Pellet, HermiT, and FaCT++ implement decision procedures for various description logic fragments [18].

### 2.4 Related Surveys and Positioning

Several surveys have addressed aspects of knowledge representation for AI. Wang et al. [19] provided a comprehensive survey of knowledge graph embedding methods. Ji et al. [4] surveyed knowledge graphs covering construction, representation learning, and applications. Hogan et al. [5] offered a detailed examination of knowledge graph concepts and systems.

In the ontology domain, Keet [20] surveyed ontology engineering methodologies. Fensel et al. [21] examined ontologies for enterprise applications. More recently, Giglou et al. [22] explored the intersection of large language models and ontology learning.

This survey differentiates itself by specifically focusing on the integration of multi-graph structures with ontological frameworks, examining recent advances in neuro-symbolic approaches, and synthesizing findings from the most recent literature (2024-2026) to provide an up-to-date perspective on this rapidly evolving field.

---

## 3. Multi-Graph Representations for AI

### 3.1 Heterogeneous Information Networks

Heterogeneous information networks (HINs) extend traditional homogeneous graphs by incorporating multiple types of nodes and edges [23]. Formally, a heterogeneous graph can be defined as G = (V, E, A, R) where V is the node set, E the edge set, A the node type mapping function, and R the edge type mapping function [24].

HINs naturally model many real-world scenarios where different entity types interact through various relationship types. For example, an academic network contains authors, papers, venues, and institutions connected through relationships like "authored," "published-in," and "affiliated-with." This rich structure enables more nuanced analysis than homogeneous graph representations.

Meta-paths, defined as composite relations connecting node types through specific sequences of edge types, provide powerful mechanisms for capturing semantic relationships in HINs [25]. For instance, the meta-path "Author-Paper-Author" captures co-authorship relationships, while "Author-Paper-Venue-Paper-Author" captures authors publishing in the same venues. Meta-path-based methods have achieved state-of-the-art results in various tasks including node classification, link prediction, and recommendation [26].

Recent advances have introduced hierarchical heterogeneous graph transformers (HHGT) that capture both local structural patterns and global semantic relationships [27]. These architectures employ attention mechanisms at multiple levels: node-level attention for aggregating neighbor information, semantic-level attention for combining different meta-paths, and hierarchical attention for integrating multi-scale representations.

### 3.2 Temporal Knowledge Graphs

Temporal knowledge graphs (TKGs) extend static knowledge graphs by incorporating temporal information, enabling representation of facts that change over time [28]. A temporal fact can be represented as a quadruple (s, r, o, t) where s is the subject entity, r the relation, o the object entity, and t the temporal annotation (timestamp or time interval) [29].

TKGs support two primary reasoning tasks: interpolation (reasoning about facts within observed time periods) and extrapolation (predicting future facts based on historical patterns) [30]. Interpolation methods leverage temporal constraints to improve knowledge graph completion, while extrapolation methods model temporal dynamics for forecasting.

Representation learning methods for TKGs can be categorized into temporal embedding approaches, recurrent architectures, and graph neural network-based methods [28]. Temporal embedding approaches like TTransE, HyTE, and TA-DistMult extend static embedding methods with temporal components [31]. Recurrent approaches like RE-NET and CyGNet model temporal sequences using LSTM or GRU architectures [32]. Graph neural network methods like TGAT and TGN aggregate temporal neighborhood information through attention mechanisms [33].

The Temporal Graph Benchmark 2.0 (TGB 2.0) introduced in 2024 provides standardized evaluation protocols for temporal knowledge graph methods, including both temporal knowledge graph datasets and temporal heterogeneous graph datasets [34]. This benchmark enables fair comparison across different approaches and has revealed significant performance gaps between current methods and optimal performance, indicating substantial room for improvement.

### 3.3 Multi-Modal Knowledge Graphs

Multi-modal knowledge graphs (MMKGs) integrate information from multiple modalities including text, images, video, and audio [35]. As real-world knowledge inherently exists in multi-modal forms, MMKGs provide richer representations than text-only knowledge graphs.

Construction of MMKGs involves several challenges: aligning entities across modalities, extracting relationships from non-textual data, and fusing information from heterogeneous sources [36]. Recent approaches leverage vision-language models for image-text alignment and employ multi-modal transformers for cross-modal reasoning [37].

The MOSAIC benchmark introduced comprehensive evaluation protocols for multi-modal graph learning, covering eight datasets across vision, text, and cross-modal tasks [38]. Results indicate that while current methods achieve reasonable performance on within-modality tasks, cross-modal reasoning remains challenging, particularly for tasks requiring deep semantic understanding.

Multi-modal knowledge graph reasoning supports applications including visual question answering, cross-modal retrieval, and multi-modal recommendation [39]. The noise-powered multi-modal knowledge graph representation framework demonstrates how strategic noise injection during training can improve robustness of multi-modal representations [40].

### 3.4 Knowledge Graph Embedding Methods

Knowledge graph embedding (KGE) methods learn low-dimensional vector representations of entities and relations that preserve structural and semantic properties of the graph [41]. These embeddings enable efficient similarity computation, link prediction, and integration with downstream neural models.

Translation-based models, initiated by TransE [42], interpret relations as translations in embedding space: h + r ≈ t for valid triples (h, r, t). Extensions include TransH (hyperplane-based), TransR (relation-specific spaces), and TransD (dynamic matrices) [43]. While elegant, translation-based models struggle with complex relation patterns including symmetry, inversion, and composition.

Semantic matching models like RESCAL, DistMult, and ComplEx use bilinear or tensor operations to model triple plausibility [44]. DistMult uses diagonal relation matrices, achieving efficiency but limiting expressiveness to symmetric relations. ComplEx extends to complex-valued embeddings, enabling asymmetric relation modeling [45].

RotatE represents relations as rotations in complex space, elegantly handling symmetric, asymmetric, inverse, and composition patterns [46]. The model defines relations as element-wise rotation: t = h ∘ r where ∘ denotes the Hadamard product in complex space. Recent extensions include QuatE (quaternion embeddings) and DualE (dual quaternions) for richer geometric representations [47].

A comprehensive survey on capturing relation properties identified that embedding space geometry significantly impacts which relation patterns can be modeled [48]. Translation-based models perform better with pairwise losses, while multiplicative models benefit from pointwise losses, suggesting that model architecture and training objective selection should be jointly considered.

---

## 4. Ontology-Based Knowledge Systems

### 4.1 Ontology Languages and Formalisms

The Web Ontology Language (OWL) provides the foundation for formal ontology representation on the Semantic Web [7]. OWL 2, the current version, offers three profiles optimized for different use cases: OWL 2 EL for large ontologies with polynomial-time reasoning, OWL 2 QL for query-answering applications, and OWL 2 RL for rule-based reasoning [49].

Description logics (DLs) provide the formal semantics underlying OWL [50]. The expressivity-tractability tradeoff in DLs is well-studied: more expressive logics enable richer modeling but incur higher computational complexity for reasoning. The SROIQ description logic underlying OWL 2 DL supports concept constructors (intersection, union, complement), role constructors (inverse, composition, transitivity), and cardinality restrictions while maintaining decidability [51].

Recent work has explored extensions to standard description logics for handling uncertainty and probabilistic knowledge [52]. Probabilistic description logics like Prob-EL and BEL enable representation of uncertain knowledge with associated confidence scores. These extensions are particularly relevant for integrating ontological knowledge with machine learning systems that inherently produce probabilistic outputs.

### 4.2 Ontology Reasoning and Inference

Ontological reasoning enables inference of implicit knowledge from explicit assertions and axioms [18]. Key reasoning tasks include consistency checking, concept satisfiability, subsumption, and instance checking. Modern reasoners implement optimized tableau algorithms and employ various optimization techniques including absorption, model caching, and parallelization [53].

Classification reasoning computes the complete subsumption hierarchy of concepts in an ontology, enabling efficient query answering [54]. Realization determines the most specific concepts that each individual belongs to. Conjunctive query answering extends instance retrieval to complex queries involving multiple variables and joins.

Rule-based reasoning systems complement DL reasoning by supporting Horn-clause style rules [55]. SWRL (Semantic Web Rule Language) combines OWL with RuleML, enabling expressive rule definitions while maintaining decidability for restricted rule forms. The Jena Rules engine, Drools, and CLIPS provide efficient rule execution for practical applications.

Recent developments have explored integration of ontological reasoning with large language models. The DeepOnto package provides a Python interface for ontology engineering with deep learning, bridging the gap between traditional OWL-based systems and modern neural approaches [56]. The framework supports ontology processing, reasoning invocation, and embedding generation through a unified API.

### 4.3 Ontology Embeddings

Ontology embedding methods project ontological entities and axioms into continuous vector spaces while preserving semantic relationships [57]. Unlike knowledge graph embeddings that focus primarily on triple patterns, ontology embeddings must additionally capture logical axioms including subsumption, disjointness, and restrictions.

OWL2Vec* represents the state-of-the-art in ontology embedding, combining structural, lexical, and logical information [58]. The method extracts three types of corpora from ontologies: a structure document capturing graph topology, a lexical document incorporating labels and descriptions, and a combined document integrating both. Word embedding models (Word2Vec or transformers) trained on these corpora produce embeddings that support similarity computation, link prediction, and classification.

Onto2Vec and OPA2Vec provide alternative approaches focusing on axiom serialization [59]. These methods treat ontology axioms as sentences in a corpus, applying language modeling techniques to learn distributed representations. Extensions incorporate annotation properties and external text corpora to enhance embedding quality.

Recent work has explored ontology embeddings for alignment tasks [60]. OWL2Vec4OA adapts ontology embeddings for ontology alignment, achieving competitive performance with specialized alignment systems. The approach demonstrates that general-purpose ontology embeddings can support diverse downstream tasks without task-specific modifications.

### 4.4 Integration of Ontologies with Machine Learning

The integration of ontologies with machine learning has emerged as a significant research direction, combining the formal semantics of ontological knowledge with the pattern recognition capabilities of neural networks [61]. This integration takes multiple forms: using ontologies to structure neural network architectures, incorporating ontological constraints into learning objectives, and leveraging ontologies for interpretability.

Ontology-guided neural architecture design uses class hierarchies to structure output layers [62]. For classification tasks with hierarchically organized classes, neural networks can incorporate ontological structure through hierarchical softmax or embedding spaces that respect subsumption relationships. This approach improves both accuracy and interpretability, particularly for fine-grained classification with many classes.

Incorporating ontological constraints as regularization terms guides learning toward semantically consistent representations [63]. Loss functions can penalize predictions violating domain constraints like disjointness or cardinality restrictions. Knowledge distillation methods transfer ontological knowledge from symbolic reasoners to neural networks [64].

A systematic literature review on combining machine learning and ontology identified 127 papers employing various integration strategies [65]. The review found increasing adoption of hybrid approaches, with applications spanning natural language processing, computer vision, and healthcare. Key challenges include scalability to large ontologies and bridging the representation gap between logical and neural formalisms.

---

## 5. Integration Approaches: Multi-Graph Meets Ontology

### 5.1 Neuro-Symbolic AI Paradigms

Neuro-symbolic AI represents a paradigm shift toward systems that combine the learning capabilities of neural networks with the reasoning strengths of symbolic AI [3]. In the context of knowledge representation, neuro-symbolic approaches integrate graph-based learning with ontological reasoning to achieve systems that are both data-efficient and capable of systematic generalization.

A systematic review of neuro-symbolic AI in 2024 analyzed 167 papers meeting inclusion criteria [11]. The review identified five primary research areas: learning and inference (63% of papers), logic and reasoning (35%), knowledge representation (44%), explainability and trustworthiness (28%), and meta-cognition (5%). The concentration of research in learning and knowledge representation indicates the centrality of knowledge graph and ontology integration to the neuro-symbolic agenda.

DeLong et al. [66] propose a taxonomy of neuro-symbolic approaches for knowledge graph reasoning with three major categories: (1) logically-informed embedding approaches that use symbolic inference to enhance training data, (2) embedding approaches with logical constraints that incorporate symbolic knowledge into neural learning objectives, and (3) rule learning approaches that extract symbolic rules from neural representations. This taxonomy provides a useful framework for understanding the design space of integration methods.

### 5.2 Knowledge-Enhanced Language Models

The integration of knowledge graphs and ontologies with large language models (LLMs) has emerged as a major research direction for improving LLM reliability and grounding [67]. While LLMs demonstrate impressive natural language capabilities, they suffer from hallucination, limited factual accuracy, and inability to incorporate updated knowledge. Knowledge enhancement addresses these limitations through retrieval-augmented generation and knowledge-guided architectures.

Retrieval-augmented generation (RAG) systems retrieve relevant knowledge from external sources to augment LLM context during generation [68]. Graph-based RAG (GraphRAG) specifically leverages knowledge graph structure for retrieval, enabling more semantically coherent context construction than document-based retrieval [69]. Recent work has introduced GNN-RAG, combining graph neural network retrieval with language model generation for improved factual accuracy [70].

Knowledge graph language (KGL-LLM) provides a dedicated language for precise LLM-KG integration, reducing completion errors through real-time context retrieval [71]. The framework demonstrates that structured interfaces between LLMs and knowledge graphs can significantly improve accuracy compared to naive text-based integration.

Think-on-Graph 2.0 introduces deep and interpretable reasoning by allowing LLMs to perform multi-hop reasoning over knowledge graphs with explicit reasoning traces [72]. The approach achieves state-of-the-art results on complex question answering while providing interpretable reasoning paths that can be verified against the knowledge graph.

### 5.3 Ontology-Constrained Graph Neural Networks

Graph neural networks (GNNs) have demonstrated powerful capabilities for learning over graph-structured data, making them natural candidates for knowledge graph representation learning [73]. Integrating ontological constraints into GNNs enables incorporating domain knowledge into the learning process while maintaining the flexibility of neural approaches.

Relational graph convolutional networks (R-GCN) extend GCNs to multi-relational graphs characteristic of knowledge graphs [74]. R-GCNs learn relation-specific transformation matrices, enabling differentiated message passing based on edge types. While effective for knowledge graph completion, R-GCNs have been shown to not learn sound logical rules in certain cases, highlighting the gap between statistical pattern learning and logical reasoning [75].

CompGCN introduces composition-based multi-relational graph convolutions that jointly embed entities and relations [76]. By treating relations as first-class objects in the embedding space, CompGCN achieves more expressive representations than R-GCN while maintaining computational efficiency through parameter sharing.

Ontology-enhanced GNNs incorporate class hierarchy and property constraints into graph neural network architectures [77]. Type constraints can be enforced through masked attention mechanisms that prevent invalid entity combinations. Subsumption relationships can guide hierarchical pooling operations for multi-scale graph representations.

### 5.4 Hybrid Reasoning Systems

Hybrid reasoning systems combine neural embedding-based inference with symbolic rule-based reasoning for more complete and reliable knowledge graph reasoning [78]. Neural methods excel at handling noisy data and discovering statistical patterns, while symbolic methods provide guaranteed soundness and interpretability.

Neural-symbolic integration for knowledge graph completion uses embedding predictions to seed rule learning, with learned rules refining embedding predictions [79]. The iterative interaction between neural and symbolic components enables each to compensate for the other's weaknesses. Experimental results demonstrate that hybrid approaches outperform either neural or symbolic methods alone across multiple benchmarks.

Neuro-symbolic AI for conflict-aware learning addresses the challenge of conflicting information in knowledge graphs from heterogeneous sources [80]. The approach uses symbolic reasoning to detect logical conflicts and neural methods to resolve them based on source reliability and temporal factors. This capability is essential for practical knowledge graph systems that must integrate information of varying quality.

The integration of probabilistic reasoning with ontological knowledge has produced sophisticated hybrid systems [81]. Probabilistic soft logic (PSL) enables weighted first-order rules over knowledge graphs, with weights learned from data. Combined with ontological background knowledge, PSL systems can perform nuanced reasoning under uncertainty while respecting domain constraints.

---

## 6. Applications and Case Studies

### 6.1 Healthcare and Biomedical Applications

Healthcare represents a domain where the integration of multi-graph and ontological knowledge representation has achieved significant practical impact [82]. Medical ontologies including SNOMED CT, the Unified Medical Language System (UMLS), and the Gene Ontology provide foundational terminologies and relationships for clinical and biomedical applications.

DR.KNOWS (Diagnostic Reasoning Knowledge Graph System) demonstrates the integration of medical knowledge graphs with large language models for improved diagnostic prediction [83]. The system retrieves case-specific knowledge paths from a medical knowledge graph and provides them as context to LLMs, improving diagnostic accuracy while maintaining interpretability through explicit knowledge graph grounding.

Biomedical knowledge graphs such as iKraph integrate relation data from public databases with high-throughput genomics datasets, creating comprehensive resources for drug discovery and repurposing [84]. A survey on biomedical knowledge graphs identified applications spanning drug-target interaction prediction, disease gene prioritization, and clinical decision support [85].

The construction of sepsis care knowledge graphs using LLM-driven pipelines illustrates the potential for automated knowledge graph creation in healthcare [86]. By combining structured clinical data with unstructured clinical notes, these systems create comprehensive knowledge representations that support clinical decision-making.

### 6.2 Scientific Discovery and Research

Knowledge graphs and ontologies increasingly support scientific discovery across multiple domains [87]. Scientific knowledge graphs capture entities such as researchers, publications, datasets, and concepts along with relationships like authorship, citation, and methodology application.

The OpenAlex knowledge graph indexes over 200 million scholarly works with rich metadata and relationship information [88]. Integration with domain ontologies enables sophisticated queries across the scientific literature, supporting systematic reviews, research trend analysis, and collaboration discovery.

Materials science knowledge graphs encode information about chemical compounds, synthesis procedures, and material properties [89]. Combined with machine learning models, these knowledge graphs enable prediction of novel materials with desired properties, accelerating the materials discovery pipeline.

GraphRAG approaches have been applied to engineering research, combining knowledge graph structure with retrieval-augmented generation for context-aware question answering [90]. The approach demonstrates improved accuracy compared to traditional RAG, particularly for queries requiring multi-hop reasoning over technical knowledge.

### 6.3 Enterprise Knowledge Management

Enterprise knowledge management applications leverage knowledge graphs and ontologies for information integration, search, and analytics [91]. Organizations increasingly adopt knowledge graph technologies to unify data silos and provide intelligent access to institutional knowledge.

Enterprise knowledge graphs integrate information from diverse sources including databases, documents, and streaming data [92]. Ontologies provide standardized vocabularies and semantic interoperability across business units. Applications include enterprise search, recommendation systems, and business intelligence dashboards.

The integration of knowledge graphs with conversational AI enables intelligent virtual assistants for enterprise applications [93]. By grounding responses in organizational knowledge graphs, these systems provide accurate, up-to-date information while maintaining enterprise data governance requirements.

Multi-level refined knowledge graph design supports healthcare chatbot systems that combine structured medical knowledge with conversational capabilities [94]. The hierarchical ontological structure enables appropriate response generation at different levels of specificity based on user expertise and query context.

### 6.4 Autonomous Systems and Robotics

Knowledge representation for autonomous systems requires integration of symbolic reasoning with real-time perception and action [95]. Ontologies defining world models, action capabilities, and safety constraints combine with learned representations from sensor data.

Neuro-symbolic AI approaches for robustness, uncertainty quantification, and intervenability address critical requirements for autonomous systems [96]. Symbolic knowledge provides interpretable decision frameworks while neural components handle perceptual uncertainty and environment variability.

Knowledge graph-based reasoning supports autonomous driving through integration of traffic rules, road networks, and dynamic scene understanding [97]. The combination of ontological traffic regulations with learned trajectory prediction enables both rule-compliant and anticipatory driving behavior.

---

## 7. Challenges and Future Directions

### 7.1 Scalability Challenges

Scalability remains a fundamental challenge for multi-graph and ontology-based knowledge representation [98]. Large-scale knowledge graphs containing billions of entities and triples require efficient storage, indexing, and query processing. Ontological reasoning over such large knowledge bases can become computationally prohibitive, particularly for expressive description logics.

Distributed knowledge graph systems partition large graphs across multiple nodes, enabling parallel processing and horizontal scaling [99]. Challenges include maintaining consistency during updates, optimizing distributed query execution, and balancing load across partitions. Graph partitioning algorithms must consider both structural properties and query workload characteristics.

Approximate reasoning techniques trade precision for scalability in ontological inference [100]. Incomplete reasoning methods return sound but potentially incomplete results, enabling practical deployment on large ontologies. Anytime reasoning approaches progressively improve result completeness given additional computation time.

### 7.2 Knowledge Integration and Alignment

Integrating knowledge from heterogeneous sources with different schemas, granularities, and quality levels presents significant challenges [101]. Ontology alignment identifies correspondences between concepts in different ontologies, enabling semantic interoperability. Entity alignment in knowledge graphs links mentions of the same real-world entity across different graphs.

Multi-view knowledge graph fusion addresses the challenge of integrating different perspectives on the same domain [102]. Global-aware convolution processes relational and entity features separately, while multi-perspective entity embeddings capture different aspects of entity semantics. The challenge lies in appropriately weighting and combining views without losing discriminative information.

Knowledge graph construction from unstructured text using large language models shows promise for automated knowledge acquisition [103]. However, challenges remain in ensuring extraction accuracy, handling temporal information, and maintaining consistency with existing knowledge. Iterative refinement approaches that combine LLM extraction with human validation offer practical solutions.

### 7.3 Reasoning Under Uncertainty

Real-world knowledge inherently involves uncertainty from incomplete information, conflicting sources, and inherent ambiguity [104]. Both knowledge graphs and ontologies must accommodate uncertainty while enabling meaningful reasoning.

Probabilistic knowledge graphs represent facts with associated confidence scores, enabling reasoning that appropriately propagates uncertainty [105]. Challenges include learning confidence scores from data, combining evidence from multiple sources, and maintaining computational tractability for probabilistic inference.

Fuzzy description logics extend classical description logics with graded truth values [106]. These extensions enable representation of vague concepts like "tall" or "young" that lack crisp boundaries. Integration with neural networks that produce continuous outputs provides natural interfaces between symbolic and sub-symbolic reasoning.

### 7.4 Explainability and Trustworthiness

As AI systems based on knowledge graphs and ontologies are deployed in high-stakes applications, explainability and trustworthiness become critical requirements [107]. Users must understand and verify system reasoning, particularly for decisions affecting health, safety, or rights.

Knowledge graph-grounded explanations leverage explicit graph paths to justify predictions [108]. Unlike opaque neural models, knowledge graph reasoning can expose the specific facts and relationships supporting a conclusion. Challenges include generating human-understandable explanations and handling the complexity of multi-hop reasoning.

Ontological explanations draw on formal logical structures to provide principled justifications [109]. Reasoner explanations identify the specific axioms entailing a conclusion. User studies indicate that structured explanations improve user trust and appropriate reliance on AI systems.

### 7.5 Future Research Directions

Several promising directions emerge from the current state of multi-graph and ontology-based knowledge representation:

**Foundation Models for Graphs**: Development of large pre-trained models for knowledge graphs, analogous to language model foundation models, could enable transfer learning across knowledge graph tasks [110]. GFM-RAG (Graph Foundation Model for RAG) represents early progress in this direction.

**Continuous Knowledge Evolution**: Methods for continuously updating knowledge representations while maintaining consistency present significant challenges [111]. Unlike static knowledge bases, real-world knowledge constantly evolves, requiring systems that can incrementally incorporate new information.

**Multi-Modal Integration**: Deeper integration of multi-modal information—text, images, video, and structured data—into unified knowledge representations remains challenging [112]. Current approaches often treat modalities separately, missing opportunities for cross-modal reasoning.

**Meta-Cognition and Self-Improvement**: Neuro-symbolic systems capable of reasoning about their own knowledge and limitations represent a frontier challenge [11]. Such systems could identify knowledge gaps, request relevant information, and improve their own reasoning capabilities.

---

## 8. Conclusion

This survey has examined the integration of multi-graph structures and ontologies as knowledge representation frameworks for AI systems. We have seen how multi-relational graphs—including heterogeneous information networks, temporal knowledge graphs, and multi-modal knowledge graphs—provide flexible structures for encoding complex real-world relationships. Ontologies complement these structures by providing formal semantic foundations through description logics, enabling consistency checking and logical inference.

The emergence of neuro-symbolic approaches represents a significant advancement in combining the strengths of neural learning with symbolic reasoning. Knowledge-enhanced language models, ontology-constrained graph neural networks, and hybrid reasoning systems demonstrate practical paths toward AI systems that are both data-efficient and capable of systematic reasoning. Applications across healthcare, scientific discovery, and enterprise knowledge management illustrate the real-world impact of these approaches.

Significant challenges remain in scalability, knowledge integration, reasoning under uncertainty, and explainability. Addressing these challenges requires continued interdisciplinary collaboration across artificial intelligence, database systems, logic, and domain sciences. Future directions including foundation models for graphs, continuous knowledge evolution, and meta-cognitive systems point toward increasingly sophisticated knowledge representation capabilities.

As AI systems become more prevalent in high-stakes applications, the importance of robust, explainable, and trustworthy knowledge representation will only grow. The integration of multi-graph and ontological approaches provides a promising foundation for building AI systems that can effectively leverage the breadth of human knowledge while maintaining the rigor required for reliable deployment.

---

## References

[1] Lecun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.

[2] Marcus, G. (2020). The next decade in AI: Four steps towards robust artificial intelligence. arXiv preprint arXiv:2002.06177.

[3] Garcez, A. d., & Lamb, L. C. (2023). Neurosymbolic AI: The 3rd wave. Artificial Intelligence Review, 56, 12387-12406.

[4] Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S. (2022). A survey on knowledge graphs: Representation, acquisition, and applications. IEEE Transactions on Neural Networks and Learning Systems, 33(2), 494-514.

[5] Hogan, A., Blomqvist, E., Cochez, M., et al. (2021). Knowledge graphs. ACM Computing Surveys, 54(4), 1-37.

[6] Gruber, T. R. (1993). A translation approach to portable ontology specifications. Knowledge Acquisition, 5(2), 199-220.

[7] Hitzler, P., Krötzsch, M., Parsia, B., Patel-Schneider, P. F., & Rudolph, S. (2012). OWL 2 web ontology language primer. W3C Recommendation, 27(1), 123.

[8] Brachman, R. J., & Levesque, H. J. (2004). Knowledge representation and reasoning. Morgan Kaufmann.

[9] Davis, R., Shrobe, H., & Szolovits, P. (1993). What is a knowledge representation? AI Magazine, 14(1), 17-33.

[10] Hinton, G. E. (1990). Mapping part-whole hierarchies into connectionist networks. Artificial Intelligence, 46(1-2), 47-75.

[11] Colelough, B., et al. (2025). Neuro-Symbolic AI in 2024: A systematic review. arXiv preprint arXiv:2501.05435.

[12] Bollacker, K., Evans, C., Paritosh, P., Sturge, T., & Taylor, J. (2008). Freebase: A collaboratively created graph database for structuring human knowledge. SIGMOD, 1247-1250.

[13] Suchanek, F. M., Kasneci, G., & Weikum, G. (2007). YAGO: A core of semantic knowledge. WWW, 697-706.

[14] Zhu, X., Li, Z., Wang, X., et al. (2024). Multi-modal knowledge graph construction and application: A survey. ACM Computing Surveys.

[15] Paulheim, H. (2017). Knowledge graph refinement: A survey of approaches and evaluation methods. Semantic Web, 8(3), 489-508.

[16] Staab, S., & Studer, R. (Eds.). (2009). Handbook on ontologies. Springer Science & Business Media.

[17] Baader, F., Calvanese, D., McGuinness, D. L., Nardi, D., & Patel-Schneider, P. F. (Eds.). (2003). The description logic handbook. Cambridge University Press.

[18] Glimm, B., Horrocks, I., Motik, B., Stoilos, G., & Wang, Z. (2014). HermiT: An OWL 2 reasoner. Journal of Automated Reasoning, 53(3), 245-269.

[19] Wang, Q., Mao, Z., Wang, B., & Guo, L. (2017). Knowledge graph embedding: A survey of approaches and applications. IEEE Transactions on Knowledge and Data Engineering, 29(12), 2724-2743.

[20] Keet, C. M. (2018). An introduction to ontology engineering. University of Cape Town.

[21] Fensel, D., Şimşek, U., Angele, K., et al. (2020). Knowledge graphs: Methodology, tools and selected use cases. Springer.

[22] Giglou, H. B., D'Souza, J., & Auer, S. (2024). A short review for ontology learning: Stride to large language models trend. arXiv preprint arXiv:2404.14991.

[23] Sun, Y., & Han, J. (2012). Mining heterogeneous information networks: Principles and methodologies. Morgan & Claypool Publishers.

[24] Wang, X., Bo, D., Shi, C., et al. (2022). A survey on heterogeneous graph neural networks: Methods, applications, and challenges. IEEE TKDE.

[25] Sun, Y., Han, J., Yan, X., Yu, P. S., & Wu, T. (2011). PathSim: Meta path-based top-k similarity search in heterogeneous information networks. VLDB, 4(11), 992-1003.

[26] Dong, Y., Chawla, N. V., & Swami, A. (2017). metapath2vec: Scalable representation learning for heterogeneous networks. KDD, 135-144.

[27] Liu, G., Zhang, Y., Li, Y., & Yao, Q. (2025). HHGT: Hierarchical heterogeneous graph transformer for heterogeneous graph representation learning. WSDM.

[28] Cai, L., et al. (2024). A survey on temporal knowledge graph: Representation learning and applications. arXiv preprint arXiv:2403.04782.

[29] Leblay, J., & Chekol, M. W. (2018). Deriving validity time in knowledge graph. WWW, 1771-1776.

[30] Jin, W., Qu, M., Jin, X., & Ren, X. (2020). Recurrent event network: Autoregressive structure inference over temporal knowledge graphs. EMNLP, 6669-6683.

[31] García-Durán, A., Dumančić, S., & Niepert, M. (2018). Learning sequence encoders for temporal knowledge graph completion. EMNLP, 4816-4821.

[32] Zhu, C., Chen, M., Fan, C., Cheng, G., & Zhang, Y. (2021). Learning from history: Modeling temporal knowledge graphs with distributed representations. AAAI, 4763-4770.

[33] Xu, D., Ruan, C., Korpeoglu, E., Kumar, S., & Achan, K. (2020). Inductive representation learning on temporal graphs. ICLR.

[34] Huang, S., et al. (2024). TGB 2.0: A benchmark for learning on temporal knowledge graphs and heterogeneous graphs. NeurIPS.

[35] Zhu, X., Li, Z., Wang, X., et al. (2024). A survey of multi-modal knowledge graphs: Technologies and trends. ACM Computing Surveys.

[36] Liu, Y., Li, H., Garcia-Duran, A., et al. (2019). MMKG: Multi-modal knowledge graphs. ESWC, 459-474.

[37] Chen, J., et al. (2025). Noise-powered multi-modal knowledge graph representation framework. COLING.

[38] Zhu, J., Zhou, Y., Qian, S., et al. (2025). Mosaic of modalities: A comprehensive benchmark for multimodal graph learning. CVPR.

[39] Wang, H., et al. (2024). Multi-modal knowledge graph completion: A survey. IEEE TKDE.

[40] Li, S., et al. (2024). MGIF: Global-aware convolution for multi-modal knowledge graph embedding. IJCAI.

[41] Cai, H., Zheng, V. W., & Chang, K. C. C. (2018). A comprehensive survey of graph embedding: Problems, techniques, and applications. IEEE TKDE, 30(9), 1616-1637.

[42] Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. NeurIPS, 2787-2795.

[43] Lin, Y., Liu, Z., Sun, M., Liu, Y., & Zhu, X. (2015). Learning entity and relation embeddings for knowledge graph completion. AAAI, 2181-2187.

[44] Nickel, M., Tresp, V., & Kriegel, H. P. (2011). A three-way model for collective learning on multi-relational data. ICML, 809-816.

[45] Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). Complex embeddings for simple link prediction. ICML, 2071-2080.

[46] Sun, Z., Deng, Z. H., Nie, J. Y., & Tang, J. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. ICLR.

[47] Zhang, S., Tay, Y., Yao, L., & Liu, Q. (2019). Quaternion knowledge graph embeddings. NeurIPS, 2731-2741.

[48] Li, Z., et al. (2024). Knowledge graph embeddings: A comprehensive survey on capturing relation properties. arXiv preprint arXiv:2410.14733.

[49] Motik, B., Grau, B. C., Horrocks, I., Wu, Z., Fokoue, A., & Lutz, C. (2012). OWL 2 web ontology language profiles. W3C Recommendation, 27, 61.

[50] Baader, F., Horrocks, I., Lutz, C., & Sattler, U. (2017). An introduction to description logic. Cambridge University Press.

[51] Horrocks, I., Kutz, O., & Sattler, U. (2006). The even more irresistible SROIQ. KR, 57-67.

[52] Ceylan, İ. İ., & Peñaloza, R. (2017). Bayesian description logics. AAAI, 1013-1019.

[53] Sirin, E., Parsia, B., Grau, B. C., Kalyanpur, A., & Katz, Y. (2007). Pellet: A practical OWL-DL reasoner. Web Semantics, 5(2), 51-53.

[54] Baader, F., Brandt, S., & Lutz, C. (2005). Pushing the EL envelope. IJCAI, 364-369.

[55] Horrocks, I., Patel-Schneider, P. F., Boley, H., Tabet, S., Grosof, B., & Dean, M. (2004). SWRL: A semantic web rule language combining OWL and RuleML. W3C Member Submission, 21, 79.

[56] He, Y., Chen, J., Dong, H., et al. (2024). DeepOnto: A Python package for ontology engineering with deep learning. Semantic Web Journal.

[57] Chen, J., He, Y., Geng, Y., Jiménez-Ruiz, E., Dong, H., & Horrocks, I. (2023). Contextual semantic embeddings for ontology alignment. ISWC, 172-189.

[58] Chen, J., Hu, P., Jimenez-Ruiz, E., Holter, O. M., Antoniazzi, D., & Horrocks, I. (2021). OWL2Vec*: Embedding of OWL ontologies. Machine Learning, 110(7), 1813-1845.

[59] Smaili, F. Z., Gao, X., & Hoehndorf, R. (2018). OPA2Vec: Combining formal and informal content of biomedical ontologies to improve similarity-based prediction. Bioinformatics, 35(12), 2133-2140.

[60] Kolyvakis, P., et al. (2024). OWL2Vec4OA: Tailoring knowledge graph embeddings for ontology alignment. arXiv preprint arXiv:2408.06310.

[61] Bourgeais, V., et al. (2024). Combining machine learning and ontology: A systematic literature review. HAL preprint hal-04373122.

[62] Mao, J., Gan, C., Kohli, P., Tenenbaum, J. B., & Wu, J. (2019). The neuro-symbolic concept learner. ICLR.

[63] Xu, J., Zhang, Z., Friedman, T., Liang, Y., & Van den Broeck, G. (2018). A semantic loss function for deep learning with symbolic knowledge. ICML, 5502-5511.

[64] Tian, Y., Pei, S., Zhang, X., Zhang, C., & Chawla, N. V. (2025). Knowledge distillation on graphs: A survey. ACM Computing Surveys.

[65] van Bekkum, M., de Boer, M., van Harmelen, F., Meyer-Vitali, A., & Teije, A. T. (2021). Modular design patterns for hybrid learning and reasoning systems. Applied Intelligence, 51(9), 6528-6546.

[66] DeLong, L. N., et al. (2024). Neurosymbolic AI for reasoning over knowledge graphs: A survey. IEEE TNNLS.

[67] Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., & Wu, X. (2024). Unifying large language models and knowledge graphs: A roadmap. IEEE TKDE.

[68] Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS, 9459-9474.

[69] Edge, D., et al. (2024). From local to global: A graph RAG approach to query-focused summarization. arXiv preprint arXiv:2404.16130.

[70] Mavromatis, C., & Karypis, G. (2024). GNN-RAG: Graph neural retrieval for large language model reasoning. ICLR.

[71] Guo, Z., et al. (2025). KGL-LLM: Knowledge graph language for precise LLM-KG integration. Frontiers in Computer Science.

[72] Sun, J., et al. (2024). Think-on-Graph 2.0: Deep and interpretable large language model reasoning with knowledge graph-guided retrieval. ICLR.

[73] Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2021). A comprehensive survey on graph neural networks. IEEE TNNLS, 32(1), 4-24.

[74] Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M. (2018). Modeling relational data with graph convolutional networks. ESWC, 593-607.

[75] Morris, P., et al. (2024). Relational graph convolutional networks do not learn sound rules. KR.

[76] Vashishth, S., Sanyal, S., Niber, V., & Talukdar, P. (2020). Composition-based multi-relational graph convolutional networks. ICLR.

[77] Zhang, Z., Cai, J., Zhang, Y., & Wang, J. (2020). Learning hierarchy-aware knowledge graph embeddings for link prediction. AAAI, 3065-3072.

[78] Ren, H., et al. (2020). Query2box: Reasoning over knowledge graphs in vector space using box embeddings. ICLR.

[79] Sadeghian, A., Armandpour, M., Ding, P., & Wang, D. Z. (2019). DRUM: End-to-end differentiable rule mining on knowledge graphs. NeurIPS, 15347-15357.

[80] Rosati, J., et al. (2025). Neuro-symbolic AI for conflict-aware learning over knowledge graphs. Springer LNCS.

[81] Bach, S. H., Broecheler, M., Huang, B., & Getoor, L. (2017). Hinge-loss Markov random fields and probabilistic soft logic. JMLR, 18(109), 1-67.

[82] Chandak, P., Huang, K., & Zitnik, M. (2024). Graph artificial intelligence in medicine. Annual Review of Biomedical Data Science.

[83] Chen, H., et al. (2025). DR.KNOWS: Leveraging medical knowledge graphs into large language models for diagnosis prediction. JMIR AI.

[84] Liu, Y., et al. (2024). A comprehensive large scale biomedical knowledge graph for AI powered data driven biomedical research. Scientific Data.

[85] Yang, J., et al. (2025). Biomedical knowledge graph: A survey of domains, tasks, and real-world applications. arXiv preprint arXiv:2501.11632.

[86] Guo, H., et al. (2025). Large language model-driven knowledge graph construction in sepsis care using multicenter clinical databases. JMIR.

[87] Dessì, D., Osborne, F., Reforgiato Recupero, D., Buscaldi, D., & Motta, E. (2022). AI-KG: An automatically generated knowledge graph of artificial intelligence. ISWC, 127-143.

[88] Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv preprint arXiv:2205.01833.

[89] Deng, J., et al. (2023). MatKG: A comprehensive materials science knowledge graph for machine learning. Matter, 6(3), 701-713.

[90] Chen, X., et al. (2025). Advancing engineering research through context-aware and knowledge graph-based retrieval-augmented generation. Frontiers in AI.

[91] Noy, N. F., Gao, Y., Jain, A., Narayanan, A., Patterson, A., & Taylor, J. (2019). Industry-scale knowledge graphs: Lessons and challenges. Communications of the ACM, 62(8), 36-43.

[92] Galkin, M., Trivedi, P., Maheshwari, G., Usbeck, R., & Lehmann, J. (2020). Message passing for hyper-relational knowledge graphs. EMNLP, 7346-7359.

[93] Ren, X., et al. (2023). A survey on knowledge graph-based conversational systems. TKDE.

[94] Hsueh, H. C., et al. (2024). A novel multi-level refined knowledge graph design and chatbot system for healthcare applications. JMIR Medical Informatics.

[95] Konidaris, G., & Barto, A. G. (2024). Knowledge representation for robots: A survey. Annual Review of Control, Robotics, and Autonomous Systems.

[96] Al-Rifai, M., et al. (2025). A comprehensive review of neuro-symbolic AI for robustness, uncertainty quantification, and intervenability. Arabian Journal for Science and Engineering.

[97] Hu, Y., et al. (2024). Autonomous driving knowledge graphs: A survey. IEEE Intelligent Transportation Systems Magazine.

[98] Liang, K., et al. (2024). A survey of knowledge graph reasoning on graph types: Static, dynamic, and multi-modal. IEEE TPAMI.

[99] Zeng, K., Yang, J., Wang, H., Shao, B., & Wang, Z. (2013). A distributed graph engine for web scale RDF data. VLDB, 6(4), 265-276.

[100] Horrocks, I., & Sattler, U. (2005). A tableaux decision procedure for SHOIQ. IJCAI, 448-453.

[101] Shvaiko, P., & Euzenat, J. (2013). Ontology matching: State of the art and future challenges. IEEE TKDE, 25(1), 158-176.

[102] Li, Z., et al. (2024). A survey on multi-view knowledge graph: Generation, fusion, applications. IJCAI.

[103] Lippolis, A., et al. (2025). LLM-empowered knowledge graph construction: A survey. arXiv preprint arXiv:2510.20345.

[104] Chen, X., Jia, S., & Xiang, Y. (2020). A review: Knowledge reasoning over knowledge graph. Expert Systems with Applications, 141, 112948.

[105] Kimmig, A., Bach, S., Broecheler, M., Huang, B., & Getoor, L. (2012). A short introduction to probabilistic soft logic. NIPS Workshop on Probabilistic Programming, 1-4.

[106] Lukasiewicz, T., & Straccia, U. (2008). Managing uncertainty and vagueness in description logics for the Semantic Web. Web Semantics, 6(4), 291-308.

[107] Gilpin, L. H., Bau, D., Yuan, B. Z., Bajwa, A., Specter, M., & Kagal, L. (2018). Explaining explanations: An overview of interpretability of machine learning. IEEE DSAA, 80-89.

[108] Zhao, X., et al. (2024). Explainable knowledge graph reasoning via deep neural networks. AAAI.

[109] Horridge, M., Parsia, B., & Sattler, U. (2010). Explaining inconsistencies in OWL ontologies. SUM, 124-137.

[110] Hu, Z., et al. (2025). GFM-RAG: Graph foundation model for retrieval augmented generation. arXiv preprint arXiv:2502.01113.

[111] Trivedi, R., Dai, H., Wang, Y., & Song, L. (2017). Know-evolve: Deep temporal reasoning for dynamic knowledge graphs. ICML, 3462-3471.

[112] Balažević, I., Allen, C., & Hospedales, T. M. (2019). Multi-relational poincaré graph embeddings. NeurIPS, 4463-4473.
