"""20 AI/ML paper abstracts for seeding the Second Brain demo."""

PAPERS = [
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "abstract": "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) models that combine pre-trained parametric and non-parametric memory for language generation. We build RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline (BART). RAG models achieve state-of-the-art results on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures.",
        "authors": "Lewis et al.",
        "year": 2020
    },
    {
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "abstract": "Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. We show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets, our dense retriever outperforms a strong Lucene-BM25 system greatly by 9%-19% absolute in terms of top-20 passage retrieval accuracy.",
        "authors": "Karpukhin et al.",
        "year": 2020
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. The pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference.",
        "authors": "Devlin et al.",
        "year": 2019
    },
    {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality.",
        "authors": "Vaswani et al.",
        "year": 2017
    },
    {
        "title": "A Survey on Knowledge Graphs: Representation, Acquisition, and Applications",
        "abstract": "Human knowledge provides a formal understanding of the world. Knowledge graphs that represent structural relations between entities have become an increasingly popular research direction towards cognition and human-level intelligence. In this survey, we provide a comprehensive review of knowledge graph covering knowledge graph representation learning, knowledge acquisition and completion, temporal knowledge graphs, and knowledge-aware applications. We propose a full-view categorization and new taxonomies on these topics.",
        "authors": "Hogan et al.",
        "year": 2021
    },
    {
        "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "abstract": "The use of retrieval-augmented generation (RAG) to retrieve relevant information from an external knowledge source enables large language models (LLMs) to answer questions over private and/or previously unseen document collections. However, RAG fails on global questions directed at an entire text corpus, such as 'What are the main themes in the dataset?' We propose Graph RAG, which uses an LLM-derived knowledge graph to generate community summaries, enabling both local and global query-focused summarization.",
        "authors": "Edge et al.",
        "year": 2024
    },
    {
        "title": "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering",
        "abstract": "Existing question answering (QA) datasets fail to train QA systems to perform complex reasoning and provide explanations for answers. We introduce HotpotQA, a new dataset with 113k Wikipedia-based question-answer pairs that requires reasoning over multiple supporting documents to answer. It features natural, multi-hop questions, strong supervision for supporting facts, and a new type of factoid comparison questions to test systems' ability to extract relevant information.",
        "authors": "Yang et al.",
        "year": 2018
    },
    {
        "title": "MuSiQue: Multihop Questions via Single-hop Question Composition",
        "abstract": "Multi-hop reasoning remains a challenging task. We introduce MuSiQue, a new multihop QA dataset with 25K questions that are difficult to answer via single-hop reasoning. We construct MuSiQue by composing single-hop questions with a novel method that ensures compositionality and connected reasoning. Our dataset has a unique property: each multi-hop question is paired with a set of single-hop sub-questions, enabling detailed analysis of model reasoning capabilities across 2-hop, 3-hop, and 4-hop questions.",
        "authors": "Trivedi et al.",
        "year": 2022
    },
    {
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "abstract": "Despite their remarkable capabilities, large language models often produce responses containing factual inaccuracies due to their sole reliance on the parametric knowledge they encapsulate. We introduce Self-RAG, a new framework that trains a single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special reflection tokens. Self-RAG significantly outperforms state-of-the-art LLMs and retrieval-augmented models.",
        "authors": "Asai et al.",
        "year": 2024
    },
    {
        "title": "The Probabilistic Relevance Framework: BM25 and Beyond",
        "abstract": "The probabilistic relevance framework (PRF) is a formal framework for document retrieval that provides the theoretical basis for the widely used BM25 ranking function. BM25 and its variants represent a family of highly effective ranking functions used in text retrieval. We provide a comprehensive overview of the PRF, including its theoretical foundations, extensions to handle document structure, and connections to language modeling and information theory.",
        "authors": "Robertson and Zaragoza",
        "year": 2009
    },
    {
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "abstract": "BERT and RoBERTa has set a new state-of-the-art performance on sentence-pair regression tasks like semantic textual similarity. However, it requires that both sentences are fed into the network, which causes a massive computational overhead. We present Sentence-BERT (SBERT), a modification of the pretrained BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity.",
        "authors": "Reimers and Gurevych",
        "year": 2019
    },
    {
        "title": "A Survey on Named Entity Recognition: Methods and Applications",
        "abstract": "Named Entity Recognition (NER) is a fundamental task in natural language processing that seeks to locate and classify named entities in text into predefined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, and percentages. This survey covers traditional approaches including CRF and BiLSTM models, as well as modern transformer-based methods. We discuss applications in knowledge graph construction, information extraction, and question answering systems.",
        "authors": "Li et al.",
        "year": 2022
    },
    {
        "title": "Neural Word Embedding as Implicit Matrix Factorization",
        "abstract": "We analyze skip-gram with negative-sampling (SGNS) word embedding and show that it is implicitly factorizing a word-context matrix whose cells are the pointwise mutual information (PMI) of the respective word and context pairs, shifted by a global constant. We draw connections between the SGNS embedding and the traditional distributional approach to semantics based on PMI and positive PMI (PPMI) matrices, showing that these classic methods remain competitive with modern neural approaches.",
        "authors": "Levy and Goldberg",
        "year": 2014
    },
    {
        "title": "OWL 2 Web Ontology Language: Structural Specification and Functional-Style Syntax",
        "abstract": "The OWL 2 Web Ontology Language is designed to formalize domain knowledge through class hierarchies, property characteristics, and logical axioms. OWL 2 supports ontological reasoning through description logic, enabling automated inference of implicit knowledge from explicitly stated facts. Key features include transitive properties, inverse properties, class subsumption, and domain/range constraints. Materialization through forward-chaining enables efficient query-time reasoning over knowledge graphs.",
        "authors": "Motik et al.",
        "year": 2012
    },
    {
        "title": "Building a Second Brain: A Proven Method to Organize Your Digital Life",
        "abstract": "The concept of a Second Brain refers to a personal knowledge management system that externalizes and augments human cognition. By systematically capturing, organizing, distilling, and expressing information, individuals can leverage their accumulated knowledge more effectively. This approach draws on cognitive science principles of external cognition, distributed intelligence, and the extended mind thesis. Digital tools enable building interconnected knowledge networks that support creative thinking and decision-making.",
        "authors": "Forte",
        "year": 2022
    },
    {
        "title": "Billion-scale Similarity Search with GPUs using FAISS",
        "abstract": "We present FAISS, a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. FAISS supports multiple index types including flat indexes, inverted file indexes (IVF), and product quantization. The library achieves state-of-the-art performance for billion-scale nearest neighbor search, making it practical to deploy dense retrieval systems at scale.",
        "authors": "Johnson et al.",
        "year": 2019
    },
    {
        "title": "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer",
        "abstract": "The capacity of a neural network to absorb information is limited by its number of parameters. We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. The MoE enables conditional computation, where different parts of the network are activated for different inputs, achieving significant capacity gains with manageable computational costs.",
        "authors": "Shazeer et al.",
        "year": 2017
    },
    {
        "title": "Cross-Attention in Transformer Architecture for Multi-Modal Fusion",
        "abstract": "Cross-attention mechanisms enable transformer models to attend to information from different modalities or sources simultaneously. In multi-modal and multi-source settings, cross-attention computes query-key-value interactions across source boundaries, allowing the model to selectively focus on relevant information from each source. This mechanism has proven effective in document retrieval re-ranking, where candidate documents from multiple retrieval sources can be jointly evaluated through cross-attention fusion layers.",
        "authors": "Tsai et al.",
        "year": 2019
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "abstract": "We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning. We show that chain-of-thought prompting is an emergent ability of model scale. Experiments on arithmetic, commonsense, and symbolic reasoning benchmarks demonstrate that chain-of-thought reasoning enables multi-step problem decomposition, leading to substantial performance gains on tasks requiring sequential reasoning.",
        "authors": "Wei et al.",
        "year": 2022
    },
    {
        "title": "Survey of Hallucination in Natural Language Generation: Grounding and Retrieval",
        "abstract": "Neural language models are prone to hallucinate text that is fluent but factually incorrect. Retrieval-augmented generation (RAG) has emerged as a key approach to reduce hallucination by grounding generation in retrieved evidence. This survey examines hallucination across different NLG tasks, categorizing types of hallucination and evaluating mitigation strategies. We find that multi-source retrieval with diverse evidence types provides stronger grounding than single-source approaches, reducing hallucination rates by 15-30%.",
        "authors": "Ji et al.",
        "year": 2023
    },
]
