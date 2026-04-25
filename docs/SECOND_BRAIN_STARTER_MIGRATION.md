# second-brain-starter migration note

This note explains how the new `prototype/second-brain-starter/` MVP relates to the existing root-level `rag-second-brain` codebase, and how to evolve the repository without breaking current work.

## Why this exists

The repository currently contains two different shapes of "RAG Second Brain":

1. **Existing root project (`main`)**
   - research / proof-of-concept oriented
   - centered on multi-retriever experiments
   - menu-driven and experiment-heavy
   - organized around statistical retrieval, dense retrieval, KG retrieval, ontology, and evaluation

2. **New MVP (`prototype/second-brain-starter/`)**
   - product-like, local-first operational shell
   - centered on ingest → extract → knowledge objects → review → trust-state → retrieval → answer
   - provenance-aware and governance-aware
   - includes lightweight browser UI for daily use

Both are valid, but they optimize for different jobs.

## Old vs new direction

### Existing root project
Best for:
- retrieval experiments
- paper support
- component benchmarking
- architecture exploration

Primary shape:
- `src/cooccurrence.py`
- `src/dense_retrieval.py`
- `src/kg_retrieval.py`
- `src/ontology.py`
- `src/pipeline.py`
- `tests/test_components.py`

### second-brain-starter MVP
Best for:
- real document ingestion
- provenance-preserving knowledge capture
- review workflow
- trust-state progression
- redundancy cleanup
- local operational usage via UI/CLI

Primary shape:
- `prototype/second-brain-starter/src/second_brain/ingest.py`
- `prototype/second-brain-starter/src/second_brain/extractors.py`
- `prototype/second-brain-starter/src/second_brain/review.py`
- `prototype/second-brain-starter/src/second_brain/status.py`
- `prototype/second-brain-starter/src/second_brain/redundancy.py`
- `prototype/second-brain-starter/src/second_brain/webapp.py`

## Recommended repository strategy

### Short term
Keep the MVP under `prototype/second-brain-starter/`.

Reason:
- avoids breaking the existing top-level research code
- makes review easier
- keeps the architectural shift explicit
- allows side-by-side comparison before committing to a repo-wide move

### Medium term
Promote shared ideas before promoting directories.

Recommended sequence:
1. stabilize the MVP contracts
2. decide which pieces are reusable at repo root
3. extract shared docs / schemas / fixtures
4. only then consider a root-level package migration

### Long term options

#### Option A — Dual-track repo
Keep both:
- root = research framework
- `prototype/second-brain-starter/` = operational product slice

Good when both research and product prototyping matter.

#### Option B — Root migration toward product shell
Adopt the MVP direction as the new primary structure.

Good when the goal is no longer retrieval experimentation alone, but a usable second-brain system.

Suggested staged path:
1. create a new top-level package namespace plan
2. merge CLI conventions
3. move docs / config conventions upward
4. introduce root-level `make` / script entrypoints for the MVP
5. migrate tests
6. deprecate or archive old demo entrypoints

#### Option C — Extract MVP into its own repository
Useful if the MVP starts moving faster than the research code.

Good when:
- different release cycles emerge
- product ergonomics diverge strongly from research code
- keeping one repo begins to slow both tracks down

## Suggested near-term actions

1. Keep PR #1 focused on landing the MVP cleanly.
2. Review whether the seeded knowledge examples should stay in-repo or move to fixtures/testdata.
3. Add a small comparison table to the main README so newcomers understand the two tracks.
4. If adoption continues, create a follow-up branch that adds root-level convenience entrypoints pointing into the MVP.

## My recommendation

Do **not** move the MVP to repo root yet.

Instead:
- merge the clean prototype branch first
- validate real use for a bit
- then promote conventions and entrypoints in a second pass

That is the lowest-risk path and keeps the repo honest about the fact that this is an architectural shift, not a tiny refactor.
