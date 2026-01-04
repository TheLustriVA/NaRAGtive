# Perplexity Labs: NaRAGtive TUI Gap Analysis & Phase 3+ Roadmap

You have access to the TheLustriVA/NaRAGtive repository on GitHub.

## Your Task

Analyze the NaRAGtive repository and produce two deliverables:

### Deliverable 1: Feature Gap Analysis Document
Create a comprehensive document comparing:
1. **Current CLI/Core Functionality** — What NaRAGtive can do via command-line (from main.py and README)
2. **Phase 1 TUI Capabilities** — What the TUI Phase 1 adds (from docs/TEXTUAL_LABS_PROMPT.md)
3. **Phase 2 TUI Capabilities** — What the TUI Phase 2 adds (from docs/TEXTUAL_PHASE2_PROMPT.md)
4. **Functionality Gap Analysis** — Features in CLI not yet in TUI
5. **Feature Priority Matrix** — Which gaps should be Phase 3, Phase 4, etc. (by user value and complexity)
6. **Recommended Phase 3 Scope** — What features should Phase 3 implement

### Deliverable 2: Phase 3 Specification (if applicable)
If Phase 3 scope is clear from your analysis, generate a `TEXTUAL_PHASE3_PROMPT.md` file following the same format as:
- docs/TEXTUAL_LABS_PROMPT.md (Phase 1)
- docs/TEXTUAL_PHASE2_PROMPT.md (Phase 2)

Format should include:
- Assignment overview
- What you're building
- Integration points with existing code
- Technical requirements
- Files to create/modify
- Keybindings
- Success criteria
- Testing strategy
- Deliverables

---

## Context: NaRAGtive Architecture

NaRAGtive is a **Narrative RAG (Retrieval-Augmented Generation) system** for fiction projects.

### Current CLI Commands (from main.py)

**Ingestion:**
- `ingest-neptune` — Ingest Neptune AI RPG narrative exports
- `ingest-llama` — Ingest llama-server chat exports
- `ingest-chat` — Ingest generic chat transcripts (JSON/text)
- `ingest-help` — Show detailed ingestion workflows

**Search & Query:**
- `search` — Semantic search with optional BGE reranking
- `interactive` — Interactive search mode with model caching
- `list` — List scenes by metadata filters (character, location, date)
- `stats` — Show vector store statistics
- `export` — Export search results in multiple formats (bge, llm-context, llamafile, jsonl, rag)

**Store Management:**
- `list-stores` — List all registered vector stores
- `create-store` — Register a new named store
- `set-default` — Set default store for commands
- `migrate` — (Legacy) Migrate from ChromaDB to Polars

### Core Technologies
- **Storage**: Polars parquet files + VectorStoreRegistry
- **Embeddings**: all-MiniLM-L6-v2 (384-dim)
- **Reranking**: BGE v2 M3 (optional)
- **TUI Framework**: Textual 6.4.0+

### Current TUI Phases (Documented)

**Phase 1 (Core Infrastructure):**
- Dashboard showing registered vector stores
- Store list widget with metadata display
- Navigation framework for future screens
- Global keybindings (ctrl+c/d quit, f1 help)
- Dashboard keybindings: s=search, i=ingest, m=manage, r=refresh, enter=set default

**Phase 2 (Search Functionality):**
- Search input modal with query history
- Results table (DataTable) with sorting and relevance scores
- Full result detail view with scene text and metadata
- Optional BGE reranking with progress indicator
- Export results to JSON/CSV
- All async (non-blocking UI)

---

## Analysis Questions to Answer

As you analyze the repository, address these specifically:

### 1. CLI Feature Inventory
**What can users do via CLI that they CANNOT do yet in TUI?**
- Ingest narratives (Neptune, llama-server, generic chat)?
- Manage stores (create, register, set default)?
- Filter scenes by metadata (character, location, date)?
- View statistics and store info?
- Export in multiple formats?
- Use interactive mode with model caching?

### 2. Phase 1+2 Coverage
**Of the CLI features, which are already covered in Phases 1-2?**
- Phase 1: Dashboard with store list (partial - no store creation/management)
- Phase 2: Search and reranking (yes, but what about export? metadata filtering?)
- What's missing from each phase's spec?

### 3. Obvious Gaps
**Which CLI features are completely missing from TUI?**
- Create/register new stores from TUI?
- Set default store from TUI?
- Ingest data from TUI (currently CLI-only)?
- Filter by metadata (character, location, date) in TUI?
- Export results from TUI?
- View statistics from TUI?
- Interactive multi-query mode in TUI?

### 4. Priority Assessment
**Which gaps are highest priority?**
For each missing feature, consider:
- **User Value**: How important is this to daily workflow?
- **Frequency**: How often would users access this?
- **Complexity**: How hard is it to implement?
- **Dependencies**: Does this block other features?

### 5. Phase 3+ Recommendations
**Create a phased roadmap:**
- Which features should go in Phase 3 (next immediate)?
- Which should be Phase 4 (later)?
- Which are nice-to-have vs. critical for feature parity?

---

## Files to Review

When analyzing, refer to:

1. **main.py** — CLI command definitions and handlers
2. **README.md** — Feature overview and examples
3. **docs/TEXTUAL_LABS_PROMPT.md** — Phase 1 spec (Core Infrastructure)
4. **docs/TEXTUAL_PHASE2_PROMPT.md** — Phase 2 spec (Search Functionality)
5. **naragtive/store_registry.py** — Store management API
6. **naragtive/polars_vectorstore.py** — Search API
7. **naragtive/bge_reranker_integration.py** — Reranking API
8. **naragtive/ingest_chat_transcripts.py** — Ingestion API

---

## Output Format

### Part 1: Feature Gap Analysis Document

Structure as:

```markdown
# NaRAGtive TUI Feature Gap Analysis

## Executive Summary
[Brief overview of coverage, gaps, recommendations]

## 1. Current CLI Functionality
[Organized by feature category]

## 2. Phase 1 TUI Coverage
[What Phase 1 implements, what it doesn't]

## 3. Phase 2 TUI Coverage
[What Phase 2 implements, what it doesn't]

## 4. Functionality Gap Matrix
[Table: Feature | CLI Available | Phase 1 | Phase 2 | Missing From TUI]

## 5. Priority Assessment
[Features ranked by user value + complexity]

## 6. Phase 3 Scope Recommendation
[What should Phase 3 implement and why]

## 7. Phase 4+ Roadmap
[Future phases with estimated effort]

## Appendix: Feature Comparison Table
[Detailed feature-by-feature breakdown]
```

### Part 2: Phase 3 Specification (if applicable)

Follow the same format as TEXTUAL_PHASE2_PROMPT.md with sections:
- Assignment
- What You're Building
- Integration Points
- Technical Requirements
- Files to Create/Modify
- Keybindings
- Success Criteria
- Testing Strategy
- Deliverables

---

## Success Criteria

Your analysis should:

✅ Identify ALL CLI features from main.py
✅ Map each feature to Phase 1, 2, or "missing"
✅ Explain WHY features are missing (design choice or incomplete)
✅ Prioritize gaps by importance to users
✅ Recommend specific Phase 3 scope
✅ Provide Phase 3 specification (if it's clear what should go there)
✅ Include estimated effort/timeline for each phase
✅ Consider integration dependencies
✅ Flag any architectural blockers

---

## Notes

- The TUI is a **progressive replacement** of the CLI, not a simple duplication of CLI functionality. It reimagines user workflows with better UX (interactive, visual, real-time feedback) rather than just recreating CLI commands as buttons. Prioritize workflows over command parity.
- Phase 1-2 are already documented; your job is to **plan the rest**
- Focus on **user workflows** (what do users need to accomplish?)
- Consider **UI patterns** from Textual (screens, widgets, async patterns)
- Maintain **feature parity** with CLI as end-goal
- Account for **TUI-specific advantages** (interactive, visual, keyboard-driven)

---

## Questions to Address in Your Analysis

Before creating Phase 3 spec, answer:

1. **Is ingest critical for Phase 3?** (or Phase 4?)
2. **Should store management be before search?** (current order: Phase 1 dashboard, Phase 2 search)
3. **What's the minimum viable TUI?** (just search, or search + ingest + management?)
4. **Can Phase 3 be non-blocking from Phase 2?** (or does it need Phase 2 complete first?)
5. **What metadata filtering is most important?** (all three? or prioritize?)
6. **Is export critical?** (already in CLI, but is it needed in TUI?)

---

## Timeline Expectation

- **Gap Analysis**: 10-15 minutes
- **Phase 3 Draft Spec**: 10-15 minutes if scope is obvious
- **Total**: 20-30 minutes

---

Good luck! The NaRAGtive codebase is clean and well-documented, which should make analysis straightforward.
