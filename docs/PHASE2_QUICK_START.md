# Phase 2 Quick Start - Send to Perplexity Labs

## What You're Sending

Phase 2 adds complete search functionality to the NaRAGtive TUI, including:
- Search input modal with query history
- Results table with sortable columns and relevance scores
- Full result detail view with scene text and metadata
- Optional BGE reranking support
- Export results to JSON
- All async (UI never freezes)

## Files to Attach to Labs

1. `docs/TEXTUAL_PHASE2_PROMPT.md` - Complete Phase 2 specification
2. `naragtive/tui/TEXTUAL_REFERENCE.md` - Textual framework reference (from Phase 1)
3. `docs/TEXTUAL_GETTING_STARTED.md` - How to run tests (from Phase 1)
4. `README.md` - Project overview
5. `pyproject.toml` - Dependencies

## What to Paste Into Labs

Copy the entire content of `docs/TEXTUAL_PHASE2_PROMPT.md` and paste into Perplexity Labs chat.

## Timeline Expectations

- **0:00-0:30** Setup and creating widget files
- **0:30-1:30** Implementing SearchScreen and widgets
- **~1:00** First approval popup (create branch)
- **1:30-2:30** Async search and reranking integration
- **~2:00** Second approval popup (create PR)
- **2:30-3:30** Tests and error handling
- **~3:00** Final approval popup (push/ready)
- **3:30+** Final polish and completion

**Total: 3.5-4 hours**

## Files Labs Will Create

**New files:**
- `naragtive/tui/screens/search.py`
- `naragtive/tui/widgets/search_input.py`
- `naragtive/tui/widgets/results_table.py`
- `naragtive/tui/widgets/result_detail.py`
- `naragtive/tui/search_utils.py`
- `tests/test_tui_search.py`

**Modified files:**
- `naragtive/tui/app.py` (add reactive properties, search handlers)
- `naragtive/tui/screens/dashboard.py` (add search keybinding)
- `naragtive/tui/styles/app.tcss` (add search screen styles)

## Keybindings for Testing

After Phase 2 completes:

```bash
python -m naragtive.tui.app
```

- Dashboard:
  - `s` → Go to search
  - `i` → Ingest (from Phase 1)
  - `m` → Manage stores (from Phase 1)
- Search screen:
  - `/` → Open search input
  - Type query, press `enter` to search
  - Press `enter` on result to see details
  - `r` → Toggle reranking
  - `e` → Export results
  - `esc` → Back to dashboard

## How to Know It's Done

- ✅ Search button on dashboard works
- ✅ Can enter query and see results in 2-3 seconds
- ✅ Results show relevance percentage
- ✅ Can view full scene text by pressing enter
- ✅ Can export results to JSON file
- ✅ Can toggle reranking (if BGE installed)
- ✅ All tests pass (`pytest tests/test_tui_search.py`)
- ✅ No UI freezing during search
- ✅ Dashboard still works from Phase 1

## Integration with Phase 1

Phase 2 doesn't break Phase 1:
- Dashboard still shows stores
- Can still set default store
- Existing keybindings unchanged
- Can still go back to dashboard from search

## Testing Before/After

**Before sending to Labs:**
```bash
# Make sure Phase 1 tests still pass
pytest tests/test_tui_basic.py -v
```

**After Labs completes:**
```bash
# Run all TUI tests
pytest tests/test_tui_basic.py tests/test_tui_search.py -v

# Run the app
python -m naragtive.tui.app
```

## Common Issues & Fixes

**"Store not loaded" error**
- Make sure you have registered a store first
- In dashboard, press `i` to ingest data
- Then go back and try search

**Search returns no results**
- Try different search terms
- Make sure your store has data
- Queries must be at least 3 characters

**Reranking fails**
- BGE model might not be installed
- Labs will handle this gracefully with error message
- Original results will still show

**Results table shows nothing**
- Search might still be running (takes 2-5 seconds)
- Check terminal for any error messages
- Try simpler query term

## Next: Phase 3

After Phase 2 is complete and merged, Phase 3 will add:
- Ingest wizard screen
- File selection dialog
- Progress tracking during ingestion
- Store registration from TUI

Same process: create Phase 3 prompt and send to Labs.

## Questions?

Refer to:
- `docs/TEXTUAL_PHASE2_PROMPT.md` - Complete technical spec
- `docs/LABS_APPROVAL_MONITORING.md` - How to watch the process
- `naragtive/polars_vectorstore.py` - Search API reference
- `naragtive/bge_reranker_integration.py` - Reranking API reference
