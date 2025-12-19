# NaRAGtive TUI - Phase 2: Search Functionality

You have access to the TheLustriVA/NaRAGtive repository on GitHub.

## Assignment

Create Phase 2 of the NaRAGtive Terminal User Interface (TUI). Phase 1 created the core infrastructure and dashboard. Phase 2 adds comprehensive search functionality, allowing users to query their vector stores with semantic search and optional reranking.

## What You're Building

A complete search workflow with:
1. **Search Input Screen** - Modal screen for entering search queries with history
2. **Results Table Widget** - Displays search results with relevance scores, sortable columns
3. **Result Detail Panel** - Full scene text display with metadata and character info
4. **Reranking Toggle** - Optional BGE reranking with progress indication
5. **Export Functionality** - Export results to JSON/CSV formats
6. **Non-blocking Async Search** - Searches happen without freezing the UI
7. **Search History** - Recent searches available for quick re-running

## Integration Points with Existing Code

### PolarsVectorStore (naragtive/polars_vectorstore.py)
Use the `query()` method:
```python
store = PolarsVectorStore("./scenes.parquet")
store.load()
results = store.query("Admiral leadership", n_results=20)
# Returns: {"ids": [...], "documents": [...], "scores": [...], "metadatas": [...]}
```

### BGE Reranker Integration (naragtive/bge_reranker_integration.py)
Use for reranking results:
```python
from naragtive.bge_reranker_integration import BGEReranker

reranker = BGEReranker(model_name="BAAI/bge-reranker-base")
reranked = reranker.rerank(
    query="Admiral leadership",
    documents=results["documents"],
    top_k=10
)
# Returns: [{"index": N, "score": 0.95, "text": "..."}]
```

### VectorStoreRegistry (naragtive/store_registry.py)
Get current store:
```python
from naragtive.store_registry import VectorStoreRegistry

registry = VectorStoreRegistry()
store = registry.get("store_name")  # Returns PolarsVectorStore instance
```

## Technical Requirements

**Framework & Version:**
- Textual >= 6.4.0, < 7.0
- Python 3.13+
- asyncio for non-blocking search
- numpy for reranking score handling

**Key Textual Concepts:**
- Use DataTable widget for results display
- Async operations with `run_in_executor()` for search/reranking
- Reactive attributes for search state (query, results, reranking_enabled, etc.)
- Modal screens for search input
- Background tasks for long-running operations

**Code Quality:**
- Type hints throughout (Python 3.13)
- Docstrings for all classes and methods
- ≥80% test coverage
- No blocking I/O on main thread
- Proper error handling for search failures

## Files to Create/Modify

### New Files (Create)

**Screens:**
- `naragtive/tui/screens/search.py` - Main search screen
  - Composed of: SearchInputWidget + ResultsTableWidget + ResultDetailWidget
  - Manages search workflow, reranking toggle, export
  - Keybindings: /query, r=rerank, e=export, esc=back

**Widgets:**
- `naragtive/tui/widgets/search_input.py` - Search query input modal
  - Text input with autocomplete from history
  - Keybindings: enter=search, esc=cancel, up/down=history
  - Emits: `SearchRequested` event with query text

- `naragtive/tui/widgets/results_table.py` - Results display DataTable
  - Columns: "Relevance", "ID", "Scene Date", "Location", "POV"
  - Rows: one per search result, sorted by relevance descending
  - Keybindings: enter=view_detail, r=rerank_this, s=sort_by
  - Emits: `ResultSelected` event when row clicked

- `naragtive/tui/widgets/result_detail.py` - Full result detail display
  - Shows: full text, metadata (date, location, characters, POV)
  - Scrollable text area
  - Keybindings: q=close, y=copy_id, c=copy_text

**Utilities:**
- `naragtive/tui/search_utils.py` - Helper functions
  - `async_search()` - Async wrapper for store.query()
  - `async_rerank()` - Async wrapper for BGE reranker
  - `format_result_score()` - Format relevance score as percentage
  - `format_metadata()` - Parse and format metadata for display

**Tests:**
- `tests/test_tui_search.py` - Search screen tests
  - Test search query input
  - Test async search with mocked store
  - Test result display and sorting
  - Test reranking workflow
  - Test export functionality
  - ≥80% coverage

### Modified Files

**`naragtive/tui/app.py`**
- Add reactive properties: `current_query`, `search_results`, `reranking_enabled`
- Add `on_search_requested()` message handler
- Add async search method that shows progress indicator
- Keep dashboard navigation intact

**`naragtive/tui/screens/dashboard.py`**
- Add "s" keybinding to push SearchScreen
- Existing store list and management unchanged

**`naragtive/tui/styles/app.tcss`**
- Add styles for DataTable rows, header
- Add styles for result detail panel (scrollable text)
- Add styles for search input modal
- Add progress indicator styles (for reranking)

## Data Flow

```
User presses 's' on Dashboard
    ↓
Push SearchScreen
    ↓
User enters query in SearchInputWidget
    ↓
SearchRequested event emitted
    ↓
App.on_search_requested() runs async_search()
    ↓
Search runs in executor (non-blocking)
    ↓
Results displayed in ResultsTableWidget
    ↓
User can:
  - Press Enter to see full detail
  - Press 'r' to rerank (if BGE available)
  - Press 'e' to export results
  - Press 's' to sort by different column
```

## Keybindings (Phase 2)

**Search Screen (Main):**
- `/` → Open search input modal
- `r` → Toggle reranking (on/off)
- `e` → Export results to JSON
- `s` → Sort results by column (cycle: relevance → date → id)
- `enter` → View selected result detail
- `esc` → Back to dashboard

**Search Input Modal:**
- `enter` → Execute search
- `esc` → Cancel and close modal
- `up/down` → Navigate search history
- `ctrl+u` → Clear input

**Result Detail Panel:**
- `q` → Close and return to results
- `y` → Copy scene ID to clipboard
- `c` → Copy full text to clipboard
- `↑/↓` → Scroll text

## Success Criteria

✅ Search input modal works with history
✅ Query text sent to PolarsVectorStore.query()
✅ Results displayed in sorted DataTable
✅ Relevance scores shown as percentages (e.g., "94%")
✅ Can click/enter to view full detail
✅ Reranking toggle works (with progress indicator during reranking)
✅ Export to JSON shows results with metadata
✅ All searches are async (UI never freezes)
✅ Error handling for:
  - Store not loaded
  - Search query too short
  - Reranker not available
  - No results found
✅ All tests pass (≥80% coverage)
✅ No breaking changes to Phase 1 code
✅ Responsive to terminal size changes

## Result Data Structure

From `PolarsVectorStore.query()`, you'll receive:
```python
results = {
    "ids": ["scene-123", "scene-456"],
    "documents": ["Full scene text...", "Another scene..."],
    "scores": [0.94, 0.87],  # Already 0.0-1.0 range
    "metadatas": [
        {
            "scene_id": "scene-123",
            "date_iso": "2024-01-15",
            "location": "Throne Room",
            "pov_character": "Admiral Zelenskyy",
            "characters_present": '["Admiral Z", "King"]'
        },
        # ...
    ]
}
```

**Important:** Parse `characters_present` as JSON string before display.

## Reranker Integration Notes

- BGEReranker is optional (if not available, just show message)
- Reranking is slow (~5-10 seconds for 20 results)
- Use progress modal during reranking
- Reranker returns list of dicts with "score" and "text" fields
- Re-sort results by reranker scores after reranking completes

## Error Handling

Handle these cases gracefully:
1. **Store not loaded** → Show error, offer to go back
2. **Query too short** → "Query must be at least 3 characters"
3. **No results found** → "No scenes match your query"
4. **Search timeout** → "Search took too long, try more specific query"
5. **Reranker not available** → "BGE reranker not installed. Continue with original results?"
6. **Export fails** → Show error message, suggest debugging

## Testing Strategy

**Unit Tests:**
- Test search_utils functions with mocked store
- Test metadata parsing
- Test score formatting

**Integration Tests:**
- Test SearchScreen with mock PolarsVectorStore
- Test async search doesn't freeze UI
- Test reranking workflow
- Test DataTable update with new results

**Manual Tests:**
- Run search with 0 results
- Run search with 1000+ results (pagination?)
- Rerank results
- Export to file and verify JSON
- Use search history

## Reference Documentation

Textual:
- DataTable: https://textual.textualize.io/widgets/data_table/
- Input widget: https://textual.textualize.io/widgets/input/
- Static (for detail panel): https://textual.textualize.io/widgets/static/
- Async operations: https://textual.textualize.io/guide/events/#async-handlers
- Progress indicator patterns: https://github.com/Textualize/textual/blob/main/examples/progress.py

NaRAGtive APIs:
- PolarsVectorStore.query(): See naragtive/polars_vectorstore.py
- BGEReranker: See naragtive/bge_reranker_integration.py
- VectorStoreRegistry: See naragtive/store_registry.py

## Deliverables

- Feature branch: `feature/tui-phase2-search`
- Pull request with all files listed above
- Test file with ≥80% coverage
- No modifications to Phase 1 code
- PR description explaining:
  - Search workflow
  - Reranking integration
  - Async operation patterns used
  - Known limitations or TODOs for Phase 3

## Notes

- Phase 1 dashboard must remain fully functional
- Don't modify existing CLI code
- Search and reranking run asynchronously (no UI blocking)
- Phase 3 will add ingest functionality
- Consider pagination if result sets are very large (1000+ results)

## Success Indicators

When complete, you should be able to:
1. Run the TUI app
2. See dashboard with stores
3. Press 's' to go to search
4. Type a query and press enter
5. See results in a table within 2-3 seconds
6. Press enter on a result to see full text
7. Toggle reranking with 'r' and see scores update
8. Export results with 'e' and verify JSON file
9. Go back to dashboard with 'esc'
10. All tests pass without errors
