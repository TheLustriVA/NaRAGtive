# NaRAGtive TUI - Phase 1: Core Infrastructure

You have access to the TheLustriVA/NaRAGtive repository on GitHub.

## Assignment

Create Phase 1 of a Terminal User Interface (TUI) for NaRAGtive using the Python Textual framework. 
This phase establishes the foundation for a complete replacement of the existing CLI with a modern, 
keyboard-driven TUI.

## What You're Building

A multi-screen Textual application providing:
1. Dashboard home screen showing registered vector stores
2. Screen navigation framework for future search and ingest screens
3. Store list widget that displays stores with metadata
4. Global keybindings and responsive layout
5. TCSS styling for consistent appearance

## Minimum NaRAGtive Features to Integrate

- Read from `VectorStoreRegistry()` in `naragtive/store_registry.py`
- Load stores on app startup
- Display: store name, type, record count, creation date, default indicator (⭐)
- Show default store prominently

## Technical Requirements

**Framework & Version:**
- Textual >= 6.4.0, < 7.0 (stable as of Phase 1 development)
- Python 3.13+ (per NaRAGtive requirements)
- asyncio for any blocking operations

**Key Concepts (See Textual docs):**
- Use `@reactive` attributes for state management
- Multi-screen navigation with push/pop
- TCSS styling for consistency
- Pilot framework for testing

**Code Quality:**
- Type hints throughout (Python 3.13)
- Docstrings for all classes
- ≥80% test coverage
- No blocking I/O on main thread

## Files to Create

Code:
- `naragtive/tui/__init__.py`
- `naragtive/tui/app.py` - Main app class
- `naragtive/tui/screens/__init__.py`
- `naragtive/tui/screens/base.py` - Base screen class
- `naragtive/tui/screens/dashboard.py` - Home screen
- `naragtive/tui/widgets/__init__.py`
- `naragtive/tui/widgets/store_list.py` - Store display widget
- `naragtive/tui/styles/app.tcss` - TCSS stylesheet

Tests:
- `tests/test_tui_basic.py` - Navigation and display tests

## Keybindings (Minimum)

Global:
- ctrl+c, ctrl+d → Quit
- f1 → Help/Keybindings
- tab → Next widget, shift+tab → Previous

Dashboard:
- s → Search (pushes SearchScreen placeholder)
- i → Ingest (pushes IngestScreen placeholder)
- m → Manage Stores (pushes ManageStoresScreen placeholder)
- r → Refresh stores
- enter → Set selected store as default

## Success Criteria

✅ App runs: `python -m naragtive.tui.app`
✅ Loads stores from VectorStoreRegistry on startup
✅ Dashboard displays stores with all metadata
✅ Navigation between placeholder screens works
✅ Keybindings respond correctly
✅ Store list updates reactively when stores change
✅ All tests pass (≥80% coverage)
✅ Responsive to terminal size changes
✅ No breaking changes to existing CLI

## Reference Documentation

Textual official docs (use these for implementation):
- Getting started: https://textual.textualize.io/getting_started/
- Reactivity guide: https://textual.textualize.io/guide/reactivity/
- App API: https://textual.textualize.io/api/app/
- Screen API: https://textual.textualize.io/api/screen/
- Widget API: https://textual.textualize.io/api/widget/
- Testing: https://textual.textualize.io/guide/testing/
- TCSS: https://textual.textualize.io/guide/CSS/

Example Textual apps (for reference patterns):
- https://github.com/Textualize/textual/blob/main/examples/calculator.py
- https://github.com/Textualize/textual/blob/main/examples/clock.py

## Notes

- Don't modify existing NaRAGtive CLI code in main.py
- Integrate with naragtive module (read-only)
- Phase 1 is foundation only - no search/ingest logic yet
- Phase 2 will add search functionality
- Test with Pilot (Textual's testing framework)

## Deliverables

- Feature branch: feature/tui-phase1-core
- Pull request with all files listed above
- Updated pyproject.toml to add: textual>=6.4.0,<7.0
- Full test coverage (≥80%)
- PR description explaining Phase 1 scope
