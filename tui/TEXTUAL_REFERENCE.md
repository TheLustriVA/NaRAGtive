# Textual Framework Quick Reference for NaRAGtive TUI

## Essential APIs for This Project

### App Structure

```python
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer

class NaRAGtiveApp(App):
    TITLE = "NaRAGtive"
    CSS = "..."  # TCSS stylesheet
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
```

Reactive Attributes (State Management)

```python
from textual.reactive import reactive
from typing import List
from naragtive.store_registry import StoreMetadata

class NaRAGtiveApp(App):
    stores: reactive[List[StoreMetadata]] = reactive([], recompose=True)
    current_store: reactive[str | None] = reactive(None)
    
    def watch_stores(self, new_stores: List[StoreMetadata]):
        # Called automatically when stores changes
        # Update UI here
        pass
```

Screen Navigation

```python
class DashboardScreen(Screen):
    def action_go_search(self):
        self.app.push_screen(SearchScreen())

class SearchScreen(Screen):
    def action_back(self):
        self.app.pop_screen()
```

Async Operations (Non-blocking)

```python
async def async_operation():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, blocking_function, args)
    return result
```

Testing with Pilot

```python
async def test_search():
    app = MyApp()
    async with app.run_test() as pilot:
        # Simulate user actions
        await pilot.click("Button#search")
        # Assert results
        assert app.search_results is not None
```

Key Textual Concepts

- Reactive: Use @reactive for automatic UI updates on data change
- Screens: Each workflow (search, ingest) is a separate Screen
- Widgets: Compose UI from buttons, inputs, tables, etc.
- CSS/TCSS: Style with CSS-like syntax (file extension: .tcss)
- Events: React to user input with on_* handlers
- Testing: Use Pilot to test interactions without terminal

Links to Official Docs

- Reactivity: [https://textual.textualize.io/guide/reactivity/](https://textual.textualize.io/guide/reactivity/)
- Screens: [https://textual.textualize.io/guide/screens/](https://textual.textualize.io/guide/screens/)
- Testing: [https://textual.textualize.io/guide/testing/](https://textual.textualize.io/guide/testing/)
- All Widgets: [https://textual.textualize.io/widgets/](https://textual.textualize.io/widgets/)
- CSS Guide: [https://textual.textualize.io/guide/CSS/](https://textual.textualize.io/guide/CSS/)
