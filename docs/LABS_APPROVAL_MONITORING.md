# ⚠️ CRITICAL: Monitoring Perplexity Labs GitHub Approvals

## Important: You MUST Watch the Entire Process

Perplexity Labs will pop up GitHub approval requests during the TUI implementation. **If you don't approve within the timeout window, the entire process crashes and leaves your repo in a half-changed state.**

### What Happens

1. Labs starts creating files and commits
2. When it needs to push branches, create PRs, or make API calls, it asks for your approval
3. **⏰ You have a limited time window (typically 5-15 minutes)**
4. If you don't click "Approve" within that window → Labs aborts → half-finished code in repo

### What You Need to Do

**BEFORE starting Labs:**
- ✅ Clear your schedule for 3-4 hours
- ✅ Keep Perplexity Labs window in focus or easily accessible
- ✅ Watch for popup notifications
- ✅ Be ready to click "Approve" immediately when prompted

**DURING Labs execution:**
- 👀 Keep the window open and visible
- 🔔 Watch for any approval popups
- ⚡ Click "Approve" as soon as you see them
- 📝 Don't switch away to other tasks

**Typical approval prompts you'll see:**
- "Approve: Create pull request to TheLustriVA/NaRAGtive?"
- "Approve: Commit to feature/tui-phase1-core?"
- "Approve: Push branch to GitHub?"

### If Something Goes Wrong

If the process crashes or halts:

1. Check your repo status:
   ```bash
   git status
   git log --oneline -10
   ```

2. Look for incomplete branches:
   ```bash
   git branch -a | grep tui
   ```

3. Clean up if needed:
   ```bash
   git reset --hard HEAD
   git branch -D feature/tui-phase1-core  # local
   git push origin :feature/tui-phase1-core  # remote (if it exists)
   ```

4. Start over with a fresh Labs task

### Estimated Timeline

- **0:00-0:30** → Labs sets up files and structure
- **0:30-1:00** → First batch of code generation
- **~1:00** → ⚠️ First approval prompt (CREATE BRANCH)
- **1:00-2:00** → More file creation and implementation
- **~2:00** → ⚠️ Second approval prompt (CREATE PULL REQUEST)
- **2:00-2:30** → Final code polish and test generation
- **~2:30** → ⚠️ Final approval prompt (PUSH/MERGE)
- **2:30-3:00** → Cleanup and PR generation

### Pro Tips

1. **Use a separate monitor** if available - keep Labs on one, work on something else on the other
2. **Set a timer** for when you expect each approval (roughly every 30 minutes)
3. **Have your GitHub credentials ready** in case Labs needs them
4. **Keep the browser DevTools closed** - sometimes they interfere with popups
5. **Use fullscreen mode** if possible

### Success Indicators

When Labs finishes successfully:
- ✅ Your branch `feature/tui-phase1-core` exists
- ✅ Pull request is created or ready
- ✅ All 8 new Python files exist in `naragtive/tui/`
- ✅ Test file exists in `tests/test_tui_basic.py`
- ✅ `pyproject.toml` updated with textual dependency
- ✅ No uncommitted changes

### Don't Panic If

- Console shows "Waiting for approval" → This is normal, just click Approve
- There are multiple commits → This is expected, Labs makes incremental commits
- It takes 3+ hours → Completely normal for comprehensive implementation
- You see GitHub API errors → Usually transient, Labs will retry

---

**tl;dr: Block 3-4 hours, keep the window open, click "Approve" when popups appear. You cannot leave it unattended.**
