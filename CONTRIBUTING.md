# Contributing to PaperPilot

Thanks for your interest! Bug reports, ideas and pull requests are welcome.

## Setup

```bash
git clone https://github.com/Chitira25/PaperPilot.git
cd PaperPilot
python -m venv venv && venv\Scripts\activate    # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Before opening a pull request

1. Run both test scripts and make sure they print `ALL ... PASSED`:
   ```bash
   python scripts/test_pipeline.py
   python scripts/test_agent.py
   ```
2. Add a test for any new behaviour.
3. Keep the core rules intact:
   - Never invent papers, quotes or numbers.
   - Verification labels are decided by code in `modules/verify.py`, never by a model.
   - Offline mode must stay deterministic and key-free.
4. Add a line to `CHANGELOG.md`.

## Reporting bugs

Open an issue with the steps to reproduce, your Python version, and the text shown in the app's status line if a search source failed.
