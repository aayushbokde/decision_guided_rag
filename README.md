
## Streamlit research demo

The project includes a single-question interactive demo in `app.py`.

From the project root:

```powershell
uv sync
uv run streamlit run app.py
```

The UI exposes the current Normal RAG, decision-guided filtering, and accumulated-evidence early-exit pipelines. The decision modes currently use `KeywordOracle` as a deterministic development proxy; this is explicitly labelled in the UI and is **not JEV**.
