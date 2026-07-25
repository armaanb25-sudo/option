# Wheel Strategy Analyzer

A Streamlit web app that imports a manual option-chain CSV, pulls 3 years of Yahoo Finance history, prices options with Black-Scholes, compares model price with current market price, and ranks wheel strategy candidates.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the local URL Streamlit prints.

## CSV Columns

The importer is flexible. Use any of these common names:

- `Strike` or `Strike Price`
- `Call LTP`, `Call Bid`, `Call Ask`, `Call IV`, `Call OI`, `Call Volume`
- `Put LTP`, `Put Bid`, `Put Ask`, `Put IV`, `Put OI`, `Put Volume`

It also recognizes `CE LTP`, `PE LTP`, `CE Bid`, `PE Ask`, and similar variants.

## Deploy On Streamlit Cloud

Push this folder to GitHub, then go to Streamlit Cloud and create a new app with:

- Repository: your GitHub repo
- Branch: `main`
- Main file path: `streamlit_app.py`

Streamlit Cloud will install `requirements.txt` automatically.
