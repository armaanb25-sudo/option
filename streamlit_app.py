from __future__ import annotations

import pandas as pd
import streamlit as st

from wheel_analyzer.data import load_history, market_snapshot, normalize_option_chain, read_option_csv
from wheel_analyzer.strategy import analyze_chain


st.set_page_config(
    page_title="Wheel Strategy Analyzer",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; }
      div[data-testid="stMetric"] {
        border: 1px solid #d9dee7;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
      }
      .small-note {
        color: #5d6673;
        font-size: 0.9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    st.title("Wheel Strategy Analyzer")
    st.caption("Upload an option-chain CSV, pull 3 years of Yahoo Finance history, and rank cash-secured puts and covered calls.")

    with st.sidebar:
        st.header("Inputs")
        symbol = st.text_input("Yahoo Finance Symbol", value="RELIANCE.NS", help="Examples: RELIANCE.NS, TCS.NS, INFY.NS")
        dte = st.number_input("Days To Expiry", min_value=1, max_value=365, value=30, step=1)
        lot_size = st.number_input("Lot Size", min_value=1, value=250, step=1)
        capital = st.number_input("Capital Available", min_value=0.0, value=500000.0, step=10000.0)
        risk_free_rate = st.number_input("Risk-Free Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.3f")
        uploaded_file = st.file_uploader("Option Chain CSV", type=["csv"])

        st.download_button(
            "Download sample CSV",
            data=_sample_csv(),
            file_name="sample_option_chain.csv",
            mime="text/csv",
            use_container_width=True,
        )

        analyze = st.button("Analyze", type="primary", use_container_width=True)

    if not uploaded_file:
        st.info("Upload a CSV to begin. The importer accepts columns like Strike, Call LTP, Call Bid, Call Ask, Put LTP, Put Bid, Put Ask, OI, Volume, and IV.")
        return

    if not analyze:
        st.info("CSV uploaded. Click Analyze when you are ready.")
        return

    try:
        with st.spinner("Fetching 3 years of price history and analyzing the option chain..."):
            chain = normalize_option_chain(read_option_csv(uploaded_file.getvalue()))
            history = load_history(symbol, period="3y")
            snapshot = market_snapshot(history)
            result = analyze_chain(symbol, snapshot, chain, int(dte), float(capital), int(lot_size), float(risk_free_rate))
    except Exception as exc:
        st.error(str(exc))
        return

    render_result(result)


def render_result(result: dict) -> None:
    snapshot = result["snapshot"]
    st.subheader(f"{result['symbol']} - {result['dte']} DTE Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Spot", _money(snapshot["spot"]))
    col2.metric("3Y Realized Vol", _pct(snapshot["realized_vol"] * 100))
    col3.metric("90D Range Position", _pct(snapshot["range_pos_90d"]))
    col4.metric("From 52W Low", _pct(snapshot["pct_from_52w_low"]))

    csp_df = pd.DataFrame(result["top_csp"])
    cc_df = pd.DataFrame(result["top_covered_calls"])

    chart_df = pd.concat([csp_df.head(5), cc_df.head(5)], ignore_index=True)
    if not chart_df.empty:
        chart_df["candidate"] = chart_df["strategy"] + " " + chart_df["strike"].astype(str)
        st.bar_chart(chart_df.set_index("candidate")["score"])

    tab1, tab2, tab3 = st.tabs(["Cash-Secured Puts", "Covered Calls", "All Candidates"])
    with tab1:
        render_table(csp_df)
    with tab2:
        render_table(cc_df)
    with tab3:
        render_table(pd.DataFrame(result["all_candidates"]))


def render_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No candidates found.")
        return

    display_cols = [
        "recommendation",
        "strike",
        "market_price",
        "model_price",
        "mispricing",
        "mispricing_pct",
        "delta",
        "annualized_return_pct",
        "premium",
        "collateral",
        "breakeven",
        "spread_pct",
        "score",
        "reasons",
    ]
    safe_cols = [col for col in display_cols if col in df.columns]
    table = df[safe_cols].copy()
    if "reasons" in table.columns:
        table["reasons"] = table["reasons"].apply(lambda value: " ".join(value) if isinstance(value, list) else value)

    st.dataframe(table, use_container_width=True, hide_index=True)


def _sample_csv() -> str:
    return """Strike,Call LTP,Call Bid,Call Ask,Call IV,Call OI,Call Volume,Put LTP,Put Bid,Put Ask,Put IV,Put OI,Put Volume
1350,88,86,90,21.5,2400,310,12,11.5,12.5,24.1,4500,620
1400,52,51,53,20.8,3900,540,23,22,24,23.8,6700,880
1450,28,27.5,29,20.2,7100,760,42,41,43,23.2,8200,920
1500,14,13.5,14.5,19.9,9300,1250,75,73,77,22.5,7300,810
1550,7,6.8,7.2,19.4,8500,870,116,114,118,22.2,5200,490
"""


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _pct(value: float) -> str:
    return f"{value:.2f}%"


if __name__ == "__main__":
    main()
