"""Scanner d'opportunités & assistant d'allocation (PEA + CTO/Crypto).

Lancer : streamlit run app.py
Dépendances : pip install streamlit yfinance pandas numpy plotly requests
"""
import hmac

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Scanner d'opportunités", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# Écran de connexion
# ----------------------------------------------------------------------------
APP_PASSWORD = "CryptoPEA2026"

if not st.session_state.get("authenticated"):
    pwd = st.text_input("Mot de passe", type="password")
    if pwd and hmac.compare_digest(pwd, APP_PASSWORD):
        st.session_state["authenticated"] = True
        st.rerun()
    st.warning("Accès restreint")
    if pwd:
        st.error("Mot de passe incorrect")
    st.stop()

# ----------------------------------------------------------------------------
# Univers
# ----------------------------------------------------------------------------
WATCHLIST_STOCKS = {
    "Tech": ["NVDA", "MSFT", "AAPL"],
    "Value/Dividendes": ["JNJ", "KO", "PG", "O", "PEP"],
}
# symbole -> (id CoinGecko, ticker yfinance de secours)
WATCHLIST_CRYPTO = {
    "SOL": ("solana", "SOL-USD"),
    "AVAX": ("avalanche-2", "AVAX-USD"),
    "LINK": ("chainlink", "LINK-USD"),
    "ONDO": ("ondo-finance", "ONDO-USD"),
    "FET": ("fetch-ai", "FET-USD"),
    "ARB": ("arbitrum", "ARB-USD"),
}
PEA_LINES = ["MSCI Europe", "Nasdaq 100", "MSCI Emerging Markets", "Veolia"]
CRYPTO_LINES = ["ETH", "XRP", "TRX"]
DEFAULT_TARGETS = {"MSCI Europe": 30, "Nasdaq 100": 30, "MSCI Emerging Markets": 20, "Veolia": 20}

# Positions détenues : ligne -> (ticker yfinance, quantité). Sert à pré-remplir
# la valeur actuelle de chaque ligne (quantité × dernier prix).
POSITIONS = {
    # PEA (quantités en parts, prix en €)
    "MSCI Europe": {"ticker": "MEUD.PA", "qty": 15, "target_eur": 584.55},
    "Nasdaq 100": {"ticker": "PUUST.PA", "qty": 6, "target_eur": 654.72},
    "MSCI Emerging Markets": {"ticker": "PAEEM.PA", "qty": 36},
    "Veolia": {"ticker": "VIE.PA", "qty": 3, "target_eur": 91.83},
    # Crypto (quantités en unités, prix en USD)
    "ETH": {"ticker": "ETH-USD", "qty": 1.00785},
    "XRP": {"ticker": "XRP-USD", "qty": 60.740026},
    "TRX": {"ticker": "TRX-USD", "qty": 34.790716},
}

# Ticker yfinance -> id CoinGecko (pour les cours crypto en direct en EUR)
COINGECKO_IDS = {
    "ETH-USD": "ethereum",
    "XRP-USD": "ripple",
    "TRX-USD": "tron",
}

RSI_THRESHOLD = 35
LOOKBACK = 20
VOLUME_MULT = 2.0


# ----------------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)  # cours PEA/ETF mis en cache 10 min
def fetch_yf(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="4mo", interval="1d", auto_adjust=True)
    df = df[["Close", "High", "Volume"]].dropna().tail(60)
    df.index = df.index.tz_localize(None)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_coingecko(coin_id: str) -> pd.DataFrame:
    r = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        params={"vs_currency": "eur", "days": 60, "interval": "daily"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    prices = pd.Series({pd.to_datetime(t, unit="ms"): p for t, p in data["prices"]})
    vols = pd.Series({pd.to_datetime(t, unit="ms"): v for t, v in data["total_volumes"]})
    df = pd.DataFrame({"Close": prices, "Volume": vols}).dropna()
    df["High"] = df["Close"]  # CoinGecko daily n'expose pas le plus haut intraday
    return df.tail(60)


def fetch_crypto(symbol: str) -> pd.DataFrame:
    coin_id, yf_ticker = WATCHLIST_CRYPTO[symbol]
    try:
        return fetch_coingecko(coin_id)
    except Exception:
        return fetch_yf(yf_ticker)  # repli si CoinGecko est limité / indisponible


@st.cache_data(ttl=600, show_spinner=False)  # cache 10 min
def latest_price(ticker: str) -> float:
    try:
        return float(fetch_yf(ticker)["Close"].iloc[-1])
    except Exception:
        return 0.0  # repli si le ticker est indisponible


def eur_usd_rate() -> float:
    """Taux EURUSD=X : nombre de USD pour 1 EUR (repli 1.0 si indisponible)."""
    rate = latest_price("EURUSD=X")
    return rate if rate > 0 else 1.0


@st.cache_data(ttl=600, show_spinner=False)  # cours crypto mis en cache 10 min
def fetch_crypto_prices_eur() -> dict:
    """Cours crypto en direct, en EUR, via l'API publique CoinGecko simple/price.

    Retourne {ticker_yf: prix_eur}. Repli sur {} si l'API est indisponible.
    """
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(COINGECKO_IDS.values()), "vs_currencies": "eur"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return {tk: float(data[cg]["eur"]) for tk, cg in COINGECKO_IDS.items() if cg in data}
    except Exception:
        return {}


def unit_price_eur(line: str) -> float:
    """Dernier prix unitaire d'une ligne en EUR.

    Crypto : cours direct en EUR via CoinGecko ; repli yfinance (USD) converti via EURUSD=X.
    PEA/ETF : cours yfinance, déjà en EUR.
    """
    pos = POSITIONS.get(line)
    if not pos:
        return 0.0
    ticker = pos["ticker"]
    if line in CRYPTO_LINES:
        eur_prices = fetch_crypto_prices_eur()
        if ticker in eur_prices:
            return eur_prices[ticker]
        return latest_price(ticker) / eur_usd_rate()  # repli si CoinGecko indisponible
    return latest_price(ticker)


def position_value(line: str) -> float:
    """Valeur actuelle d'une ligne (€) = quantité détenue × dernier prix unitaire."""
    pos = POSITIONS.get(line)
    if not pos:
        return 0.0
    return pos["qty"] * unit_price_eur(line)


# ----------------------------------------------------------------------------
# Analyse technique
# ----------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def analyse(df: pd.DataFrame) -> dict | None:
    if len(df) < LOOKBACK + 2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-1 - LOOKBACK:-1]  # 20 séances précédentes (hors aujourd'hui)
    rsi_now = float(rsi(df["Close"]).iloc[-1])
    vol_ratio = float(last["Volume"] / prev["Volume"].mean()) if prev["Volume"].mean() > 0 else 0.0
    high20 = float(prev["High"].max())
    return {
        "price": float(last["Close"]),
        "rsi": rsi_now,
        "high20": high20,
        "vol_ratio": vol_ratio,
        "perf_60d": float(last["Close"] / df["Close"].iloc[0] - 1),
        "dip": rsi_now < RSI_THRESHOLD,
        "breakout": bool(last["Close"] > high20 and vol_ratio > VOLUME_MULT),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def run_scan() -> tuple[pd.DataFrame, dict]:
    rows, histories = [], {}
    jobs = [(t, grp, "Action", fetch_yf) for grp, ts in WATCHLIST_STOCKS.items() for t in ts]
    jobs += [(s, "Crypto", "Crypto", fetch_crypto) for s in WATCHLIST_CRYPTO]
    for symbol, group, kind, fetcher in jobs:
        try:
            df = fetcher(symbol)
            res = analyse(df)
        except Exception:
            res = None
        if res is None:
            continue
        histories[symbol] = df
        rows.append({"Actif": symbol, "Univers": group, "Type": kind, **res})
    return pd.DataFrame(rows), histories


# ----------------------------------------------------------------------------
# Aide à la décision (RSI 14j · Moyennes mobiles MM50 / MM200)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_close(ticker: str, period: str = "2y") -> pd.Series:
    """Historique de clôtures assez long pour calculer la MM200."""
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    close = df["Close"].dropna()
    close.index = close.index.tz_localize(None)
    return close


def rsi_advice(value: float) -> str:
    if value < 30:
        return "🟢 Opportunité (Survente)"
    if value > 70:
        return "🔴 Attention (Surachat)"
    return "⚪ Neutre"


def trend_signal(price: float, mm200: float) -> str:
    if mm200 != mm200:  # NaN : pas assez d'historique pour la MM200
        return "—"
    return "🟢 Tendance Haussière" if price > mm200 else "🔴 Tendance Baissière"


@st.cache_data(ttl=1800, show_spinner=False)
def decision_table() -> pd.DataFrame:
    """RSI(14) + MM50/MM200 pour chaque ligne détenue (PEA & Crypto)."""
    rows = []
    for line, pos in POSITIONS.items():
        try:
            close = fetch_close(pos["ticker"])
            price = float(close.iloc[-1])
            rsi_now = float(rsi(close).iloc[-1])
            mm50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float("nan")
            mm200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float("nan")
        except Exception:
            continue
        rows.append({
            "Univers": "Crypto" if line in CRYPTO_LINES else "PEA",
            "Ligne": line,
            "Ticker": pos["ticker"],
            "Prix": price,
            "RSI(14)": rsi_now,
            "Conseil RSI": rsi_advice(rsi_now),
            "MM50": mm50,
            "MM200": mm200,
            "Tendance": trend_signal(price, mm200),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Allocation
# ----------------------------------------------------------------------------
def allocate_core(envelope: float, current: dict, targets: dict) -> pd.DataFrame:
    """Répartit l'enveloppe pour se rapprocher des poids cibles après versement."""
    tw = pd.Series(targets, dtype=float)
    tw = tw / tw.sum()
    cur = pd.Series(current, dtype=float)[tw.index]
    total_after = cur.sum() + envelope
    deficit = (tw * total_after - cur).clip(lower=0)
    if deficit.sum() > envelope:
        buy = deficit / deficit.sum() * envelope
    else:  # tous les écarts sont comblés : le reliquat suit les poids cibles
        buy = deficit + tw * (envelope - deficit.sum())
    after = cur + buy
    return pd.DataFrame({
        "Ligne": tw.index,
        "Valeur actuelle (€)": cur.values,
        "Poids actuel": (cur / cur.sum()).fillna(0).values if cur.sum() else 0.0,
        "Poids cible": tw.values,
        "À investir (€)": buy.values,
        "Poids après": (after / total_after).values,
    })


def allocate_opportunities(envelope: float, signals: pd.DataFrame, max_n: int = 3) -> pd.DataFrame:
    """Top 2-3 signaux, Dip prioritaires (RSI le plus bas d'abord), puis Breakouts."""
    if signals.empty or envelope <= 0:
        return pd.DataFrame()
    s = signals.copy()
    s["score"] = np.where(s["dip"], 1.0 + (RSI_THRESHOLD - s["rsi"]) / RSI_THRESHOLD,
                          0.5 + s["vol_ratio"] / 10)
    s = s.sort_values("score", ascending=False).head(max_n)
    s["Montant (€)"] = s["score"] / s["score"].sum() * envelope
    s["Signal"] = np.where(s["dip"], "🟢 Dip", "🔵 Breakout")
    return s


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("💶 Budget du mois")
    budget = st.slider("Budget à investir (€)", 100, 500, 500, step=25)
    core_pct = st.slider("Cœur PEA (%)", 0, 100, 60, step=5,
                         help="Le reste va aux opportunités CTO/Crypto")
    st.caption(f"Cœur PEA **{core_pct}%** · Opportunités **{100 - core_pct}%**")

    with st.form("holdings"):
        st.subheader("Valeur actuelle des lignes")
        st.markdown("**PEA**")
        pea_values = {n: st.number_input(f"{n} (€)", min_value=0.0, value=position_value(n), step=50.0) for n in PEA_LINES}
        st.markdown("**Crypto**")
        crypto_values = {n: st.number_input(f"{n} (€)", min_value=0.0, value=position_value(n), step=50.0) for n in CRYPTO_LINES}
        st.markdown("**Allocation cible PEA (%)**")
        targets = {n: st.number_input(f"Cible {n}", 0, 100, DEFAULT_TARGETS[n], step=5) for n in PEA_LINES}
        st.form_submit_button("Mettre à jour", use_container_width=True)

core_env = budget * core_pct / 100
opp_env = budget - core_env

# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("📈 Scanner d'opportunités & allocation")
c1, c2, c3 = st.columns(3)
c1.metric("Budget du mois", f"{budget:,.0f} €")
c2.metric("Enveloppe Cœur PEA", f"{core_env:,.0f} €", f"{core_pct}%")
c3.metric("Enveloppe Opportunités", f"{opp_env:,.0f} €", f"{100 - core_pct}%")

tab_core, tab_scan, tab_opp, tab_decision = st.tabs(
    ["🏛️ Cœur PEA", "🔍 Scanner", "🎯 Opportunités", "🧭 Aide à la décision"])

with tab_core:
    if sum(targets.values()) == 0:
        st.error("Les cibles PEA sont toutes à 0.")
    else:
        if sum(targets.values()) != 100:
            st.warning(f"Les cibles totalisent {sum(targets.values())}% : elles sont normalisées à 100%.")
        core_df = allocate_core(core_env, pea_values, targets)
        cols = st.columns(len(core_df))
        for col, (_, r) in zip(cols, core_df.iterrows()):
            col.metric(r["Ligne"], f"{r['À investir (€)']:,.0f} €",
                       f"{(r['Poids après'] - r['Poids actuel']) * 100:+.1f} pts de poids")
        st.dataframe(
            core_df.style.format({
                "Valeur actuelle (€)": "{:,.0f}", "À investir (€)": "{:,.0f}",
                "Poids actuel": "{:.1%}", "Poids cible": "{:.1%}", "Poids après": "{:.1%}",
            }).background_gradient(subset=["À investir (€)"], cmap="Greens"),
            hide_index=True, use_container_width=True,
        )
        if sum(crypto_values.values()):
            st.caption("Crypto détenue : " + " · ".join(f"{k} {v:,.0f} €" for k, v in crypto_values.items()))

with tab_decision:
    # --- Modules 1 & 2 : RSI(14) + Moyennes mobiles MM50/MM200 ---
    st.subheader("📊 Signaux techniques — RSI 14j & Tendance MM50/MM200")
    with st.spinner("Calcul des indicateurs…"):
        deci = decision_table()
    if deci.empty:
        st.error("Aucune donnée récupérée pour les lignes détenues (réseau / yfinance).")
    else:
        st.dataframe(
            deci[["Univers", "Ligne", "Ticker", "Prix", "RSI(14)", "Conseil RSI",
                  "MM50", "MM200", "Tendance"]].style.format({
                "Prix": "{:,.2f}", "RSI(14)": "{:.1f}", "MM50": "{:,.2f}", "MM200": "{:,.2f}"}),
            hide_index=True, use_container_width=True,
        )
        st.caption("RSI < 30 : survente (opportunité) · > 70 : surachat (attention) · "
                   "Tendance selon la position du prix vs MM200. "
                   "Prix/MM des cryptos exprimés en USD (signaux inchangés).")

    st.divider()

    # --- Module 3 : Rebalancement DCA ---
    st.subheader("💸 Rebalancement DCA")
    lines = list(POSITIONS.keys())
    dca_amount = st.number_input("Montant d'investissement mensuel (€)", min_value=0.0,
                                 value=200.0, step=50.0, key="dca_amount")
    st.markdown("**Allocation cible (%)** — modifiable par ligne")
    default_pct = round(100 / len(lines), 1)
    dca_targets, cols = {}, st.columns(len(lines))
    for col, line in zip(cols, lines):
        dca_targets[line] = col.number_input(line, min_value=0.0, value=float(default_pct),
                                              step=1.0, key=f"dca_target_{line}")

    if sum(dca_targets.values()) == 0:
        st.error("Les allocations cibles sont toutes à 0.")
    elif dca_amount <= 0:
        st.info("Saisis un montant mensuel pour obtenir les quantités à acheter.")
    else:
        if round(sum(dca_targets.values()), 1) != 100:
            st.warning(f"Les cibles totalisent {sum(dca_targets.values()):.0f}% : "
                       "elles sont normalisées à 100%.")
        current = {line: position_value(line) for line in lines}
        dca = allocate_core(dca_amount, current, dca_targets)
        dca["Prix unitaire (€)"] = dca["Ligne"].map(unit_price_eur)
        dca["Qté à acheter"] = (dca["À investir (€)"]
                                / dca["Prix unitaire (€)"].replace(0, np.nan)).fillna(0)
        st.dataframe(
            dca[["Ligne", "Valeur actuelle (€)", "Poids actuel", "Poids cible",
                 "À investir (€)", "Prix unitaire (€)", "Qté à acheter"]].style.format({
                "Valeur actuelle (€)": "{:,.0f}", "Poids actuel": "{:.1%}", "Poids cible": "{:.1%}",
                "À investir (€)": "{:,.2f}", "Prix unitaire (€)": "{:,.2f}", "Qté à acheter": "{:,.4f}",
            }).background_gradient(subset=["Qté à acheter"], cmap="Greens"),
            hide_index=True, use_container_width=True,
        )
        st.caption(f"Répartition de **{dca_amount:,.0f} €** ce mois-ci pour converger vers l'allocation cible "
                   "(quantités en titres pour le PEA, en jetons pour la crypto).")

with st.spinner("Scan des marchés en cours…"):
    scan, histories = run_scan()

if scan.empty:
    for tab in (tab_scan, tab_opp):
        tab.error("Aucune donnée récupérée (réseau, yfinance ou CoinGecko indisponible).")
    st.stop()

signals = scan[scan["dip"] | scan["breakout"]]

with tab_scan:
    st.caption(f"Dip : RSI(14) < {RSI_THRESHOLD} · Breakout : prix > plus haut {LOOKBACK}j "
               f"et volume > {VOLUME_MULT:.0f}× la moyenne {LOOKBACK}j · fenêtre 60 jours")
    view = scan.assign(Signal=np.select([scan.dip, scan.breakout], ["🟢 Dip", "🔵 Breakout"], "—"))
    st.dataframe(
        view[["Actif", "Univers", "price", "rsi", "high20", "vol_ratio", "perf_60d", "Signal"]].rename(columns={
            "price": "Prix", "rsi": "RSI(14)", "high20": f"Plus haut {LOOKBACK}j",
            "vol_ratio": "Volume / moy.", "perf_60d": "Perf 60j"}).style.format({
            "Prix": "{:,.2f}", "RSI(14)": "{:.1f}", f"Plus haut {LOOKBACK}j": "{:,.2f}",
            "Volume / moy.": "{:.2f}×", "Perf 60j": "{:+.1%}"}),
        hide_index=True, use_container_width=True,
    )

with tab_opp:
    if signals.empty:
        st.info("Aucun signal Dip ou Breakout aujourd'hui : l'enveloppe Opportunités peut être mise de côté.")
    else:
        alloc = allocate_opportunities(opp_env, signals)
        st.subheader(f"Allocation de {opp_env:,.0f} € sur {len(alloc)} signal(aux)")
        cols = st.columns(len(alloc))
        for col, (_, r) in zip(cols, alloc.iterrows()):
            col.metric(f"{r['Signal']} · {r['Actif']}", f"{r['Montant (€)']:,.0f} €",
                       f"RSI {r['rsi']:.0f} · {r['perf_60d']:+.1%} /60j",
                       delta_color="normal" if r["dip"] else "off")

        st.subheader("Actifs en alerte")
        for _, r in signals.iterrows():
            df = histories[r["Actif"]]
            fig = go.Figure(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=r["Actif"],
                                       line=dict(color="#16a34a" if r["dip"] else "#2563eb")))
            fig.add_hline(y=r["high20"], line_dash="dot", annotation_text=f"Plus haut {LOOKBACK}j")
            fig.update_layout(title=f"{r['Actif']} — {'Dip' if r['dip'] else 'Breakout'} "
                                    f"(RSI {r['rsi']:.0f}, vol {r['vol_ratio']:.1f}×)",
                              height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ Outil indicatif, pas un conseil en investissement.")
