"""Scanner d'opportunités & assistant d'allocation (PEA + CTO/Crypto).

Lancer : streamlit run app.py
"""
import hmac
import os
import requests
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
import resend

# Configuration de la page (DOIT être la première commande Streamlit)
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
    "Eau": ["AWAT.PA", "XYL", "AWK"],
    "Terres Rares": ["REMX", "ALB"]
}

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

POSITIONS = {
    "MSCI Europe": {"ticker": "PCEU.PA", "qty": 15, "target_eur": 584.55},
    "Nasdaq 100": {"ticker": "PUST.PA", "qty": 6, "target_eur": 654.72},
    "MSCI Emerging Markets": {"ticker": "PAEEM.PA", "qty": 36},
    "Veolia": {"ticker": "VIE.PA", "qty": 3, "target_eur": 91.83},
    "ETH": {"ticker": "ETH-USD", "qty": 1.00785},
    "XRP": {"ticker": "XRP-USD", "qty": 60.740026},
    "TRX": {"ticker": "TRX-USD", "qty": 34.790716},
}

COINGECKO_IDS = {
    "ETH-USD": "ethereum",
    "XRP-USD": "ripple",
    "TRX-USD": "tron",
}

DCA_DEFAULT_TARGETS = {
    "MSCI Europe": 35,
    "Nasdaq 100": 25,
    "MSCI Emerging Markets": 15,
    "Veolia": 5,
    "ETH": 15,
    "XRP": 2.5,
    "TRX": 2.5,
}

RSI_THRESHOLD = 35
LOOKBACK = 20
VOLUME_MULT = 2.0

# ----------------------------------------------------------------------------
# Configuration Resend (Hardcodée)
# ----------------------------------------------------------------------------
RESEND_API_KEY = st.secrets["RESEND_API_KEY"]
DEST_EMAIL = "wolff.g@sfeir.com"
HISTORY_FILE = "history.csv"


# ----------------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
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
    df["High"] = df["Close"]
    return df.tail(60)


def fetch_crypto(symbol: str) -> pd.DataFrame:
    coin_id, yf_ticker = WATCHLIST_CRYPTO[symbol]
    try:
        return fetch_coingecko(coin_id)
    except Exception:
        return fetch_yf(yf_ticker)


@st.cache_data(ttl=600, show_spinner=False)
def latest_price(ticker: str) -> float:
    try:
        return float(fetch_yf(ticker)["Close"].iloc[-1])
    except Exception:
        return 0.0


def eur_usd_rate() -> float:
    rate = latest_price("EURUSD=X")
    return rate if rate > 0 else 1.0


@st.cache_data(ttl=600, show_spinner=False)
def fetch_crypto_prices_eur() -> dict:
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
    pos = POSITIONS.get(line)
    if not pos:
        return 0.0
    ticker = pos["ticker"]
    if line in CRYPTO_LINES:
        eur_prices = fetch_crypto_prices_eur()
        if ticker in eur_prices:
            return eur_prices[ticker]
        return latest_price(ticker) / eur_usd_rate()
    return latest_price(ticker)


def position_value(line: str) -> float:
    pos = POSITIONS.get(line)
    if not pos:
        return 0.0
    return pos["qty"] * unit_price_eur(line)


# ----------------------------------------------------------------------------
# Analyse technique & Prix Cibles (Évolution 5)
# ----------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def calculate_target_price_rsi30(df: pd.DataFrame, target_rsi: float = 30.0) -> float:
    """Calcule le prix théorique auquel l'actif atteindrait un RSI cible (30)."""
    if len(df) < 15:
        return 0.0
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    
    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]
    last_close = close.iloc[-1]
    
    # Résolution de l'équation du Wilder's Smoothing RSI
    target_rs = target_rsi / (100 - target_rsi)
    required_change = (target_rs * (13 * last_loss) - 13 * last_gain) / (1 - target_rs)
    target_price = last_close + required_change
    return float(max(target_price, 0.0))


def analyse(df: pd.DataFrame) -> dict | None:
    if len(df) < LOOKBACK + 2:
        return None
        
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    last = df.iloc[-1]
    prev = df.iloc[-1 - LOOKBACK:-1]
    rsi_now = float(rsi(df["Close"]).iloc[-1])
    vol_ratio = float(last["Volume"] / prev["Volume"].mean()) if prev["Volume"].mean() > 0 else 0.0
    high20 = float(prev["High"].max())
    
    target_price_rsi30 = calculate_target_price_rsi30(df, 30.0)
    
    return {
        "price": float(last["Close"]),
        "rsi": rsi_now,
        "high20": high20,
        "vol_ratio": vol_ratio,
        "perf_60d": float(last["Close"] / df["Close"].iloc[0] - 1),
        "dip": rsi_now < RSI_THRESHOLD,
        "breakout": bool(last["Close"] > high20 and vol_ratio > VOLUME_MULT),
        "macd": float(last["MACD"]),
        "macd_signal": float(last["Signal_Line"]),
        "macd_hist": float(last["MACD_Hist"]),
        "target_rsi30": target_price_rsi30,
    }


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
# Suivi Historique de la Valeur (Évolution 4)
# ----------------------------------------------------------------------------
def save_portfolio_snapshot(total_pea: float, total_crypto: float):
    """Sauvegarde l'historique de la valeur du portefeuille."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    total_val = total_pea + total_crypto
    new_entry = pd.DataFrame([{"Date": today, "PEA": total_pea, "Crypto": total_crypto, "Total": total_val}])
    
    if os.path.exists(HISTORY_FILE):
        hist_df = pd.read_csv(HISTORY_FILE)
        hist_df = pd.concat([hist_df[hist_df["Date"] != today], new_entry], ignore_index=True)
    else:
        hist_df = new_entry
        
    hist_df.to_csv(HISTORY_FILE, index=False)
    return hist_df


def load_portfolio_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["Date", "PEA", "Crypto", "Total"])


# ----------------------------------------------------------------------------
# Aide à la décision
# ----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_close(ticker: str, period: str = "2y") -> pd.Series:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    close = df["Close"].dropna()
    close.index = close.index.tz_localize(None)
    return close


def rsi_advice(value: float) -> str:
    if value < 30: return "🟢 Opportunité (Survente)"
    if value > 70: return "🔴 Attention (Surachat)"
    return "⚪ Neutre"


def trend_signal(price: float, mm200: float) -> str:
    if mm200 != mm200: return "—"
    return "🟢 Tendance Haussière" if price > mm200 else "🔴 Tendance Baissière"


@st.cache_data(ttl=1800, show_spinner=False)
def decision_table() -> pd.DataFrame:
    rows = []
    for line, pos in POSITIONS.items():
        try:
            close = fetch_close(pos["ticker"])
            price = float(close.iloc[-1])
            rsi_now = float(rsi(close).iloc[-1])
            mm50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float("nan")
            mm200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float("nan")
            pru = pos.get("pru")
            if pru and pru > 0:
                pv = unit_price_eur(line) / pru - 1
            else:
                pv = price / float(close.iloc[0]) - 1 if len(close) else float("nan")
        except Exception:
            continue
        rows.append({
            "Univers": "Crypto" if line in CRYPTO_LINES else "PEA",
            "Ligne": line, "Ticker": pos["ticker"], "Prix": price,
            "RSI(14)": rsi_now, "Conseil RSI": rsi_advice(rsi_now),
            "MM50": mm50, "MM200": mm200, "Tendance": trend_signal(price, mm200),
            "PV latente": pv,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Envoi d'e-mails (Resend API)
# ----------------------------------------------------------------------------
def send_resend_email(subject, body):
    resend.api_key = RESEND_API_KEY
    try:
        resend.Emails.send({
            "from": "BOURSE APP <onboarding@resend.dev>",
            "to": DEST_EMAIL,
            "subject": subject,
            "text": body,
        })
        return True, f"✅ E-mail envoyé avec succès à {DEST_EMAIL} !"
    except Exception as e:
        return False, f"❌ Erreur Resend : {e}"

def rsi_alerts(deci: pd.DataFrame) -> list:
    msgs = []
    if deci is None or deci.empty: return msgs
    for _, r in deci.iterrows():
        val = r["RSI(14)"]
        if val < 35: msgs.append(f"🟢 {r['Ligne']} ({r['Ticker']}) — Opportunité : RSI {val:.0f} (< 35)")
        elif val > 75: msgs.append(f"🔴 {r['Ligne']} ({r['Ticker']}) — Surachat : RSI {val:.0f} (> 75)")
    return msgs

# ----------------------------------------------------------------------------
# Allocation & DCA
# ----------------------------------------------------------------------------
def allocate_core(envelope: float, current: dict, targets: dict) -> pd.DataFrame:
    tw = pd.Series(targets, dtype=float)
    tw = tw / tw.sum()
    cur = pd.Series(current, dtype=float)[tw.index]
    total_after = cur.sum() + envelope
    deficit = (tw * total_after - cur).clip(lower=0)
    if deficit.sum() > envelope: buy = deficit / deficit.sum() * envelope
    else: buy = deficit + tw * (envelope - deficit.sum())
    after = cur + buy
    return pd.DataFrame({
        "Ligne": tw.index, "Valeur actuelle (€)": cur.values,
        "Poids actuel": (cur / cur.sum()).fillna(0).values if cur.sum() else 0.0,
        "Poids cible": tw.values, "À investir (€)": buy.values,
        "Poids après": (after / total_after).values,
    })

def allocate_opportunities(envelope: float, signals: pd.DataFrame, max_n: int = 3) -> pd.DataFrame:
    if signals.empty or envelope <= 0: return pd.DataFrame()
    s = signals.copy()
    s["score"] = np.where(s["dip"], 1.0 + (RSI_THRESHOLD - s["rsi"]) / RSI_THRESHOLD, 0.5 + s["vol_ratio"] / 10)
    s = s.sort_values("score", ascending=False).head(max_n)
    s["Montant (€)"] = s["score"] / s["score"].sum() * envelope
    s["Signal"] = np.where(s["dip"], "🟢 Dip", "🔵 Breakout")
    return s

DCA_MESSAGES = {
    ("ETF", "block"): "🔴 Surachat ETF (RSI > 70) — Achat temporairement bloqué",
    ("Crypto", "block"): "🔴 Surchauffe Crypto (RSI > 75) — Achat temporairement bloqué",
    ("ETF", "boost"): "🟢 Opportunité ETF (RSI < 40) — Surpondérer l'achat",
    ("Crypto", "boost"): "🟢 Opportunité Crypto (RSI < 35) — Surpondérer l'achat",
}
DCA_BOOST = 0.5

def smart_dca_allocate(envelope: float, current: dict, targets: dict, info: dict) -> pd.DataFrame:
    base = allocate_core(envelope, current, targets)
    idx = list(base["Ligne"])
    buy = pd.Series(base["À investir (€)"].to_numpy(), index=idx, dtype=float)
    rsi = pd.Series({l: info.get(l, {}).get("rsi", float("nan")) for l in idx})
    trend = pd.Series({l: info.get(l, {}).get("trend", "—") for l in idx})
    is_crypto = pd.Series({l: l in CRYPTO_LINES for l in idx})

    blocked = (~is_crypto & (rsi > 70)) | (is_crypto & (rsi > 75))
    boosted = (~is_crypto & (rsi < 40)) | (is_crypto & (rsi < 35))

    freed = float(buy.where(blocked, 0.0).sum())
    buy[blocked] = 0.0
    if freed > 0:
        cand = ~blocked
        rsi_pool = rsi.where(cand) if cand.any() else rsi
        if rsi_pool.notna().any():
            recip = (rsi == rsi_pool.min())
            if cand.any(): recip = recip & cand
        else: recip = cand if cand.any() else pd.Series(True, index=idx)
        n = int(recip.sum())
        if n > 0: buy[recip] = buy[recip] + freed / n

    if boosted.any():
        extra = float((buy.where(boosted, 0.0) * DCA_BOOST).sum())
        donors = ~boosted & ~blocked
        donor_amt = buy.where(donors, 0.0)
        if extra > 0 and donor_amt.sum() >= extra:
            buy[boosted] = buy[boosted] * (1 + DCA_BOOST)
            buy[donors] = buy[donors] - extra * donor_amt[donors] / donor_amt.sum()

    def message(line: str) -> str:
        cls = "Crypto" if line in CRYPTO_LINES else "ETF"
        if blocked[line]: return DCA_MESSAGES[(cls, "block")]
        if boosted[line]: return DCA_MESSAGES[(cls, "boost")]
        return ""

    cur = pd.Series(base["Valeur actuelle (€)"].to_numpy(), index=idx, dtype=float)
    total_after = cur.sum() + envelope
    return pd.DataFrame({
        "Ligne": idx, "Valeur actuelle (€)": cur.to_numpy(),
        "Poids actuel": base["Poids actuel"].to_numpy(), "Poids cible": base["Poids cible"].to_numpy(),
        "RSI(14)": rsi.to_numpy(), "Tendance": trend.to_numpy(),
        "À investir (€)": buy.to_numpy(), "Poids après": ((cur + buy) / total_after).to_numpy() if total_after else 0.0,
        "Conseil DCA": [message(l) for l in idx],
    })


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("💶 Budget du mois")
    budget = st.slider("Budget à investir (€)", 100, 500, 500, step=25)
    core_pct = st.slider("Cœur PEA (%)", 0, 100, 60, step=5)
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

    st.divider()
    st.subheader("📩 Service d'alertes")
    st.caption(f"Expéditeur : **BOURSE APP**\nDestinataire : **{DEST_EMAIL}**")

    if st.button("📩 Tester l'envoi d'un e-mail d'alerte", use_container_width=True):
        ok, detail = send_resend_email(
            "Test d'alerte — Scanner d'opportunités",
            "✅ Ceci est un e-mail de test configuré automatiquement avec ton compte Resend."
        )
        (st.success if ok else st.error)(detail)

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

tab_core, tab_scan, tab_opp, tab_decision, tab_bilan = st.tabs(
    ["🏛️ Cœur PEA", "🔍 Scanner", "🎯 Opportunités", "🧭 Aide à la décision", "🧾 Bilan & Fiscalité"])

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

with tab_decision:
    st.subheader("📊 Signaux techniques — RSI 14j & Tendance MM50/MM200")
    with st.spinner("Calcul des indicateurs…"): deci = decision_table()
    if deci.empty: st.error("Aucune donnée récupérée pour les lignes détenues (réseau / yfinance).")
    else:
        st.dataframe(
            deci[["Univers", "Ligne", "Ticker", "Prix", "RSI(14)", "Conseil RSI", "MM50", "MM200", "Tendance"]].style.format({
                "Prix": "{:,.2f}", "RSI(14)": "{:.1f}", "MM50": "{:,.2f}", "MM200": "{:,.2f}"}),
            hide_index=True, use_container_width=True,
        )
    st.divider()
    st.subheader("🛡️ Prise de profit & Sécurisation")
    if not deci.empty:
        overheated = pd.Series(np.where(deci["Univers"] == "Crypto", deci["RSI(14)"] > 75, deci["RSI(14)"] > 70), index=deci.index)
        secure = deci[(deci["PV latente"] > 0.30) & overheated]
        if secure.empty: st.success("Aucune ligne à sécuriser.")
        else:
            st.dataframe(secure[["Univers", "Ligne", "Ticker", "Prix", "PV latente", "RSI(14)"]].style.format(
                    {"Prix": "{:,.2f}", "PV latente": "{:+.1%}", "RSI(14)": "{:.1f}"}), hide_index=True, use_container_width=True)
    st.divider()
    st.subheader("💸 Smart DCA — rebalancement & conseils conditionnels")
    lines = list(POSITIONS.keys())
    dca_amount = st.number_input("Montant mensuel (€)", min_value=0.0, value=200.0, step=50.0)
    dca_targets, cols = {}, st.columns(len(lines))
    for col, line in zip(cols, lines):
        dca_targets[line] = col.number_input(line, min_value=0.0, value=float(DCA_DEFAULT_TARGETS.get(line, round(100 / len(lines), 1))), step=1.0)

    if dca_amount > 0 and sum(dca_targets.values()) > 0:
        current = {line: position_value(line) for line in lines}
        info = ({r["Ligne"]: {"rsi": r["RSI(14)"], "trend": r["Tendance"]} for _, r in deci.iterrows()} if not deci.empty else {})
        dca = smart_dca_allocate(dca_amount, current, dca_targets, info)
        dca["Prix unitaire (€)"] = dca["Ligne"].map(unit_price_eur)
        dca["Qté à acheter"] = (dca["À investir (€)"] / dca["Prix unitaire (€)"].replace(0, np.nan)).fillna(0)
        st.dataframe(
            dca[["Ligne", "RSI(14)", "Tendance", "À investir (€)", "Prix unitaire (€)", "Qté à acheter", "Conseil DCA"]].style.format({
                "RSI(14)": "{:.1f}", "À investir (€)": "{:,.2f}", "Prix unitaire (€)": "{:,.2f}", "Qté à acheter": "{:,.4f}"
            }).background_gradient(subset=["Qté à acheter"], cmap="Greens"),
            hide_index=True, use_container_width=True,
        )

with tab_bilan:
    st.subheader("🧾 Bilan & Fiscalité")
    
    # --- Module Historique du Portefeuille (Évolution 4) ---
    st.markdown("### 📈 Suivi Historique du Portefeuille")
    total_pea = sum(pea_values.values())
    total_crypto = sum(crypto_values.values())
    
    col_hist1, col_hist2 = st.columns([1, 3])
    with col_hist1:
        st.metric("Total PEA", f"{total_pea:,.0f} €")
        st.metric("Total Crypto", f"{total_crypto:,.0f} €")
        st.metric("Valeur Totale", f"{total_pea + total_crypto:,.0f} €")
        if st.button("📸 Sauvegarder ce point historique", use_container_width=True):
            save_portfolio_snapshot(total_pea, total_crypto)
            st.success("Point historique enregistré !")
            
    with col_hist2:
        hist_df = load_portfolio_history()
        if not hist_df.empty:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["PEA"], mode="lines+markers", name="PEA", line=dict(color="#2563eb")))
            fig_hist.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Crypto"], mode="lines+markers", name="Crypto", line=dict(color="#f59e0b")))
            fig_hist.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Total"], mode="lines+markers", name="Total", line=dict(color="#16a34a", width=3)))
            fig_hist.update_layout(title="Progression du Portefeuille (€)", height=250, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Clique sur le bouton pour enregistrer la première mesure de ton portefeuille.")

    st.divider()
    st.markdown("### 📊 Plafond PEA")
    versements = st.number_input("Versements cumulés sur le PEA (€)", min_value=0.0, max_value=1_000_000.0, value=float(round(total_pea)), step=500.0)
    PLAFOND_PEA = 150_000.0
    b1, b2, b3 = st.columns(3)
    b1.metric("Versements", f"{versements:,.0f} €")
    b2.metric("Plafond", f"{PLAFOND_PEA:,.0f} €")
    b3.metric("Disponible", f"{max(PLAFOND_PEA - versements, 0.0):,.0f} €")
    st.progress(min(versements / PLAFOND_PEA, 1.0))
    st.divider()
    st.markdown("### 📩 Alertes RSI")
    alerts = rsi_alerts(deci)
    if not alerts:
        st.success("Aucun actif en opportunité ou en surachat actuellement.")
    else:
        for a in alerts: st.write("- " + a)
        if st.button("📩 Envoyer ces alertes par e-mail", use_container_width=True):
            ok, detail = send_resend_email("Alertes RSI — Scanner d'opportunités", "📊 Alertes RSI :\n" + "\n".join(alerts))
            (st.success if ok else st.error)(detail)

with st.spinner("Scan des marchés en cours…"):
    scan, histories = run_scan()

if scan.empty:
    st.stop()

signals = scan[scan["dip"] | scan["breakout"]]

with tab_scan:
    view = scan.assign(Signal=np.select([scan.dip, scan.breakout], ["🟢 Dip", "🔵 Breakout"], "—"))
    st.caption("Prix Cible RSI 30 : Cours théorique à atteindre pour placer un ordre d'achat limite en zone d'opportunité.")
    st.dataframe(
        view[["Actif", "Univers", "price", "rsi", "target_rsi30", "macd", "macd_signal", "high20", "vol_ratio", "perf_60d", "Signal"]].rename(columns={
            "price": "Prix", "rsi": "RSI(14)", "target_rsi30": "Prix Cible (RSI 30)", "macd": "MACD", "macd_signal": "Signal MACD", f"high20": f"Plus haut {LOOKBACK}j",
            "vol_ratio": "Volume / moy.", "perf_60d": "Perf 60j"}).style.format({
            "Prix": "{:,.2f}", "RSI(14)": "{:.1f}", "Prix Cible (RSI 30)": "{:,.2f}", "MACD": "{:.2f}", "Signal MACD": "{:.2f}", f"Plus haut {LOOKBACK}j": "{:,.2f}",
            "Volume / moy.": "{:.2f}×", "Perf 60j": "{:+.1%}"}),
        hide_index=True, use_container_width=True,
    )

with tab_opp:
    if not signals.empty:
        alloc = allocate_opportunities(opp_env, signals)
        cols = st.columns(len(alloc))
        for col, (_, r) in zip(cols, alloc.iterrows()):
            col.metric(f"{r['Signal']} · {r['Actif']}", f"{r['Montant (€)']:,.0f} €", f"RSI {r['rsi']:.0f} · {r['perf_60d']:+.1%} /60j", delta_color="normal" if r["dip"] else "off")
        
        # --- Visualisation Graphique MACD & Hist (Évolution 1) ---
        for _, r in signals.iterrows():
            df = histories[r["Actif"]]
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Subplot 1: Prix
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Prix", line=dict(color="#16a34a" if r["dip"] else "#2563eb")), row=1, col=1)
            fig.add_hline(y=r["high20"], line_dash="dot", annotation_text=f"Plus haut {LOOKBACK}j", row=1, col=1)
            
            # Subplot 2: MACD + Signal + Histogramme
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", line=dict(color="#2563eb")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["Signal_Line"], mode="lines", name="Signal", line=dict(color="#f97316")), row=2, col=1)
            
            colors = ["#16a34a" if val >= 0 else "#dc2626" for val in df["MACD_Hist"]]
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogramme", marker_color=colors), row=2, col=1)
            
            fig.update_layout(title=f"{r['Actif']} — Courbe Prix & Indicateur MACD", height=400, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
