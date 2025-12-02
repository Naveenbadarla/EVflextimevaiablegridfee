import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import datetime
import base64

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="E.ON EV Charging Optimisation",
    page_icon="⚡",
    layout="wide",
)

# =============================================================================
# GLOBAL PREMIUM E.ON STYLING (works for light & dark themes)
# =============================================================================
st.markdown(
    """
    <style>
    /* Overall background & container */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
    }

    .block-container {
        max-width: 1300px;
        padding-top: 0.5rem;
        padding-bottom: 3rem;
    }

    /* Cards */
    .eon-card {
        background-color: #111827;
        border-radius: 18px;
        padding: 18px 22px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.35);
        color: #e5e7eb;
    }

    .eon-card h3 {
        margin-top: 0;
        color: #f9fafb;
    }

    .eon-card p, .eon-card li {
        color: #e5e7eb;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #111827 !important;
        color: #e5e7eb !important;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.5);
        padding: 10px 12px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 1px solid rgba(15, 23, 42, 0.9);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f9fafb;
    }

    /* Inputs */
    .stNumberInput input, .stTextInput input {
        border-radius: 10px !important;
    }

    .stFileUploader > label {
        font-weight: 500;
        color: #e5e7eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# LOGO LOADING
# =============================================================================
def load_logo(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return ""

logo_base64 = load_logo("eon_logo.png")

# =============================================================================
# PREMIUM MINIMAL E.ON HEADER
# =============================================================================
st.markdown(
    """
    <style>
    @keyframes subtleShift {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    .eon-header {
        width: 100%;
        padding: 24px 40px;
        border-radius: 20px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        justify-content: flex-start;

        background: linear-gradient(90deg,
            #E2000F 0%,
            #D9001A 40%,
            #C90024 100%
        );
        background-size: 220% 220%;
        animation: subtleShift 14s ease-in-out infinite;
        box-shadow: 0 25px 60px rgba(0,0,0,0.5);
    }

    .eon-header-logo {
        width: 170px;
        margin-right: 40px;
        filter: drop-shadow(0 4px 12px rgba(0,0,0,0.4));
    }

    .eon-header-title {
        font-size: 36px;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.4px;
    }

    .eon-header-sub {
        font-size: 18px;
        font-weight: 300;
        color: rgba(255,255,255,0.9);
        margin-top: 6px;
    }

    .eon-nav {
        margin-top: 12px;
        display: flex;
        gap: 26px;
    }

    .eon-nav a {
        font-size: 15px;
        color: rgba(255,255,255,0.92);
        text-decoration: none;
        transition: 0.25s;
        padding-bottom: 4px;
        border-bottom: 2px solid transparent;
    }

    .eon-nav a:hover {
        border-bottom: 2px solid rgba(255,255,255,0.9);
    }

    @media (max-width: 768px) {
        .eon-header {
            flex-direction: column;
            padding: 22px 20px;
            text-align: center;
        }
        .eon-header-logo {
            margin-right: 0;
            margin-bottom: 12px;
            width: 150px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

header_logo_html = (
    f'<img src="data:image/png;base64,{logo_base64}" class="eon-header-logo">'
    if logo_base64
    else ""
)

st.markdown(
    f"""
    <div class="eon-header">
        {header_logo_html}
        <div>
            <div class="eon-header-title">EV Charging Optimisation Dashboard</div>
            <div class="eon-header-sub">
                Full-Year Day-Ahead & Intraday Market-Based Smart Charging
            </div>
            <div class="eon-nav">
                <a href="#overview">Overview</a>
                <a href="#prices">Market Prices</a>
                <a href="#results">Results</a>
                <a href="#details">Details</a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<h1 style='text-align:center; font-family:Trebuchet MS, Helvetica, sans-serif; font-weight:700; color:#E2000F; letter-spacing:1px; margin-top:12px;'>E.ON AI<span style='color:white;'>X</span> — AI e<span style='color:white;'>X</span>cellence Initiative</h1>", unsafe_allow_html=True)

from streamlit.components.v1 import html

html(
    """
<style>

.ev-wrapper {
    width: 100%;
    display: flex;
    justify-content: center;
    margin-top: 35px;
    margin-bottom: 40px;
}

.ev-card {
    background: linear-gradient(145deg, #1e1f22, #25262b);
    border-radius: 24px;
    padding: 40px 30px;
    width: 95%;
    max-width: 900px;
    box-shadow: 0 10px 45px rgba(0,0,0,0.7);
    border: 1px solid rgba(255,255,255,0.10);
}

.ev-car {
    width: 240px;
    height: 110px;
    border-radius: 28px;
    background: linear-gradient(180deg, #3a3d42, #2d2f34 80%);
    position: relative;
    margin: 0 auto;
    box-shadow: 0 0 30px rgba(255,0,0,0.25);
}

.ev-wheel {
    width: 46px;
    height: 46px;
    background: #000;
    border-radius: 50%;
    border: 5px solid #8b8b8b;
    position: absolute;
    bottom: -20px;
}
.ev-wheel.left { left: 32px; }
.ev-wheel.right { right: 32px; }

.ev-port {
    width: 16px;
    height: 16px;
    background: #E2000F;
    border-radius: 50%;
    position: absolute;
    right: -12px;
    top: 42px;
    box-shadow: 0 0 14px rgba(226,0,15,1);
}

.ev-cable {
    width: 140px;
    height: 4px;
    background: rgba(255,255,255,0.25);
    position: absolute;
    right: -140px;
    top: 50px;
    border-radius: 2px;
}

.ev-dot {
    width: 10px;
    height: 10px;
    background: #ff3640;
    border-radius: 50%;
    position: absolute;
    animation: cablePulse 1.8s infinite linear;
    box-shadow: 0 0 12px rgba(255,40,40,0.9);
}

.ev-charger {
    width: 90px;
    height: 155px;
    background: linear-gradient(180deg, #2e3036, #1a1a1c 85%);
    border-radius: 16px;
    position: absolute;
    right: -250px;
    top: -12px;
    border: 1px solid rgba(255,255,255,0.15);
}

.ev-charger-screen {
    width: 50px;
    height: 28px;
    background: #E2000F;
    border-radius: 6px;
    margin: 18px auto;
}

@keyframes cablePulse {
    0% { left: 0px; opacity: 1; }
    100% { left: 130px; opacity: 0; }
}

</style>

<div class="ev-wrapper">
    <div class="ev-card">
        <div style="position: relative; height: 180px;">

            <div class="ev-car">
                <div class="ev-wheel left"></div>
                <div class="ev-wheel right"></div>
                <div class="ev-port"></div>
                <div class="ev-cable"></div>

                <div class="ev-dot" style="animation-delay: 0s;"></div>
                <div class="ev-dot" style="animation-delay: 0.35s;"></div>
                <div class="ev-dot" style="animation-delay: 0.7s;"></div>

                <div class="ev-charger">
                    <div class="ev-charger-screen"></div>
                </div>
            </div>

        </div>
        <h3 style='text-align:center;margin-top:22px;color:white;font-weight:300;'>
            Smart Charging… Optimising Your Energy Costs
        </h3>
    </div>
</div>
""",
    height=350,
)

# =============================================================================
# SYNTHETIC PRICE GENERATION (fallback if no upload)
# =============================================================================
def get_synthetic_daily_price_profiles():
    hours = np.arange(24)
    base = 70
    morning_peak = 15 * np.sin((hours - 7) / 24 * 2 * np.pi) ** 2
    evening_peak = 35 * np.sin((hours - 18) / 24 * 2 * np.pi) ** 4
    da_hourly = base + morning_peak + evening_peak  # €/MWh

    rng = np.random.default_rng(42)
    da_q = np.repeat(da_hourly, 4)
    id_q = da_q + rng.normal(0, 6, size=96)

    return da_hourly, id_q

# =============================================================================
# CHARGING FREQUENCY RULES
# =============================================================================
DAY_NAME_TO_IDX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

def compute_charging_days(pattern, num_days, custom=None, weekly=None):
    if custom is None:
        custom = []

    if pattern == "Every day":
        return list(range(num_days))

    if pattern == "Every other day":
        return list(range(0, num_days, 2))

    if pattern == "Weekdays only (Mon–Fri)":
        return [d for d in range(num_days) if (d % 7) < 5]

    if pattern == "Weekends only (Sat–Sun)":
        return [d for d in range(num_days) if (d % 7) >= 5]

    if pattern == "Custom weekdays":
        allowed = {DAY_NAME_TO_IDX[d] for d in custom}
        return [d for d in range(num_days) if (d % 7) in allowed]

    if pattern == "Custom: X sessions per week":
        weekly = int(weekly)
        out = []
        weeks = num_days // 7
        for w in range(weeks):
            for s in range(weekly):
                day = w * 7 + int(np.floor(s * 7 / weekly))
                if day < num_days:
                    out.append(day)
        return out

    return list(range(num_days))

# =============================================================================
# TIME WINDOW PARSER (15-min resolution)
# =============================================================================
def build_available_quarters(arrival, departure):
    aq = int(arrival * 4) % 96
    dq = int(departure * 4) % 96
    if aq == dq:  # full day
        return list(range(96)), aq, dq
    if aq < dq:
        return list(range(aq, dq)), aq, dq
    return list(range(aq, 96)) + list(range(0, dq)), aq, dq

# =============================================================================
# COSTING LOGIC
# =============================================================================
def apply_tariffs(price_kwh, grid_fee, taxes, vat):
    """
    price_kwh: energy wholesale price in €/kWh
    grid_fee:  grid network charges in €/kWh
    taxes:     taxes & levies in €/kWh
    vat:       VAT in %
    """
    return (price_kwh + grid_fee + taxes) * (1 + vat / 100)

def compute_da_indexed_baseline_daily(E, Pmax, quarters, da_day, grid, taxes, vat):
    """DA-indexed, chronological charging, constant grid fee (no Mod3)."""
    da_q = np.repeat(da_day / 1000.0, 4)  # €/kWh
    remain = E
    cost = 0.0
    emax = Pmax * 0.25  # kWh per quarter

    for q in quarters:
        if remain <= 1e-6:
            break
        e = min(emax, remain)
        cost += e * apply_tariffs(da_q[q], grid, taxes, vat)
        remain -= e

    if remain > 1e-6:
        raise ValueError("Charging window too short to deliver requested energy.")
    return cost

def compute_optimised_daily_cost(E, Pmax, quarters, price_q, grid, taxes, vat):
    """Optimised on wholesale prices only, constant grid fee."""
    pq = price_q / 1000.0
    emax = Pmax * 0.25
    if len(quarters) * emax < E - 1e-6:
        raise ValueError("Charging window too short for requested energy.")

    sorted_q = sorted(quarters, key=lambda x: pq[x])
    remain = E
    cost = 0.0
    for q in sorted_q:
        if remain <= 1e-6:
            break
        e = min(emax, remain)
        cost += e * apply_tariffs(pq[q], grid, taxes, vat)
        remain -= e
    return cost

# =============================================================================
# MODUL 3: HT/ST/NT GRID FEE STRUCTURE
# =============================================================================

# DSO-specific prices (ct/kWh)
dso_tariffs = {
    "Westnetz": {"HT": 15.65, "ST": 9.53, "NT": 0.95},
    "Avacon": {"HT": 8.41, "ST": 6.04, "NT": 0.60},
    "MVV Netze": {"HT": 5.96, "ST": 4.32, "NT": 1.73},
    "MITNETZ": {"HT": 12.60, "ST": 6.31, "NT": 0.69},
    "Stadtwerke München": {"HT": 7.14, "ST": 6.47, "NT": 2.59},
    "Thüringer Energienetze": {"HT": 8.62, "ST": 5.56, "NT": 1.67},
    "LEW": {"HT": 8.09, "ST": 7.09, "NT": 4.01},
    "NetzeBW": {"HT": 13.20, "ST": 7.57, "NT": 3.03},
    "Bayernwerk": {"HT": 9.03, "ST": 4.72, "NT": 0.47},
    "EAM Netz": {"HT": 10.52, "ST": 5.48, "NT": 1.64},
}

# Your DSO-specific quarter validity (True = Modul 3 active)
dso_quarter_valid = {
    "Westnetz":               {1: True, 2: True, 3: True, 4: True},
    "Avacon":                 {1: True, 2: False, 3: False, 4: True},
    "MVV Netze":              {1: True, 2: False, 3: False, 4: True},
    "MITNETZ":                {1: True, 2: False, 3: False, 4: True},
    "Stadtwerke München":     {1: True, 2: False, 3: False, 4: True},
    "Thüringer Energienetze": {1: True, 2: False, 3: False, 4: True},
    "LEW":                    {1: True, 2: True, 3: True, 4: True},
    "NetzeBW":                {1: True, 2: True, 3: True, 4: True},
    "Bayernwerk":             {1: False, 2: True, 3: True, 4: False},
    "EAM Netz":               {1: True, 2: True, 3: True, 4: True},
}

# Extracted 24-hour tariff pattern from heatmap (same for all DSOs)
# 00–23: NT/ST/HT allocation
MOD3_HOURLY_PATTERN = [
    "NT", "NT", "NT", "NT", "NT",
    "ST", "ST", "ST", "ST", "ST", "ST", "ST", "ST", "ST",
    "HT", "HT", "HT", "HT", "HT", "HT", "HT",
    "ST", "ST", "NT",
]  # length 24

def build_grid_fee_series(dso, num_days):
    """
    Build a full-year 15-min grid fee series (€/kWh) for the selected DSO.
    This just repeats the daily Mod3 pattern; we'll gate by quarter later.
    """
    prices = dso_tariffs[dso]  # ct/kWh
    day_96 = []
    for code in MOD3_HOURLY_PATTERN:
        grid_ct = prices[code]      # ct/kWh
        grid_eur = grid_ct / 100.0  # €/kWh
        day_96.extend([grid_eur] * 4)  # 4×15min per hour
    return np.tile(day_96, num_days)  # length = 96 * num_days

BASE_DATE = datetime.date(2023, 1, 1)

def get_quarter(month: int) -> int:
    if month <= 3:
        return 1
    elif month <= 6:
        return 2
    elif month <= 9:
        return 3
    else:
        return 4

def is_mod3_valid_day(day_idx: int, dso: str) -> bool:
    """Return True if Modul 3 applies on this day for the selected DSO."""
    date = BASE_DATE + datetime.timedelta(days=int(day_idx))
    q = get_quarter(date.month)
    return dso_quarter_valid.get(dso, {}).get(q, False)

def compute_da_indexed_daily_mod3(E, Pmax, quarters, da_day, grid_q_day, taxes, vat):
    """DA-indexed chronological charging with time-varying grid fee."""
    da_q = np.repeat(da_day / 1000.0, 4)  # €/kWh
    remain = E
    cost = 0.0
    emax = Pmax * 0.25  # kWh per quarter

    for q in quarters:
        if remain <= 1e-6:
            break
        e = min(emax, remain)
        price = apply_tariffs(da_q[q], grid_q_day[q], taxes, vat)
        cost += e * price
        remain -= e

    if remain > 1e-6:
        raise ValueError("Charging window too short to deliver requested energy.")
    return cost

def compute_optimised_daily_cost_mod3(E, Pmax, quarters, price_q, grid_q_day, taxes, vat):
    """
    Optimised charging using full cost (wholesale + time-varying grid + taxes).
    """
    pq = price_q / 1000.0  # €/kWh
    emax = Pmax * 0.25
    if len(quarters) * emax < E - 1e-6:
        raise ValueError("Charging window too short for requested energy.")

    full_price = np.array(
        [apply_tariffs(pq[i], grid_q_day[i], taxes, vat) for i in range(len(grid_q_day))]
    )

    sorted_q = sorted(quarters, key=lambda x: full_price[x])

    remain = E
    cost = 0.0
    for q in sorted_q:
        if remain <= 1e-6:
            break
        e = min(emax, remain)
        cost += e * full_price[q]
        remain -= e

    return cost

# =============================================================================
# FILE LOADER
# =============================================================================
def load_price_series_from_csv(upload, multiple):
    df = pd.read_csv(upload)
    if "price" in df.columns:
        series = df["price"].values
    else:
        series = df.select_dtypes(include="number").iloc[:, 0].values
    if len(series) % multiple != 0:
        st.error(
            f"Uploaded file length {len(series)} is not divisible by {multiple}. "
            "Check that you have full-year data at the correct resolution."
        )
        return None, 0
    return series.astype(float), len(series) // multiple

# =============================================================================
# SIDEBAR SETTINGS
# =============================================================================
st.sidebar.title("Simulation Settings")

st.sidebar.subheader("EV & Charging")
energy = st.sidebar.number_input("Energy per session (kWh)", 1.0, 200.0, 40.0, 1.0)
power = st.sidebar.number_input("Max charging power (kW)", 1.0, 50.0, 11.0, 0.5)

arrival = st.sidebar.slider("Arrival time [h]", 0.0, 24.0, 18.0, 0.25)
departure = st.sidebar.slider("Departure time [h]", 0.0, 24.0, 7.0, 0.25)

st.sidebar.subheader("Charging Frequency")
freq = st.sidebar.selectbox(
    "Pattern",
    [
        "Every day",
        "Every other day",
        "Weekdays only (Mon–Fri)",
        "Weekends only (Sat–Sun)",
        "Custom weekdays",
        "Custom: X sessions per week",
    ],
)
custom_days = None
sessions_week = None
if freq == "Custom weekdays":
    custom_days = st.sidebar.multiselect(
        "Select weekdays",
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        default=["Mon", "Wed", "Fri"],
    )
if freq == "Custom: X sessions per week":
    sessions_week = st.sidebar.number_input("Sessions per week", 1, 14, 3)

st.sidebar.subheader("Tariffs")
flat_price = st.sidebar.number_input("Flat all-in price (€/kWh)", 0.0, 1.0, 0.35, 0.01)
grid = st.sidebar.number_input("Grid network charges (€/kWh)", 0.0, 1.0, 0.11, 0.01)
taxes = st.sidebar.number_input("Taxes & levies (€/kWh)", 0.0, 1.0, 0.05, 0.01)
vat = st.sidebar.number_input("VAT (%)", 0.0, 30.0, 19.0, 1.0)

st.sidebar.subheader("§14a Time-Variable Grid Fee (Modul 3)")
enable_mod3 = st.sidebar.checkbox("Enable time-variable grid fee (§14a EnWG Modul 3)")

selected_dso = None
if enable_mod3:
    selected_dso = st.sidebar.selectbox(
        "Select DSO",
        list(dso_tariffs.keys()),
    )

st.sidebar.subheader("Market Data")
da_file = st.sidebar.file_uploader("DA hourly prices CSV (€/MWh)")
id_file = st.sidebar.file_uploader("ID 15-min prices CSV (€/MWh)")

# =============================================================================
# MARKET DATA LOADING
# =============================================================================
da_day_syn, id_day_syn = get_synthetic_daily_price_profiles()

if da_file is not None:
    da_series, da_days = load_price_series_from_csv(da_file, 24)
else:
    da_series, da_days = np.tile(da_day_syn, 365), 365

if id_file is not None:
    id_series, id_days = load_price_series_from_csv(id_file, 96)
else:
    id_series, id_days = np.tile(id_day_syn, 365), 365

if da_series is None or id_series is None:
    st.stop()

num_days = min(da_days, id_days)
da_year = da_series[: num_days * 24]
id_year = id_series[: num_days * 96]

st.sidebar.info(f"Using **{num_days} days** of DA & ID price data.")

grid_fee_series = None
if enable_mod3 and selected_dso is not None:
    grid_fee_series = build_grid_fee_series(selected_dso, num_days)

# =============================================================================
# OVERVIEW SECTION
# =============================================================================
st.markdown("<h2 id='overview'></h2>", unsafe_allow_html=True)
col_overview, col_snapshot = st.columns([2, 1])

with col_overview:
    st.markdown(
        """
        <div class="eon-card">
            <h3>Overview</h3>
            <p>
            This dashboard compares EV charging cost scenarios over a full year
            using hourly day-ahead and 15-minute intraday prices:
            </p>
            <ul>
                <li><b>Flat retail:</b> constant €/kWh, no market exposure</li>
                <li><b>DA-indexed:</b> hourly wholesale pass-through, no optimisation</li>
                <li><b>DA-optimised:</b> smart charging using DA prices</li>
                <li><b>DA+ID-optimised:</b> smart charging using min(DA, ID)</li>
            </ul>
            <p>
            Optionally, <b>§14a EnWG Modul 3</b> applies a time-variable grid fee
            (HT / ST / NT) by DSO. Modul-3 validity can differ by DSO and quarter
            (Q1–Q4) based on the regulatory design.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_snapshot:
    modul3_state = "ON" if (enable_mod3 and selected_dso) else "OFF"
    st.markdown(
        f"""
        <div class="eon-card">
            <h3>Scenario Snapshot</h3>
            <p><b>Energy / session:</b> {energy:.1f} kWh</p>
            <p><b>Max power:</b> {power:.1f} kW</p>
            <p><b>Arrival:</b> {arrival:.2f} h</p>
            <p><b>Departure:</b> {departure:.2f} h</p>
            <p><b>Flat price:</b> {flat_price:.2f} €/kWh</p>
            <p><b>§14a Modul 3:</b> {modul3_state}{' – ' + selected_dso if (enable_mod3 and selected_dso) else ''}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# MARKET PRICE CURVES
# =============================================================================
st.markdown("<h2 id='prices'></h2>", unsafe_allow_html=True)
st.markdown('<div class="eon-card">', unsafe_allow_html=True)
st.subheader("Market Price Curves (Full Year)")

timestamps = pd.date_range(start=str(BASE_DATE), periods=num_days * 96, freq="15min")
da_quarter = np.repeat(da_year, 4)[: num_days * 96]
id_quarter = id_year[: num_days * 96]
effective = np.minimum(da_quarter, id_quarter)

df_plot = pd.DataFrame(
    {
        "timestamp": timestamps,
        "DA (€/MWh)": da_quarter,
        "ID (€/MWh)": id_quarter,
        "Effective min(DA, ID) (€/MWh)": effective,
    }
)

fig = px.line(
    df_plot,
    x="timestamp",
    y=["DA (€/MWh)", "ID (€/MWh)", "Effective min(DA, ID) (€/MWh)"],
    title="Day-Ahead & Intraday Prices (zoom & pan)",
)
fig.update_layout(
    xaxis_rangeslider_visible=True,
    height=460,
    legend=dict(orientation="h", y=-0.22),
    plot_bgcolor="#020617",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e5e7eb"),
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor="rgba(148,163,184,0.3)")

st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# COST CALCULATION
# =============================================================================
quarters, _, _ = build_available_quarters(arrival, departure)
charging_days = compute_charging_days(freq, num_days, custom_days, sessions_week)
sessions = len(charging_days)

flat_annual = flat_price * energy * sessions

# Baseline (no Mod3) costs
da_index_annual = 0.0
da_opt_annual = 0.0
da_id_annual = 0.0

# With Mod3 costs
da_index_annual_mod3 = 0.0
da_opt_annual_mod3 = 0.0
da_id_annual_mod3 = 0.0

try:
    for d in charging_days:
        da_day = da_year[d * 24 : (d + 1) * 24]
        id_day = id_year[d * 96 : (d + 1) * 96]
        da_q_day = np.repeat(da_day, 4)
        eff_day = np.minimum(da_q_day, id_day)

        # --- Baseline constant grid fee ---
        c_da_index = compute_da_indexed_baseline_daily(
            energy, power, quarters, da_day, grid, taxes, vat
        )
        c_da_opt = compute_optimised_daily_cost(
            energy, power, quarters, da_q_day, grid, taxes, vat
        )
        c_da_id = compute_optimised_daily_cost(
            energy, power, quarters, eff_day, grid, taxes, vat
        )

        da_index_annual += c_da_index
        da_opt_annual += c_da_opt
        da_id_annual += c_da_id

        # --- With Modul 3 (if enabled and valid in this quarter for this DSO) ---
        if enable_mod3 and grid_fee_series is not None and selected_dso is not None and is_mod3_valid_day(d, selected_dso):
            grid_q_day = grid_fee_series[d * 96 : (d + 1) * 96]

            c_da_index_mod3 = compute_da_indexed_daily_mod3(
                energy, power, quarters, da_day, grid_q_day, taxes, vat
            )
            c_da_opt_mod3 = compute_optimised_daily_cost_mod3(
                energy, power, quarters, da_q_day, grid_q_day, taxes, vat
            )
            c_da_id_mod3 = compute_optimised_daily_cost_mod3(
                energy, power, quarters, eff_day, grid_q_day, taxes, vat
            )
        else:
            # Not in a Mod3-valid quarter or feature off:
            c_da_index_mod3 = c_da_index
            c_da_opt_mod3 = c_da_opt
            c_da_id_mod3 = c_da_id

        da_index_annual_mod3 += c_da_index_mod3
        da_opt_annual_mod3 += c_da_opt_mod3
        da_id_annual_mod3 += c_da_id_mod3

except ValueError as e:
    st.error(str(e))

# =============================================================================
# RESULTS – 3 BLOCKS + BAR CHARTS
# =============================================================================
st.markdown("<h2 id='results'></h2>", unsafe_allow_html=True)
st.markdown('<div class="eon-card">', unsafe_allow_html=True)

# 1) Annual costs WITHOUT Modul 3
st.subheader("1️⃣ Annual Cost (WITHOUT §14a Modul 3)")
df_before = pd.DataFrame(
    {
        "Scenario": [
            "Flat retail (fixed price)",
            "DA-indexed",
            "DA-optimised",
            "DA+ID-optimised",
        ],
        "Annual cost (€)": [
            round(flat_annual),
            round(da_index_annual),
            round(da_opt_annual),
            round(da_id_annual),
        ],
    }
)
st.table(df_before)

# 2) Annual costs WITH Modul 3
if enable_mod3 and grid_fee_series is not None and selected_dso is not None:
    st.subheader("2️⃣ Annual Cost (WITH §14a Modul 3 – by DSO & quarter)")
    df_after = pd.DataFrame(
        {
            "Scenario": [
                "Flat retail (unchanged)",
                f"DA-indexed + Modul 3 ({selected_dso})",
                f"DA-optimised + Modul 3 ({selected_dso})",
                f"DA+ID-optimised + Modul 3 ({selected_dso})",
            ],
            "Annual cost (€)": [
                round(flat_annual),
                round(da_index_annual_mod3),
                round(da_opt_annual_mod3),
                round(da_id_annual_mod3),
            ],
        }
    )
    st.table(df_after)

    # 3) Modul 3 incremental savings
    st.subheader("3️⃣ Additional Savings FROM §14a Modul 3 (vs constant grid fee)")
    df_mod3 = pd.DataFrame(
        {
            "Scenario": [
                "DA-indexed → Modul 3",
                "DA-optimised → Modul 3",
                "DA+ID-optimised → Modul 3",
            ],
            "Extra savings (€ / year)": [
                round(da_index_annual - da_index_annual_mod3),
                round(da_opt_annual - da_opt_annual_mod3),
                round(da_id_annual - da_id_annual_mod3),
            ],
        }
    )
    st.table(df_mod3)

    # -----------------------------
    # BAR CHARTS FOR RESULTS
    # -----------------------------
    st.markdown("### 📊 Annual Cost Comparison (Before vs After Modul 3)")

    scenarios = ["DA-indexed", "DA-optimised", "DA+ID-optimised"]
    before_vals = [da_index_annual, da_opt_annual, da_id_annual]
    after_vals = [da_index_annual_mod3, da_opt_annual_mod3, da_id_annual_mod3]

    fig_cost = go.Figure(
        data=[
            go.Bar(name="Before Modul 3", x=scenarios, y=before_vals),
            go.Bar(name="With Modul 3", x=scenarios, y=after_vals),
        ]
    )
    fig_cost.update_layout(
        barmode="group",
        height=450,
        plot_bgcolor="#020617",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        yaxis_title="Annual Cost (€)",
        xaxis_title="Customer Type",
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown("### 📉 Additional Savings from Modul 3")

    savings_mod3 = [
        da_index_annual - da_index_annual_mod3,
        da_opt_annual - da_opt_annual_mod3,
        da_id_annual - da_id_annual_mod3,
    ]

    fig_savings = go.Figure(
        data=[go.Bar(x=scenarios, y=savings_mod3)]
    )
    fig_savings.update_layout(
        height=400,
        plot_bgcolor="#020617",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        yaxis_title="Modul 3 Extra Savings (€ / year)",
        xaxis_title="Customer Type",
    )
    st.plotly_chart(fig_savings, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# DETAILS
# =============================================================================
st.markdown("<h2 id='details'></h2>", unsafe_allow_html=True)
st.markdown('<div class="eon-card">', unsafe_allow_html=True)
st.subheader("Model Assumptions & Notes")
st.markdown(
    f"""
- Resolution: **15 minutes (96 slots per day)**  
- DA prices: **hourly** in €/MWh; ID prices: **15-min** in €/MWh  
- If no CSV files are uploaded, synthetic DA/ID profiles are repeated over {num_days} days.  
- Effective wholesale price for DA+ID optimisation: **min(DA, ID)** per 15-min slot.  
- Wholesale-based customers pay:  
  **(wholesale energy price + grid network charges + taxes & levies) × (1 + VAT)**  
- Flat retail case: all-in constant €/kWh (no extra VAT applied in the model).  
- Charging pattern: **{freq}**, giving **{sessions} sessions** across {num_days} days.  
- §14a EnWG Modul 3:  
  - Uses DSO-specific **HT/ST/NT grid prices (ct/kWh)**.  
  - Hourly pattern from the graphic is applied with 15-min resolution.  
  - Modul-3 validity by DSO & quarter is configurable and currently set as you provided.  
"""
)
st.markdown("</div>", unsafe_allow_html=True)
