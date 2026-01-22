import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, timedelta

# -------------------------
# Cute pastel rainbow style
# -------------------------
PASTEL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');

html, body, [class*="css"]  { font-family: 'Nunito', sans-serif; }

.stApp {
  background: linear-gradient(135deg,
    rgba(255, 182, 193, 0.35),
    rgba(255, 218, 185, 0.35),
    rgba(255, 255, 224, 0.35),
    rgba(193, 255, 193, 0.35),
    rgba(173, 216, 230, 0.35),
    rgba(216, 191, 216, 0.35)
  );
}

.rainbow-card {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(255,255,255,0.65);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}

.badge {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.7);
  border: 1px dashed rgba(120,120,120,0.25);
  margin-right: 6px;
}
</style>
"""

st.set_page_config(page_title="🌈 서울 기온 분석 (귀염뽀짝)", layout="wide")
st.markdown(PASTEL_CSS, unsafe_allow_html=True)

# -------------------------
# Data loading utilities
# -------------------------
REQUIRED_COLS = ["날짜", "지점", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]

def load_temperature_csv(uploaded_file=None, default_path=None) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(default_path)

    # normalize columns (exactly as you told)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    for c in ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # basic cleaning
    df = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    df["일"] = df["날짜"].dt.day
    df["MMDD"] = df["날짜"].dt.strftime("%m-%d")  # for climatology
    return df

def safe_metric_bounds_check(df: pd.DataFrame) -> pd.DataFrame:
    # flag obvious outliers (can tune)
    # Seoul: roughly -30~40C, but keep wide
    lo, hi = -40, 45
    flags = []
    for col in ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]:
        flags.append((df[col] < lo) | (df[col] > hi))
    df["is_outlier_flag"] = np.logical_or.reduce(flags)
    return df

# -------------------------
# Climatology (daily annual cycle) + overlay
# -------------------------
def build_climatology(df: pd.DataFrame, metric: str, base_start: int, base_end: int) -> pd.DataFrame:
    base = df[(df["연도"] >= base_start) & (df["연도"] <= base_end)].copy()
    # drop Feb 29 for stable MM-DD series
    base = base[base["MMDD"] != "02-29"]
    clim = base.groupby("MMDD", as_index=False)[metric].mean()
    clim["x"] = pd.to_datetime("2001-" + clim["MMDD"], format="%Y-%m-%d")  # dummy year (non-leap)
    clim = clim.sort_values("x")
    clim.rename(columns={metric: "climatology"}, inplace=True)
    return clim[["x", "MMDD", "climatology"]]

def build_year_overlay(df: pd.DataFrame, metric: str, year: int) -> pd.DataFrame:
    sub = df[df["연도"] == year].copy()
    sub = sub[sub["MMDD"] != "02-29"]
    sub["x"] = pd.to_datetime("2001-" + sub["MMDD"], format="%Y-%m-%d")
    sub = sub.sort_values("x")
    sub.rename(columns={metric: "value"}, inplace=True)
    return sub[["x", "MMDD", "value", "연도"]]

# -------------------------
# CSAT dates file
# -------------------------
def load_csat_dates(path: str = "data/csat_dates.csv") -> pd.DataFrame:
    csat = pd.read_csv(path)
    csat["exam_date"] = pd.to_datetime(csat["exam_date"], errors="coerce")
    return csat.dropna(subset=["exam_date"]).sort_values("exam_date")

# -------------------------
# 24 solar terms (approx) using skyfield
# - Implementation note: this is a *rough* approach; you asked for approximation.
# - If skyfield fails, fallback to None.
# -------------------------
SOLAR_TERMS_KO = [
    "입춘","우수","경칩","춘분","청명","곡우",
    "입하","소만","망종","하지","소서","대서",
    "입추","처서","백로","추분","한로","상강",
    "입동","소설","대설","동지","소한","대한"
]

@st.cache_data(show_spinner=False)
def compute_solar_terms_for_year_approx(year: int):
    """
    Returns: DataFrame with columns [term, date]
    Approximation strategy:
      - Use skyfield to find ecliptic longitude crossings at 15° increments (0..345)
      - Map them to Korean 24절기 names (starting near 입춘).
    This is "good enough for visualization", not for almanac-grade accuracy.
    """
    try:
        from skyfield.api import load
        from skyfield import almanac
        import pytz

        ts = load.timescale()
        eph = load('de421.bsp')  # small ephemeris; usually OK on Streamlit Cloud
        earth = eph['earth']
        sun = eph['sun']

        # define time window: Jan 1 ~ Dec 31 (plus small buffer)
        t0 = ts.utc(year, 1, 1)
        t1 = ts.utc(year, 12, 31, 23, 59, 59)

        # ecliptic longitude of the sun as seen from Earth
        f = almanac.sun_ecliptic_longitude(eph)

        # find all 15-degree boundaries
        # We'll sample by finding events with almanac.find_discrete on custom "bin"
        # Create function that returns the 24-step index (0..23)
        def term_index(t):
            lon = f(t).degrees % 360.0
            return np.floor(lon / 15.0).astype(int)

        times, idx = almanac.find_discrete(t0, t1, term_index)

        # take the first time each idx appears (boundary crossing moments)
        seen = set()
        events = []
        for t, i in zip(times, idx):
            if int(i) not in seen:
                seen.add(int(i))
                dt = t.utc_datetime().date()
                events.append((int(i), dt))

        # sort by idx
        events.sort(key=lambda x: x[0])

        # The mapping between idx and Korean term names depends on definition.
        # We'll rotate so that the term closest to early Feb becomes 입춘.
        df = pd.DataFrame(events, columns=["idx", "date"])

        # rotate by matching: 입춘 is usually around Feb 4.
        target = datetime(year, 2, 4).date()
        df["delta"] = df["date"].apply(lambda d: abs((d - target).days))
        pivot_idx = int(df.sort_values("delta").iloc[0]["idx"])

        # pivot_idx becomes 입춘
        mapping = {}
        for k, name in enumerate(SOLAR_TERMS_KO):
            mapping[(pivot_idx + k) % 24] = name

        df["term"] = df["idx"].map(mapping)
        df = df.dropna(subset=["term"]).sort_values("date")
        return df[["term", "date"]].reset_index(drop=True)

    except Exception:
        return pd.DataFrame(columns=["term", "date"])

def get_term_metric_values(df: pd.DataFrame, year: int, metric: str, window_days: int = 0) -> pd.DataFrame:
    """
    For each solar term date in 'year', compute metric on that date (or ±window average).
    Returns: [year, term, term_date, value]
    """
    terms = compute_solar_terms_for_year_approx(year)
    if terms.empty:
        return pd.DataFrame(columns=["year", "term", "term_date", "value"])

    out = []
    df_year = df[df["연도"] == year].set_index("날짜")

    for _, r in terms.iterrows():
        d = pd.to_datetime(r["date"])
        if window_days <= 0:
            val = df_year[metric].get(d, np.nan)
        else:
            w0 = d - pd.Timedelta(days=window_days)
            w1 = d + pd.Timedelta(days=window_days)
            val = df_year.loc[w0:w1][metric].mean()

        out.append((year, r["term"], d.date().isoformat(), val))

    return pd.DataFrame(out, columns=["year", "term", "term_date", "value"])

# -------------------------
# UI
# -------------------------
st.markdown(
    """
<div class="rainbow-card">
  <span class="badge">🌈 파스텔 레인보우</span>
  <span class="badge">🧸 귀염뽀짝</span>
  <span class="badge">📈 Plotly 인터랙티브</span>
  <span class="badge">❄️☀️ 비교 분석</span>
</div>
""",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ 설정")
    uploaded = st.file_uploader("같은 형식 CSV 업로드", type=["csv"])
    metric = st.selectbox("지표 선택", ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"])

# NOTE: default_path는 여러분 프로젝트의 기본 탑재 CSV 경로로 맞추세요.
DEFAULT_DATA_PATH = "data/base_.csv"

df = load_temperature_csv(uploaded_file=uploaded, default_path=DEFAULT_DATA_PATH)
df = safe_metric_bounds_check(df)

tabs = st.tabs(["📌 오늘/지정일 비교", "🧪 CSAT(수능)", "🍀 계절 추이(연중 클리마톨로지)", "🌿 24절기"])

# -------------------------
# Tab 1: 지정일 비교(기존 로직 유지 가정)
# -------------------------
with tabs[0]:
    st.subheader("📌 지정일이 같은 날짜(MM-DD) 역사와 비교하면?")
    default_date = df["날짜"].max()
    picked = st.date_input("날짜 선택", value=default_date.date())
    picked = pd.to_datetime(picked)

    same_mmdd = df[df["MMDD"] == picked.strftime("%m-%d")].copy()
    actual = df[df["날짜"] == picked][["날짜", metric]]
    if actual.empty:
        st.warning("선택한 날짜가 데이터에 없습니다.")
    else:
        actual_val = float(actual.iloc[0][metric])
        hist_mean = float(same_mmdd[metric].mean())
        diff = actual_val - hist_mean

        c1, c2, c3 = st.columns(3)
        c1.metric("선택일", picked.date().isoformat())
        c2.metric("당일 값", f"{actual_val:.1f} ℃")
        c3.metric("동일 MM-DD 평년(평균)", f"{hist_mean:.1f} ℃", delta=f"{diff:+.1f} ℃")

        fig = px.histogram(
            same_mmdd, x=metric, nbins=30,
            title=f"📊 {picked.strftime('%m-%d')}의 분포 (역대 동일 날짜)"
        )
        fig.add_vline(x=actual_val, line_width=3)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Tab 2: CSAT
# -------------------------
with tabs[1]:
    st.subheader("🧪 수학능력시험(CSAT) 날짜별 기온")
    try:
        csat = load_csat_dates("csat_dates.csv")
    except Exception:
        st.error("csat_dates.csv를 찾을 수 없습니다. 프로젝트 루트에 두세요.")
        csat = pd.DataFrame(columns=["academic_year", "exam_date", "note"])

    if not csat.empty:
        # Join with temperature data
        temp_idx = df.set_index("날짜")
        csat["value"] = csat["exam_date"].map(lambda d: temp_idx[metric].get(d, np.nan))
        st.dataframe(csat, use_container_width=True)

        fig = px.line(
            csat, x="academic_year", y="value", markers=True,
            title=f"🎓 학년도별 수능일 {metric}"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(
            csat.dropna(subset=["value"]), y="value",
            title=f"📦 수능일 {metric} 분포"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("csat_dates.csv가 비어 있습니다.")

# -------------------------
# Tab 3: Seasonal (annual cycle climatology + overlay)
# -------------------------
with tabs[2]:
    st.subheader("🍀 연중(일별) 클라이마톨로지(평년곡선) + 연도 오버레이")
    years = sorted(df["연도"].dropna().unique().tolist())
    base_start, base_end = st.select_slider(
        "평년(기준기간) 선택",
        options=years,
        value=(max(min(years), 1991), min(max(years), 2020))
    )

    overlay_years = st.multiselect(
        "오버레이할 연도(복수 선택 가능)",
        options=years,
        default=[df["연도"].max()]
    )

    clim = build_climatology(df, metric, int(base_start), int(base_end))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clim["x"], y=clim["climatology"],
        mode="lines",
        name=f"평년곡선({base_start}-{base_end})",
    ))

    for y in overlay_years:
        ov = build_year_overlay(df, metric, int(y))
        fig.add_trace(go.Scatter(
            x=ov["x"], y=ov["value"],
            mode="lines",
            name=f"{y}년",
        ))

    fig.update_layout(
        title=f"🌡️ 연중 {metric}: 평년곡선 + 연도 오버레이",
        xaxis_title="연중(월-일)",
        yaxis_title="℃",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Tab 4: 24 solar terms (boxplot + year heatmap)
# -------------------------
with tabs[3]:
    st.subheader("🌿 24절기 기온 비교")
    years = sorted(df["연도"].dropna().unique().tolist())
    year_range = st.slider("연도 범위", min_value=min(years), max_value=max(years), value=(min(years), max(years)))
    window = st.select_slider("절기값 계산 윈도(±일)", options=[0,1,2,3,5,7], value=0)

    # Build long dataframe
    rows = []
    for y in range(year_range[0], year_range[1] + 1):
        tdf = get_term_metric_values(df, y, metric, window_days=int(window))
        if not tdf.empty:
            rows.append(tdf)

    if not rows:
        st.warning("절기 계산에 실패했거나 해당 연도 데이터가 부족합니다(특히 skyfield 로딩 실패 가능).")
    else:
        long = pd.concat(rows, ignore_index=True)
        long = long.dropna(subset=["value"])

        # Boxplot by term
        fig_box = px.box(
            long,
            x="term", y="value",
            title=f"📦 24절기별 {metric} 분포 (±{window}일 {'평균' if window>0 else '당일'})"
        )
        fig_box.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_box, use_container_width=True)

        # Heatmap year x term
        pivot = long.pivot_table(index="year", columns="term", values="value", aggfunc="mean")
        # keep term order
        pivot = pivot.reindex(columns=SOLAR_TERMS_KO)
        fig_hm = px.imshow(
            pivot,
            aspect="auto",
            title=f"🧊 연도 × 24절기 히트맵: {metric} (평균)",
            labels=dict(x="절기", y="연도", color="℃")
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.caption("※ 24절기 날짜는 skyfield 기반 근사치(시각화 목적)입니다.")
