import os
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# 0) Page + Cute Pastel Rainbow Theme
# =========================================================
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
  background: rgba(255,255,255,0.75);
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
  margin-bottom: 6px;
}

.small-note {
  font-size: 0.92rem;
  opacity: 0.9;
}
</style>
"""

st.set_page_config(page_title="🌈 서울 기온 분석 (귀염뽀짝)", layout="wide")
st.markdown(PASTEL_CSS, unsafe_allow_html=True)


# =========================================================
# 1) Constants / Paths
# =========================================================
BASE_TEMP_PATH = "data/base_seoul_temp.csv"
CSAT_PATH = "data/csat_dates.csv"

REQUIRED_COLS = ["날짜", "지점", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
METRICS = ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]

SOLAR_TERMS_KO = [
    "입춘","우수","경칩","춘분","청명","곡우",
    "입하","소만","망종","하지","소서","대서",
    "입추","처서","백로","추분","한로","상강",
    "입동","소설","대설","동지","소한","대한"
]


# =========================================================
# 2) CSV Reader (flexible encoding)
# =========================================================
def read_csv_flexible(path: str) -> pd.DataFrame:
    raw = open(path, "rb").read()
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            text = raw.decode(enc)
            return pd.read_csv(io.StringIO(text))
        except Exception:
            continue
    return pd.read_csv(path, encoding="cp949")


# =========================================================
# 3) Load & Clean Temperature Data
# =========================================================
def load_temperature_csv(path: str = BASE_TEMP_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"기온 데이터 파일을 찾을 수 없습니다: {path}")

    df = read_csv_flexible(path)

    # 컬럼이 정확히 있으면 그대로, 아니면 '앞 5열'로 강제 매핑(엑셀 변형 대응)
    if not all(c in df.columns for c in REQUIRED_COLS):
        if df.shape[1] >= 5:
            df = df.iloc[:, :5].copy()
            df.columns = REQUIRED_COLS
        else:
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            raise ValueError(f"필수 컬럼 누락: {missing}")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["지점"] = pd.to_numeric(df["지점"], errors="coerce")

    for c in METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["날짜", "지점"]).sort_values("날짜").reset_index(drop=True)
    df["지점"] = df["지점"].astype(int)

    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    df["일"] = df["날짜"].dt.day
    df["MMDD"] = df["날짜"].dt.strftime("%m-%d")

    return df


def safe_metric_bounds_check(df: pd.DataFrame) -> pd.DataFrame:
    lo, hi = -40, 45
    flags = []
    for col in METRICS:
        flags.append((df[col] < lo) | (df[col] > hi))
    df["is_outlier_flag"] = np.logical_or.reduce(flags)
    return df


# =========================================================
# 4) Load CSAT Dates (Korean/English columns supported)
# =========================================================
def load_csat_dates(path: str = CSAT_PATH) -> pd.DataFrame:
    """
    지원 형식:
      (A) academic_year, exam_date, note
      (B) 시험연도, 수능일, 비고
    반환: academic_year, exam_date, note
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"수능 날짜 파일을 찾을 수 없습니다: {path}")

    csat = read_csv_flexible(path)

    rename = {}
    if "시험연도" in csat.columns: rename["시험연도"] = "academic_year"
    if "수능일" in csat.columns:   rename["수능일"]   = "exam_date"
    if "비고" in csat.columns:     rename["비고"]     = "note"

    csat = csat.rename(columns=rename)

    if "exam_date" not in csat.columns:
        raise ValueError("csat_dates.csv에 날짜 컬럼(exam_date 또는 수능일)이 없습니다.")

    if "academic_year" not in csat.columns:
        csat["academic_year"] = pd.to_datetime(csat["exam_date"], errors="coerce").dt.year

    if "note" not in csat.columns:
        csat["note"] = ""

    csat["exam_date"] = pd.to_datetime(csat["exam_date"], errors="coerce")
    csat["academic_year"] = pd.to_numeric(csat["academic_year"], errors="coerce")

    csat = csat.dropna(subset=["exam_date"]).sort_values("exam_date").reset_index(drop=True)
    return csat[["academic_year", "exam_date", "note"]]


# =========================================================
# 5) Climatology (annual daily cycle) + overlays
# =========================================================
def build_climatology(df: pd.DataFrame, metric: str, base_start: int, base_end: int) -> pd.DataFrame:
    base = df[(df["연도"] >= base_start) & (df["연도"] <= base_end)].copy()
    base = base[base["MMDD"] != "02-29"]
    clim = base.groupby("MMDD", as_index=False)[metric].mean()
    clim["x"] = pd.to_datetime("2001-" + clim["MMDD"], format="%Y-%m-%d")  # dummy non-leap year
    clim = clim.sort_values("x")
    clim.rename(columns={metric: "climatology"}, inplace=True)
    return clim[["x", "MMDD", "climatology"]]


def build_year_overlay(df: pd.DataFrame, metric: str, year: int) -> pd.DataFrame:
    sub = df[df["연도"] == year].copy()
    sub = sub[sub["MMDD"] != "02-29"]
    sub["x"] = pd.to_datetime("2001-" + sub["MMDD"], format="%Y-%m-%d")
    sub = sub.sort_values("x")
    sub.rename(columns={metric: "value"}, inplace=True)
    sub["연도"] = year
    return sub[["x", "MMDD", "value", "연도"]]


# =========================================================
# 6) 24 Solar Terms (Skyfield approximate; robust fallback)
# =========================================================
@st.cache_data(show_spinner=False)
def compute_solar_terms_for_year_approx(year: int) -> pd.DataFrame:
    """
    Returns columns: [term, date]
    Strategy:
      - Use skyfield almanac to find ecliptic longitude bin changes (15° steps).
      - Rotate mapping so the term closest to Feb 4 becomes 입춘.
    This is for visualization (approx), not almanac-grade.
    """
    try:
        from skyfield.api import load
        from skyfield import almanac

        ts = load.timescale()
        eph = load("de421.bsp")

        t0 = ts.utc(year, 1, 1)
        t1 = ts.utc(year, 12, 31, 23, 59, 59)

        f = almanac.sun_ecliptic_longitude(eph)

        def term_index(t):
            lon = f(t).degrees % 360.0
            return np.floor(lon / 15.0).astype(int)

        times, idx = almanac.find_discrete(t0, t1, term_index)

        # pick first occurrence of each index
        seen = set()
        events = []
        for t, i in zip(times, idx):
            ii = int(i)
            if ii not in seen:
                seen.add(ii)
                events.append((ii, t.utc_datetime().date()))

        if not events:
            return pd.DataFrame(columns=["term", "date"])

        events.sort(key=lambda x: x[0])
        tmp = pd.DataFrame(events, columns=["idx", "date"])

        # pivot to align 입춘 with ~ Feb 4
        target = datetime(year, 2, 4).date()
        tmp["delta"] = tmp["date"].apply(lambda d: abs((d - target).days))
        pivot_idx = int(tmp.sort_values("delta").iloc[0]["idx"])

        mapping = {}
        for k, name in enumerate(SOLAR_TERMS_KO):
            mapping[(pivot_idx + k) % 24] = name

        tmp["term"] = tmp["idx"].map(mapping)
        tmp = tmp.dropna(subset=["term"]).sort_values("date")

        return tmp[["term", "date"]].reset_index(drop=True)

    except Exception:
        # fallback: empty -> UI will show warning
        return pd.DataFrame(columns=["term", "date"])


def get_term_metric_values(df: pd.DataFrame, year: int, metric: str, window_days: int = 0) -> pd.DataFrame:
    terms = compute_solar_terms_for_year_approx(year)
    if terms.empty:
        return pd.DataFrame(columns=["year", "term", "term_date", "value"])

    df_year = df[df["연도"] == year].set_index("날짜")

    rows = []
    for _, r in terms.iterrows():
        d = pd.to_datetime(r["date"])

        if window_days <= 0:
            val = df_year[metric].get(d, np.nan)
        else:
            w0 = d - pd.Timedelta(days=window_days)
            w1 = d + pd.Timedelta(days=window_days)
            val = df_year.loc[w0:w1][metric].mean()

        rows.append((year, r["term"], d.date().isoformat(), val))

    out = pd.DataFrame(rows, columns=["year", "term", "term_date", "value"])
    return out


# =========================================================
# 7) Header / File checks
# =========================================================
st.markdown(
    """
<div class="rainbow-card">
  <div style="font-size:1.6rem; font-weight:800;">🌈 서울 기온 분석 놀이터 (귀염뽀짝)</div>
  <div class="small-note">
    🎀 data 폴더의 <b>base_seoul_temp.csv</b> + <b>csat_dates.csv</b> 두 파일만으로 작동합니다.
  </div>
  <div style="margin-top:10px;">
    <span class="badge">📌 같은 날짜(MM-DD) 분포</span>
    <span class="badge">🧪 수능날</span>
    <span class="badge">🍀 연중 클라이마톨로지</span>
    <span class="badge">🌿 24절기</span>
    <span class="badge">🧼 품질점검</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

file_ok_base = os.path.exists(BASE_TEMP_PATH)
file_ok_csat = os.path.exists(CSAT_PATH)
st.caption(f"📂 파일 확인: base_seoul_temp.csv={file_ok_base} | csat_dates.csv={file_ok_csat}")

if not file_ok_base:
    st.error(f"필수 파일이 없습니다: {BASE_TEMP_PATH}")
    st.stop()


# =========================================================
# 8) Sidebar Controls
# =========================================================
with st.sidebar:
    st.header("⚙️ 설정")
    metric = st.selectbox("지표 선택 🌡️", METRICS, index=0)

    st.markdown("---")
    st.caption("📌 입력 파일")
    st.code(f"{BASE_TEMP_PATH}\n{CSAT_PATH}", language="text")


# =========================================================
# 9) Load Data
# =========================================================
df = load_temperature_csv(BASE_TEMP_PATH)
df = safe_metric_bounds_check(df)

min_date = df["날짜"].min()
max_date = df["날짜"].max()


# =========================================================
# 10) Tabs
# =========================================================
tabs = st.tabs([
    "📌 지정일 비교",
    "🧪 CSAT(수능)",
    "🍀 연중 클라이마톨로지",
    "🌿 24절기",
    "🧼 데이터 품질"
])


# ---------------------------------------------------------
# Tab 1: 지정일 vs 동일 MM-DD 분포
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("📌 지정일이 같은 날짜(MM-DD) 역사와 비교하면?")
    default_date = max_date
    picked = st.date_input("날짜 선택 📅", value=default_date.date(), min_value=min_date.date(), max_value=max_date.date())
    picked = pd.to_datetime(picked)

    same_mmdd = df[df["MMDD"] == picked.strftime("%m-%d")].copy()
    actual = df[df["날짜"] == picked][["날짜", metric]]

    if actual.empty:
        st.warning("선택한 날짜가 데이터에 없습니다.")
    else:
        actual_val = float(actual.iloc[0][metric])
        hist_mean = float(same_mmdd[metric].mean()) if not same_mmdd.empty else np.nan
        diff = actual_val - hist_mean if pd.notnull(hist_mean) else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📅 선택일", picked.date().isoformat())
        c2.metric("🌡️ 당일 값", f"{actual_val:.1f} ℃")
        c3.metric("🧁 동일 MM-DD 평균", f"{hist_mean:.1f} ℃" if pd.notnull(hist_mean) else "N/A")
        c4.metric("🔥/🧊 평균 대비", f"{diff:+.1f} ℃" if pd.notnull(diff) else "N/A")

        fig = px.histogram(
            same_mmdd.dropna(subset=[metric]),
            x=metric,
            nbins=30,
            title=f"📊 {picked.strftime('%m-%d')}의 분포 (역대 동일 날짜)",
        )
        fig.add_vline(x=actual_val, line_width=3)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # time series around that day for context (optional nice view)
        st.subheader("🪄 주변 시계열(±30일)로 분위기 보기")
        w0 = picked - pd.Timedelta(days=30)
        w1 = picked + pd.Timedelta(days=30)
        around = df[(df["날짜"] >= w0) & (df["날짜"] <= w1)].copy()
        fig2 = px.line(around, x="날짜", y=metric, title=f"{picked.date()} 전후 30일 {metric}")
        fig2.add_vline(x=picked, line_width=2, line_dash="dash")
        fig2.update_layout(hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------
# Tab 2: CSAT(수능)
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("🧪 수학능력시험(CSAT) 날짜별 기온")

    if not file_ok_csat:
        st.warning(f"{CSAT_PATH} 파일이 없어 수능 분석을 표시할 수 없습니다.")
    else:
        try:
            csat = load_csat_dates(CSAT_PATH)
        except Exception as e:
            st.error(f"csat_dates.csv 로드 실패: {e}")
            csat = pd.DataFrame(columns=["academic_year", "exam_date", "note"])

        if csat.empty:
            st.info("csat_dates.csv가 비어 있습니다.")
        else:
            temp_idx = df.set_index("날짜")
            csat["value"] = csat["exam_date"].map(lambda d: temp_idx[metric].get(d, np.nan))

            st.markdown("<div class='small-note'>📎 수능일이 데이터에 없는 경우 value가 비어 보일 수 있습니다.</div>", unsafe_allow_html=True)
            st.dataframe(csat, use_container_width=True)

            fig = px.line(
                csat, x="academic_year", y="value", markers=True,
                title=f"🎓 학년도별 수능일 {metric}"
            )
            fig.update_layout(hovermode="x unified", xaxis_title="학년도", yaxis_title="℃")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.box(
                csat.dropna(subset=["value"]), y="value",
                title=f"📦 수능일 {metric} 분포"
            )
            fig2.update_layout(hovermode="x unified", yaxis_title="℃")
            st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------
# Tab 3: 연중(일별) 클라이마톨로지 + 연도 오버레이
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("🍀 연중(일별) 클라이마톨로지(평년곡선) + 선택 연도 오버레이")

    years = sorted(df["연도"].dropna().unique().tolist())

    # 기준기간 선택: 기본값 1991-2020(가능할 때)
    default_start = 1991 if 1991 in years else years[0]
    default_end = 2020 if 2020 in years else years[-1]
    if default_start > default_end:
        default_start, default_end = years[0], years[-1]

    base_start, base_end = st.select_slider(
        "평년(기준기간) 선택 🧁",
        options=years,
        value=(default_start, default_end)
    )

    overlay_years = st.multiselect(
        "오버레이할 연도(복수 선택 가능) 🎀",
        options=years,
        default=[years[-1]]
    )

    clim = build_climatology(df, metric, int(base_start), int(base_end))

    # Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clim["x"],
        y=clim["climatology"],
        mode="lines",
        name=f"평년곡선({base_start}-{base_end})",
        line=dict(width=4)
    ))

    for y in overlay_years:
        ov = build_year_overlay(df, metric, int(y))
        fig.add_trace(go.Scatter(
            x=ov["x"],
            y=ov["value"],
            mode="lines",
            name=f"{y}년",
            line=dict(width=2)
        ))

    # x axis tick format
    fig.update_layout(
        title=f"🌡️ 연중 {metric}: 평년곡선 + 연도 오버레이",
        xaxis_title="연중(월-일)",
        yaxis_title="℃",
        hovermode="x unified",
    )
    fig.update_xaxes(tickformat="%m-%d")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='small-note'>📝 참고: 2/29는 비교 안정성을 위해 제외했습니다.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Tab 4: 24절기 (박스플롯 + 히트맵)
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("🌿 24절기 기온 비교 (Skyfield 근사)")

    years = sorted(df["연도"].dropna().unique().tolist())
    year_range = st.slider(
        "연도 범위 🗓️",
        min_value=min(years),
        max_value=max(years),
        value=(min(years), max(years))
    )

    window = st.select_slider("절기값 계산 윈도(±일) 🍡", options=[0, 1, 2, 3, 5, 7], value=0)

    rows = []
    for y in range(year_range[0], year_range[1] + 1):
        tdf = get_term_metric_values(df, y, metric, window_days=int(window))
        if not tdf.empty:
            rows.append(tdf)

    if not rows:
        st.warning("절기 계산에 실패했거나(특히 skyfield 로딩 문제) 해당 연도 데이터가 부족합니다.")
        st.info("Streamlit Cloud에서 skyfield 설치/네트워크가 제한되면 절기 계산이 실패할 수 있습니다.")
    else:
        long = pd.concat(rows, ignore_index=True)
        long = long.dropna(subset=["value"]).copy()

        # ensure term order
        long["term"] = pd.Categorical(long["term"], categories=SOLAR_TERMS_KO, ordered=True)
        long = long.sort_values(["year", "term"])

        st.markdown(
            f"<div class='small-note'>📌 절기값: {'당일' if window==0 else f'±{window}일 평균'} 기준</div>",
            unsafe_allow_html=True
        )

        # Boxplot by term
        fig_box = px.box(
            long,
            x="term",
            y="value",
            points="outliers",
            title=f"📦 24절기별 {metric} 분포 (연도 {year_range[0]}–{year_range[1]})",
        )
        fig_box.update_layout(xaxis_tickangle=-45, hovermode="x unified", xaxis_title="절기", yaxis_title="℃")
        st.plotly_chart(fig_box, use_container_width=True)

        # Heatmap (year x term)
        pivot = long.pivot_table(index="year", columns="term", values="value", aggfunc="mean")
        pivot = pivot.reindex(columns=SOLAR_TERMS_KO)

        fig_hm = px.imshow(
            pivot,
            aspect="auto",
            title=f"🧊 연도 × 24절기 히트맵: {metric}",
            labels=dict(x="절기", y="연도", color="℃"),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.caption("※ 24절기 날짜는 skyfield 기반 근사치(시각화 목적)입니다.")


# ---------------------------------------------------------
# Tab 5: 데이터 품질
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("🧼 데이터 품질 점검")

    st.write("1) 결측치 요약")
    missing = df[REQUIRED_COLS].isna().sum()
    st.dataframe(missing.to_frame("missing_count"), use_container_width=True)

    st.write("2) 물리적 범위(-40~45℃) 밖 값 플래그")
    outliers = df[df["is_outlier_flag"]].copy()
    st.markdown("<div class='small-note'>⚠️ 범위 밖 값은 데이터 오류가 아닐 수도 있으나, 점검 포인트로 표시합니다.</div>", unsafe_allow_html=True)

    st.metric("의심(outlier_flag) 행 수", f"{len(outliers):,}")

    if outliers.empty:
        st.success("범위 밖으로 플래그된 값이 없습니다.")
    else:
        st.dataframe(outliers[["날짜", "지점"] + METRICS].head(200), use_container_width=True)

    st.write("3) 중복(날짜+지점) 체크")
    dup_count = df.duplicated(subset=["날짜", "지점"]).sum()
    st.metric("중복 행 수", f"{dup_count:,}")

    st.write("4) 기본 통계")
    st.dataframe(df[METRICS].describe().T, use_container_width=True)
