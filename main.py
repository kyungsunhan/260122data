import io
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="서울 일 기온 분석", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
EXPECTED_COLS = ["날짜", "지점", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
STATION_SEOUL = 108

def read_csv_flexible(file_like) -> pd.DataFrame:
    """
    Streamlit 업로드 파일/로컬 파일 모두 처리.
    인코딩은 utf-8-sig, cp949, euc-kr 순으로 시도.
    """
    raw = file_like.read() if hasattr(file_like, "read") else open(file_like, "rb").read()

    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            text = raw.decode(enc)
            df = pd.read_csv(io.StringIO(text))
            return df
        except Exception:
            continue
    # 마지막으로 바이너리로 pandas에 직접 시도
    return pd.read_csv(io.BytesIO(raw), encoding="cp949")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    열 이름이 깨졌거나 Unnamed가 있는 경우라도,
    '5개 컬럼'이 존재하고 순서가 맞다면 EXPECTED_COLS로 강제 매핑.
    """
    # 완전 일치 시 그대로
    if all(c in df.columns for c in EXPECTED_COLS):
        df = df[EXPECTED_COLS].copy()
        return df

    # 열이 5개면 순서대로 매핑(사용자 제공 스펙 기반)
    if df.shape[1] >= 5:
        df = df.iloc[:, :5].copy()
        df.columns = EXPECTED_COLS
        return df

    raise ValueError("CSV 컬럼이 5개 미만입니다. 형식(날짜, 지점, 평균/최저/최고기온)이 맞는지 확인해 주세요.")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    # 날짜 파싱: YYYYMMDD 또는 YYYY-MM-DD 모두 허용
    # (문자/공백 행은 NaT로 떨어짐)
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce", format=None)

    # 지점/기온 수치 변환
    df["지점"] = pd.to_numeric(df["지점"], errors="coerce")
    for c in ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 빈 행/설명 행 제거: 날짜 또는 지점 결측 제거
    df = df.dropna(subset=["날짜", "지점"]).copy()

    # 지점 int로 정리(결측 제거 후)
    df["지점"] = df["지점"].astype(int)

    # 중복 제거(동일 날짜+지점이 여러 번 있을 수 있으므로 마지막 값 유지)
    df = df.sort_values("날짜").drop_duplicates(subset=["날짜", "지점"], keep="last")

    # 파생 변수
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    df["일"] = df["날짜"].dt.day
    df["월일"] = df["날짜"].dt.strftime("%m-%d")

    return df

def merge_datasets(base_df: pd.DataFrame, uploaded_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    all_df = [base_df] + uploaded_dfs
    merged = pd.concat(all_df, ignore_index=True)
    merged = merged.sort_values("날짜").drop_duplicates(subset=["날짜", "지점"], keep="last")
    return merged

def day_of_year_comparison(df: pd.DataFrame, target_date: pd.Timestamp, station: int, metric: str):
    """
    같은 '월-일' 기준으로 과거 모든 연도의 분포와 비교:
    - 해당일 값
    - 과거 평균/중앙값/표준편차
    - 백분위(퍼센타일)
    - (정렬) 몇 번째로 추운/더운지
    """
    target_md = target_date.strftime("%m-%d")

    sub = df[df["지점"] == station].copy()
    # 같은 월일의 역사 데이터
    hist = sub[sub["월일"] == target_md].copy()

    # 해당 날짜의 실제 값
    day_row = sub[sub["날짜"] == target_date]
    if day_row.empty:
        return None, hist, target_md

    x = float(day_row.iloc[0][metric])

    hist_values = hist[metric].dropna()
    if hist_values.empty:
        return None, hist, target_md

    mean = hist_values.mean()
    median = hist_values.median()
    std = hist_values.std(ddof=0)  # 모집단 표준편차(연도 표본 수 고정 고려)
    z = (x - mean) / std if std and std > 0 else None

    # 퍼센타일(해당 값이 분포에서 어느 위치인지)
    # "추움"을 metric 기준으로는 낮을수록 추움. 퍼센타일은 낮을수록 더 추움.
    pct = (hist_values <= x).mean() * 100.0

    # 랭킹(낮을수록 더 추움)
    rank_cold = int((hist_values < x).sum() + 1)
    rank_warm = int((hist_values > x).sum() + 1)
    n = int(hist_values.shape[0])

    stats = {
        "target_value": x,
        "mean": float(mean),
        "median": float(median),
        "std": float(std) if pd.notnull(std) else None,
        "zscore": float(z) if z is not None else None,
        "percentile_colder_or_equal": float(pct),  # 낮을수록 더 추운 쪽
        "rank_cold": rank_cold,  # 1이면 가장 추운 쪽
        "rank_warm": rank_warm,  # 1이면 가장 더운 쪽
        "n_years": n
    }
    return stats, hist, target_md

# -----------------------------
# Load base data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_base():
    base_path = "data/base_seoul_temp.csv"
    df0 = read_csv_flexible(base_path)
    return clean_data(df0)

st.title("서울 일 기온 분석 (기본 탑재 + 추가 업로드 병합)")

base_df = load_base()

# -----------------------------
# Upload area
# -----------------------------
st.sidebar.header("데이터 입력")
uploaded_files = st.sidebar.file_uploader(
    "같은 형식의 CSV 추가 업로드(복수 선택 가능)",
    type=["csv"],
    accept_multiple_files=True
)

uploaded_clean = []
if uploaded_files:
    for f in uploaded_files:
        try:
            d = read_csv_flexible(f)
            d = clean_data(d)
            uploaded_clean.append(d)
        except Exception as e:
            st.sidebar.error(f"파일 '{f.name}' 처리 실패: {e}")

df = merge_datasets(base_df, uploaded_clean) if uploaded_clean else base_df

# -----------------------------
# Controls
# -----------------------------
station = st.sidebar.number_input("지점 코드", value=STATION_SEOUL, step=1)
metric = st.sidebar.selectbox("비교 지표", ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"], index=0)

# 기본 날짜: 최신 날짜
max_date = df[df["지점"] == station]["날짜"].max()
min_date = df[df["지점"] == station]["날짜"].min()

date_input = st.sidebar.date_input(
    "기준 날짜(미지정 시 최신 날짜)",
    value=max_date.date() if pd.notnull(max_date) else None,
    min_value=min_date.date() if pd.notnull(min_date) else None,
    max_value=max_date.date() if pd.notnull(max_date) else None,
)

target_date = pd.to_datetime(date_input)

# -----------------------------
# Summary
# -----------------------------
colA, colB, colC, colD = st.columns(4)
colA.metric("총 행 수(병합 후)", f"{len(df):,}")
colB.metric("기간 시작", f"{min_date.date() if pd.notnull(min_date) else '-'}")
colC.metric("기간 종료", f"{max_date.date() if pd.notnull(max_date) else '-'}")
colD.metric("지점", f"{station}")

st.divider()

# -----------------------------
# Main analytics
# -----------------------------
stats, hist, md = day_of_year_comparison(df, target_date, station, metric)

if stats is None:
    st.error("선택한 날짜/지점에서 해당 지표 값을 찾을 수 없습니다. 데이터에 그 날짜가 있는지 확인해 주세요.")
    st.stop()

# 텍스트 요약(“얼마나 추웠/더웠는지”)
delta = stats["target_value"] - stats["mean"]
direction = "더웠습니다" if delta > 0 else "추웠습니다" if delta < 0 else "평년과 같았습니다"

# “같은 날짜(월-일)” 기준 비교 문장
summary = (
    f"선택한 날짜 **{target_date.date()} ({md})**의 **{metric}**은 "
    f"**{stats['target_value']:.1f}℃** 입니다.\n\n"
    f"같은 **{md}**의 역사적 분포(n={stats['n_years']}) 대비 "
    f"평균(**{stats['mean']:.1f}℃**)보다 **{abs(delta):.1f}℃** "
    f"{'높아' if delta>0 else '낮아' if delta<0 else ''} **{direction}**.\n\n"
    f"- 백분위(낮을수록 더 ‘추운’ 쪽): **{stats['percentile_colder_or_equal']:.1f}퍼센타일**\n"
    f"- 추운 순 랭크: **{stats['rank_cold']}/{stats['n_years']}** (1이 가장 추움)\n"
)

if stats["zscore"] is not None:
    summary += f"- Z-score: **{stats['zscore']:.2f}** (0=평균, 음수=평균보다 추움)\n"

st.markdown(summary)

# -----------------------------
# Plotly charts
# -----------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("1) 전체 기간 시계열")
    sub = df[df["지점"] == station].sort_values("날짜")
    fig_ts = px.line(sub, x="날짜", y=metric, title=f"{station} | {metric} (전체 기간)")
    fig_ts.update_layout(hovermode="x unified")
    st.plotly_chart(fig_ts, use_container_width=True)

with right:
    st.subheader("2) 같은 월-일 역사적 분포와 비교")
    hist_vals = hist[metric].dropna().sort_values()

    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=hist_vals,
        name=f"{md} 분포",
        boxpoints="all",
        jitter=0.3,
        pointpos=0
    ))
    fig_box.add_hline(
        y=stats["target_value"],
        line_dash="dash",
        annotation_text=f"{target_date.date()} 값: {stats['target_value']:.1f}℃",
        annotation_position="top left"
    )
    fig_box.update_layout(
        title=f"{md}의 {metric} 분포 vs 선택 날짜",
        yaxis_title=metric,
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.subheader("3) 같은 월-일 연도별 비교(라인/막대)")
hist_year = hist[["연도", metric]].dropna().sort_values("연도")
fig_year = px.bar(hist_year, x="연도", y=metric, title=f"{md} 연도별 {metric}")
fig_year.add_hline(
    y=stats["mean"],
    line_dash="dot",
    annotation_text=f"평균 {stats['mean']:.1f}℃",
    annotation_position="top left"
)
fig_year.add_hline(
    y=stats["target_value"],
    line_dash="dash",
    annotation_text=f"{target_date.date()} {stats['target_value']:.1f}℃",
    annotation_position="top left"
)
fig_year.update_layout(hovermode="x unified")
st.plotly_chart(fig_year, use_container_width=True)

# -----------------------------
# Data quality panel
# -----------------------------
with st.expander("데이터 품질(결측/범위) 확인"):
    sub = df[df["지점"] == station].copy()

    miss = sub[EXPECTED_COLS].isna().sum()
    st.write("결측치 개수(지점 필터 적용):")
    st.dataframe(miss.to_frame("missing_count"))

    # 물리적 범위 간단 체크(필요 시 조정 가능)
    # 서울 기준 대략적인 현실 범위: -35~45℃
    phys_warn = {}
    for c in ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]:
        bad = sub[(sub[c] < -35) | (sub[c] > 45)][["날짜", "지점", c]].dropna()
        phys_warn[c] = len(bad)
        if len(bad) > 0:
            st.warning(f"{c}: 물리적 범위 밖 추정 값 {len(bad)}건")
            st.dataframe(bad.head(50))

    if all(v == 0 for v in phys_warn.values()):
        st.success("물리적 범위 밖으로 의심되는 값은 발견되지 않았습니다(-35~45℃ 기준).")

st.caption("기준 비교: 선택 날짜와 동일한 월-일(예: 01-21)의 과거 분포 대비 편차/백분위/랭크를 제공합니다.")
