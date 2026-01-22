import io
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =============================
# Page + Cute Pastel Theme
# =============================
st.set_page_config(page_title="🌈 서울 기온 놀이터", layout="wide")

PASTEL_CSS = """
<style>
:root{
  --bg1:#fff7fb; --bg2:#f3fbff; --bg3:#f7fff4;
  --p1:#ffd6e8; --p2:#d6f5ff; --p3:#e8d6ff; --p4:#d6ffe6; --p5:#fff3c7;
  --txt:#2b2b2b;
}
.stApp{
  background: linear-gradient(120deg,var(--bg1),var(--bg2),var(--bg3));
  color: var(--txt);
}
div[data-testid="stSidebar"]{
  background: linear-gradient(180deg,var(--p2),var(--p3),var(--p4));
  border-right: 2px dashed rgba(0,0,0,0.08);
}
h1,h2,h3{
  letter-spacing:-0.5px;
}
.rainbow-card{
  background: linear-gradient(90deg,var(--p1),var(--p2),var(--p3),var(--p4),var(--p5));
  padding: 14px 16px;
  border-radius: 18px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.06);
  border: 1px solid rgba(0,0,0,0.06);
}
.kpi-card{
  background: rgba(255,255,255,0.75);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: 0 6px 14px rgba(0,0,0,0.05);
}
.badge{
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  background: rgba(255,255,255,0.7);
  border: 1px solid rgba(0,0,0,0.08);
  margin-right:6px;
  font-size: 0.9rem;
}
.small-note{
  font-size: 0.9rem;
  opacity: 0.85;
}
</style>
"""
st.markdown(PASTEL_CSS, unsafe_allow_html=True)

# =============================
# Data Utilities
# =============================
EXPECTED_COLS = ["날짜", "지점", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
STATION_SEOUL = 108

def read_csv_flexible(file_like) -> pd.DataFrame:
    raw = file_like.read() if hasattr(file_like, "read") else open(file_like, "rb").read()
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            text = raw.decode(enc)
            return pd.read_csv(io.StringIO(text))
        except Exception:
            pass
    # 마지막 fallback
    return pd.read_csv(io.BytesIO(raw), encoding="cp949")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if all(c in df.columns for c in EXPECTED_COLS):
        return df[EXPECTED_COLS].copy()
    if df.shape[1] >= 5:
        tmp = df.iloc[:, :5].copy()
        tmp.columns = EXPECTED_COLS
        return tmp
    raise ValueError("CSV 컬럼이 5개 미만입니다. (날짜/지점/평균/최저/최고기온) 형식을 확인해 주세요.")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["지점"] = pd.to_numeric(df["지점"], errors="coerce")
    for c in ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 헤더/설명/공백행 제거
    df = df.dropna(subset=["날짜", "지점"]).copy()
    df["지점"] = df["지점"].astype(int)

    df = df.sort_values("날짜").drop_duplicates(subset=["날짜", "지점"], keep="last")
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    df["일"] = df["날짜"].dt.day
    df["월일"] = df["날짜"].dt.strftime("%m-%d")

    # 계절 (기상학적 계절)
    # DJF: 겨울(12,1,2), MAM: 봄(3,4,5), JJA: 여름(6,7,8), SON: 가을(9,10,11)
    def season(m):
        if m in [12,1,2]: return "겨울 ❄️"
        if m in [3,4,5]: return "봄 🌸"
        if m in [6,7,8]: return "여름 🏖️"
        return "가을 🍂"
    df["계절"] = df["월"].apply(season)

    # 겨울은 연도 경계(12월은 다음 해 겨울로 묶는 것이 일반적) — 옵션으로 처리
    df["계절연도"] = df["연도"]
    df.loc[df["월"] == 12, "계절연도"] = df.loc[df["월"] == 12, "연도"] + 1

    return df

def merge_datasets(base_df: pd.DataFrame, uploaded_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat([base_df] + uploaded_dfs, ignore_index=True)
    merged = merged.sort_values("날짜").drop_duplicates(subset=["날짜", "지점"], keep="last")
    return merged

def day_of_year_comparison(df, target_date, station, metric):
    target_md = target_date.strftime("%m-%d")
    sub = df[df["지점"] == station].copy()
    hist = sub[sub["월일"] == target_md].copy()
    day_row = sub[sub["날짜"] == target_date]
    if day_row.empty:
        return None, hist, target_md
    x = float(day_row.iloc[0][metric])
    hv = hist[metric].dropna()
    if hv.empty:
        return None, hist, target_md

    mean = hv.mean()
    median = hv.median()
    std = hv.std(ddof=0)
    z = (x - mean) / std if std and std > 0 else None

    pct = (hv <= x).mean() * 100.0  # 낮을수록 더 '추운 쪽'
    rank_cold = int((hv < x).sum() + 1)
    n = int(hv.shape[0])

    stats = dict(
        target_value=x,
        mean=float(mean),
        median=float(median),
        std=float(std) if pd.notnull(std) else None,
        zscore=float(z) if z is not None else None,
        percentile_colder_or_equal=float(pct),
        rank_cold=rank_cold,
        n_years=n
    )
    return stats, hist, target_md

# =============================
# Optional: CSAT dates (수능)
# =============================
def load_csat_dates():
    """
    data/csat_dates.csv 가 있으면 사용.
    없으면 빈 DataFrame 반환(앱에서 업로드 유도/안내).
    파일 형식: 시험연도, 수능일(YYYY-MM-DD)
    """
    path = "data/csat_dates.csv"
    if os.path.exists(path):
        cs = pd.read_csv(path)
        # columns normalize
        cols = {c.lower(): c for c in cs.columns}
        # 기대: '시험연도', '수능일'
        if "수능일" not in cs.columns:
            # fallback: first date-like column
            for c in cs.columns:
                if "일" in c and "수능" in c:
                    cs = cs.rename(columns={c: "수능일"})
                    break
        if "시험연도" not in cs.columns:
            for c in cs.columns:
                if "연" in c:
                    cs = cs.rename(columns={c: "시험연도"})
                    break
        cs["수능일"] = pd.to_datetime(cs["수능일"], errors="coerce")
        cs = cs.dropna(subset=["수능일"]).copy()
        if "시험연도" in cs.columns:
            cs["시험연도"] = pd.to_numeric(cs["시험연도"], errors="coerce")
        # 중복 제거
        cs = cs.drop_duplicates(subset=["수능일"], keep="last").sort_values("수능일")
        return cs
    return pd.DataFrame(columns=["시험연도", "수능일"])

# =============================
# Optional: 24절기 (approx)
# =============================
SOLAR_TERMS_APPROX = [
    ("소한 ❄️", 1, 5), ("대한 🧣", 1, 20),
    ("입춘 🌱", 2, 4), ("우수 🌧️", 2, 19),
    ("경칩 🐞", 3, 6), ("춘분 🌷", 3, 20),
    ("청명 🌿", 4, 4), ("곡우 🌾", 4, 20),
    ("입하 🍀", 5, 5), ("소만 🍈", 5, 21),
    ("망종 🌾", 6, 5), ("하지 ☀️", 6, 21),
    ("소서 🍉", 7, 7), ("대서 🥵", 7, 23),
    ("입추 🍂", 8, 7), ("처서 🌾", 8, 23),
    ("백로 🦢", 9, 7), ("추분 🍁", 9, 23),
    ("한로 🧥", 10, 8), ("상강 🍠", 10, 23),
    ("입동 🧤", 11, 7), ("소설 🌨️", 11, 22),
    ("대설 ⛄", 12, 7), ("동지 🕯️", 12, 21),
]

def make_solar_term_dates_for_year(year: int) -> pd.DataFrame:
    """
    24절기 날짜를 '근사 고정 월-일'로 생성.
    정확한 천문 절기일(연도별 1~2일 변동)은 반영하지 않음.
    """
    rows = []
    for name, m, d in SOLAR_TERMS_APPROX:
        rows.append({"연도": year, "절기": name, "날짜": pd.Timestamp(year=year, month=m, day=d)})
    return pd.DataFrame(rows)

# =============================
# Load base + upload merge
# =============================
@st.cache_data(show_spinner=False)
def load_base():
    base_path = "data/base_seoul_temp.csv"
    df0 = read_csv_flexible(base_path)
    return clean_data(df0)

st.markdown(
    """
<div class="rainbow-card">
  <h1>🌈 서울 기온 놀이터 ☁️🌤️🌧️❄️</h1>
  <div class="small-note">
    🎀 기본 데이터 탑재 + 업로드 자동 병합 · 📊 같은 월-일 비교 · 🧑‍🎓 수능날 · 🍀 계절 추이 · 🌿 24절기
  </div>
</div>
""",
    unsafe_allow_html=True
)

base_df = load_base()

st.sidebar.header("🧁 데이터 입력")
uploaded_files = st.sidebar.file_uploader(
    "같은 형식의 CSV 업로드(복수 가능) 📎",
    type=["csv"], accept_multiple_files=True
)

uploaded_clean = []
if uploaded_files:
    for f in uploaded_files:
        try:
            d = read_csv_flexible(f)
            d = clean_data(d)
            uploaded_clean.append(d)
        except Exception as e:
            st.sidebar.error(f"'{f.name}' 처리 실패: {e}")

df = merge_datasets(base_df, uploaded_clean) if uploaded_clean else base_df

# =============================
# Controls
# =============================
st.sidebar.header("🍡 분석 설정")
station = st.sidebar.number_input("지점 코드(서울=108) 🧭", value=STATION_SEOUL, step=1)
metric = st.sidebar.selectbox("기온 지표 🌡️", ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"], index=0)

sub_station = df[df["지점"] == station]
max_date = sub_station["날짜"].max()
min_date = sub_station["날짜"].min()

date_input = st.sidebar.date_input(
    "기준 날짜 📅 (기본: 최신)",
    value=max_date.date() if pd.notnull(max_date) else None,
    min_value=min_date.date() if pd.notnull(min_date) else None,
    max_value=max_date.date() if pd.notnull(max_date) else None,
)
target_date = pd.to_datetime(date_input)

# KPI row
k1, k2, k3, k4 = st.columns(4)
for col, label, value in [
    (k1, "🧺 총 행 수(병합 후)", f"{len(df):,}"),
    (k2, "🕰️ 기간 시작", f"{min_date.date() if pd.notnull(min_date) else '-'}"),
    (k3, "🏁 기간 종료", f"{max_date.date() if pd.notnull(max_date) else '-'}"),
    (k4, "📍 지점", f"{station}"),
]:
    col.markdown(f'<div class="kpi-card"><span class="badge">{label}</span><div style="font-size:1.35rem;font-weight:700">{value}</div></div>', unsafe_allow_html=True)

st.divider()

# =============================
# Tabs
# =============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 날짜 비교",
    "🧑‍🎓 수능날",
    "🍀 계절 추이",
    "🌿 24절기",
    "🧼 데이터 품질"
])

# -----------------------------
# TAB 1: 날짜 비교(기본 기능 강화)
# -----------------------------
with tab1:
    stats, hist, md = day_of_year_comparison(df, target_date, station, metric)
    if stats is None:
        st.error("선택한 날짜/지점에서 값을 찾을 수 없습니다. (업로드 데이터에 해당 날짜가 있는지 확인)")
    else:
        delta = stats["target_value"] - stats["mean"]
        direction = "더웠어요 🔥" if delta > 0 else "추웠어요 🧊" if delta < 0 else "평년이에요 🙂"

        st.markdown(
            f"""
<div class="rainbow-card">
  <div style="font-size:1.1rem">
    📅 <b>{target_date.date()}</b> (<b>{md}</b>)의 <b>{metric}</b>은
    <span style="font-size:1.6rem;font-weight:800"> {stats['target_value']:.1f}℃ </span>
    입니다.
  </div>
  <div class="small-note" style="margin-top:6px">
    같은 <b>{md}</b> 역사적 분포(n={stats['n_years']}) 평균 <b>{stats['mean']:.1f}℃</b> 대비
    <b>{abs(delta):.1f}℃</b> {'높아서' if delta>0 else '낮아서' if delta<0 else ''} <b>{direction}</b><br/>
    🧊 추운쪽 백분위(낮을수록 더 추움): <b>{stats['percentile_colder_or_equal']:.1f}</b> ·
    🥶 추운 순 랭크: <b>{stats['rank_cold']}/{stats['n_years']}</b>
    {" · 📐 Z-score: <b>" + f"{stats['zscore']:.2f}" + "</b>" if stats["zscore"] is not None else ""}
  </div>
</div>
""",
            unsafe_allow_html=True
        )

        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("📈 전체 기간 시계열")
            sub = df[df["지점"] == station].sort_values("날짜")
            fig_ts = px.line(sub, x="날짜", y=metric, title=f"{station} | {metric} (전체 기간)")
            fig_ts.update_layout(hovermode="x unified")
            st.plotly_chart(fig_ts, use_container_width=True)

        with right:
            st.subheader("🎁 같은 월-일 분포 vs 선택 날짜")
            hist_vals = hist[metric].dropna().sort_values()
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=hist_vals, name=f"{md} 분포",
                boxpoints="all", jitter=0.35, pointpos=0
            ))
            fig_box.add_hline(
                y=stats["target_value"], line_dash="dash",
                annotation_text=f"{target_date.date()} 값: {stats['target_value']:.1f}℃",
                annotation_position="top left"
            )
            fig_box.update_layout(title=f"{md} | {metric} 분포", yaxis_title=metric, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------
# TAB 2: 수능날(별도 기능)
# -----------------------------
with tab2:
    st.subheader("🧑‍🎓 수학능력시험(수능) 날짜별 기온")
    st.markdown(
        "<div class='small-note'>📌 이 기능은 <b>수능 날짜 목록</b>이 필요합니다. "
        "프로젝트 폴더에 <code>data/csat_dates.csv</code>를 두면 자동으로 읽습니다.</div>",
        unsafe_allow_html=True
    )

    csat_df = load_csat_dates()

    # 업로드 옵션 제공
    up_csat = st.file_uploader("수능 날짜 CSV 업로드(선택) 📎", type=["csv"], key="csat_uploader")
    if up_csat is not None:
        try:
            tmp = read_csv_flexible(up_csat)
            # 기대 컬럼: 시험연도, 수능일
            if "수능일" not in tmp.columns:
                # 첫 date-like col heuristic
                for c in tmp.columns:
                    if "일" in c:
                        tmp = tmp.rename(columns={c: "수능일"})
                        break
            if "시험연도" not in tmp.columns:
                for c in tmp.columns:
                    if "연" in c:
                        tmp = tmp.rename(columns={c: "시험연도"})
                        break
            tmp["수능일"] = pd.to_datetime(tmp["수능일"], errors="coerce")
            tmp = tmp.dropna(subset=["수능일"]).drop_duplicates(subset=["수능일"]).sort_values("수능일")
            csat_df = tmp
            st.success("업로드한 수능 날짜 목록을 사용합니다.")
        except Exception as e:
            st.error(f"수능 날짜 CSV 처리 실패: {e}")

    if csat_df.empty:
        st.warning("수능 날짜 목록이 없어 시각화를 진행할 수 없습니다. 아래 안내 형식대로 파일을 준비해 주세요.")
        st.code("시험연도,수능일\n2025,2024-11-14\n2026,2025-11-13\n...", language="text")
    else:
        # 서울 기온과 조인
        sub = df[df["지점"] == station].copy()
        join = csat_df.copy()
        join = join.rename(columns={"수능일": "날짜"})
        merged = join.merge(sub[["날짜", metric, "월일"]], on="날짜", how="left")

        missing = merged[metric].isna().sum()
        if missing > 0:
            st.info(f"수능 날짜 {len(merged)}개 중 {missing}개는 기온 데이터에 없어 제외될 수 있습니다.")

        merged_ok = merged.dropna(subset=[metric]).copy()
        if merged_ok.empty:
            st.error("수능 날짜와 기온 데이터가 겹치는 행이 없습니다.")
        else:
            fig = px.line(
                merged_ok.sort_values("날짜"),
                x="날짜", y=metric,
                markers=True,
                title=f"수능일 {metric} 추이 (지점 {station})"
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # 분포 비교: 수능일(대부분 11월 초/중순) vs 같은 월일 역사 분포 내 위치
            # 간단하게 각 수능일에 대해 '같은 월일' 분포에서 퍼센타일 계산
            md_to_values = (
                sub.groupby("월일")[metric]
                .apply(lambda s: s.dropna().values)
                .to_dict()
            )

            def percentile_in_md(md, x):
                arr = md_to_values.get(md, None)
                if arr is None or len(arr) == 0:
                    return np.nan
                return (arr <= x).mean() * 100.0

            merged_ok["추운쪽_백분위"] = merged_ok.apply(
                lambda r: percentile_in_md(r["월일"], r[metric]),
                axis=1
            )

            fig2 = px.scatter(
                merged_ok, x="날짜", y="추운쪽_백분위",
                title="수능일 기온의 ‘추운쪽 백분위’ (낮을수록 더 추운 수능)",
                hover_data=[metric]
            )
            fig2.update_layout(hovermode="x unified", yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(merged_ok.sort_values("날짜").reset_index(drop=True))

# -----------------------------
# TAB 3: 계절별 매년 온도 추이 비교
# -----------------------------
with tab3:
    st.subheader("🍀 계절별 매년 온도 추이 비교")
    st.markdown("<div class='small-note'>❄️ 겨울은 12월을 다음 해 겨울로 묶어(계절연도) 연속성을 확보합니다.</div>", unsafe_allow_html=True)

    sub = df[df["지점"] == station].copy()

    # 연-계절 평균
    g = (sub.groupby(["계절연도", "계절"])[metric]
         .mean()
         .reset_index()
         .rename(columns={"계절연도": "연도"}))

    # 선택 계절
    season_order = ["봄 🌸", "여름 🏖️", "가을 🍂", "겨울 ❄️"]
    seasons = st.multiselect("표시할 계절 선택 🎛️", season_order, default=season_order)

    g2 = g[g["계절"].isin(seasons)].copy()
    fig = px.line(
        g2.sort_values("연도"),
        x="연도", y=metric, color="계절",
        markers=True,
        title=f"연도별 계절 평균 {metric} (지점 {station})"
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 시즌별 분포(박스)
    fig_box = px.box(g2, x="계절", y=metric, points="all", category_orders={"계절": season_order},
                     title="계절 평균 기온 분포(연도별)")
    st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------
# TAB 4: 24절기
# -----------------------------
with tab4:
    st.subheader("🌿 24절기별 기온 비교")
    st.markdown(
        "<div class='small-note'>현재는 <b>근사(고정 월-일)</b>로 절기일을 구성합니다. "
        "연도별 정확 절기일(천문 기준, 1~2일 변동)을 쓰려면 별도 데이터 또는 추가 라이브러리(옵션)가 필요합니다.</div>",
        unsafe_allow_html=True
    )

    sub = df[df["지점"] == station].copy()

    year_min = int(sub["연도"].min())
    year_max = int(sub["연도"].max())
    y1, y2 = st.slider("연도 범위 선택 🗓️", min_value=year_min, max_value=year_max, value=(max(year_min, 1950), year_max))

    years = list(range(y1, y2 + 1))
    terms = pd.concat([make_solar_term_dates_for_year(y) for y in years], ignore_index=True)

    # 절기 날짜와 실제 기온 조인
    terms = terms.merge(sub[["날짜", metric]], on="날짜", how="left")

    miss = terms[metric].isna().sum()
    if miss > 0:
        st.info(f"절기 {len(terms)}개 중 {miss}개는 해당 날짜가 데이터에 없어 제외될 수 있습니다(초기 연도/누락일 등).")

    terms_ok = terms.dropna(subset=[metric]).copy()

    # 절기별 평균(연도 범위 전체)
    agg = (terms_ok.groupby("절기")[metric]
           .agg(["mean", "median", "count"])
           .reset_index()
           .sort_values("mean"))

    fig = px.bar(agg, x="절기", y="mean", title=f"24절기 평균 {metric} (선택 연도 범위)", hover_data=["median", "count"])
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 연도별 선택 절기 추이
    picked_terms = st.multiselect("추이를 볼 절기 선택 🌱", list(dict.fromkeys([t for t,_,_ in SOLAR_TERMS_APPROX])),
                                  default=["입춘 🌱", "하지 ☀️", "추분 🍁", "동지 🕯️"])
    terms_pick = terms_ok[terms_ok["절기"].isin(picked_terms)].copy()

    fig2 = px.line(terms_pick.sort_values("날짜"), x="연도", y=metric, color="절기", markers=True,
                   title="절기별 연도 추이")
    fig2.update_layout(hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(terms_ok.sort_values(["연도", "절기"]).reset_index(drop=True))

# -----------------------------
# TAB 5: 데이터 품질
# -----------------------------
with tab5:
    st.subheader("🧼 데이터 품질 점검")
    sub = df[df["지점"] == station].copy()

    miss = sub[EXPECTED_COLS].isna().sum()
    st.write("결측치 개수(지점 필터 적용):")
    st.dataframe(miss.to_frame("missing_count"))

    st.write("물리적 범위 밖(기본 -35~45℃) 의심 값:")
    phys = []
    for c in ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]:
        bad = sub[(sub[c] < -35) | (sub[c] > 45)][["날짜", "지점", c]].dropna()
        phys.append((c, len(bad)))
        if len(bad) > 0:
            st.warning(f"{c}: {len(bad)}건")
            st.dataframe(bad.head(100))
    if all(n == 0 for _, n in phys):
        st.success("의심되는 물리적 범위 밖 값은 발견되지 않았습니다.")
