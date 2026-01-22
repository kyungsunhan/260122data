def load_csat_dates(path: str = "data/csat_dates.csv") -> pd.DataFrame:
    """
    CSAT 날짜 파일 로더(유연 버전)
    - 지원 컬럼(둘 중 하나면 OK):
      (A) academic_year, exam_date, note
      (B) 시험연도, 수능일, 비고
    - 반환 컬럼은 표준화: academic_year, exam_date, note
    """
    csat = pd.read_csv(path)

    # 1) 컬럼 이름 표준화(한글/영문 모두 지원)
    colmap = {}

    # 연도
    if "academic_year" in csat.columns:
        colmap["academic_year"] = "academic_year"
    elif "시험연도" in csat.columns:
        colmap["시험연도"] = "academic_year"
    elif "연도" in csat.columns:
        colmap["연도"] = "academic_year"

    # 날짜
    if "exam_date" in csat.columns:
        colmap["exam_date"] = "exam_date"
    elif "수능일" in csat.columns:
        colmap["수능일"] = "exam_date"
    elif "date" in csat.columns:
        colmap["date"] = "exam_date"

    # 비고
    if "note" in csat.columns:
        colmap["note"] = "note"
    elif "비고" in csat.columns:
        colmap["비고"] = "note"

    csat = csat.rename(columns=colmap)

    # 2) 필수 컬럼 검증
    if "exam_date" not in csat.columns:
        raise ValueError("CSAT 파일에 날짜 컬럼(exam_date 또는 수능일)이 없습니다.")
    if "academic_year" not in csat.columns:
        # 연도가 없어도 작동은 가능하게: 날짜의 연도를 시험연도로 임시 생성(원하시면 제거 가능)
        csat["academic_year"] = pd.to_datetime(csat["exam_date"], errors="coerce").dt.year

    if "note" not in csat.columns:
        csat["note"] = ""

    # 3) 타입 변환
    csat["exam_date"] = pd.to_datetime(csat["exam_date"], errors="coerce")
    csat["academic_year"] = pd.to_numeric(csat["academic_year"], errors="coerce")

    csat = csat.dropna(subset=["exam_date"]).sort_values("exam_date").reset_index(drop=True)
    return csat[["academic_year", "exam_date", "note"]]
