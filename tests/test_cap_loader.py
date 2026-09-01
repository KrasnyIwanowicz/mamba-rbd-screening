"""
Testy dla CAPSleepLoader.parse_remlogic_txt -- w szczegolnosci regresja na
bugu ze starej wersji (CAPSleepDataset), ktory zerowal df_stages dla
kazdego pacjenta, bo "Unknown Position" ma spacje w srodku. Ta wersja uzywa
pandas.read_csv(sep=r"\t+"), co poprawnie rozwiazuje ten problem -- testy
poklej to bezposrednio, tak samo jak dla poprzedniej implementacji.

Dane syntetyczne nizej nasladuja dokladny format realnego pliku z
physionet.org/content/capslpdb/1.0.0/rbd1.txt -- wartosci sa wymyslone na
potrzeby testu, nie skopiowane z zadnego prawdziwego rekordu pacjenta.
"""
from pathlib import Path

import pytest

from src.data.cap_loader import CAPSleepLoader

SYNTHETIC_TXT = (
    "RemLogic Event Export\n"
    "Patient:\tTEST 1\n"
    "Patient ID:\n"
    "Recording Date:\t01/01/2000\n"
    "\n"
    "Events Included:\n"
    "MCAP-A1\n"
    "SLEEP-REM\n"
    "SLEEP-S0\n"
    "SLEEP-S1\n"
    "SLEEP-S2\n"
    "SLEEP-S3\n"
    "SLEEP-S4\n"
    "\n"
    "Scoring Session:\n"
    "\n"
    "Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n"
    "W\tUnknown Position\t22:00:00\tSLEEP-S0\t30\tROC-LOC\n"
    "S2\tLeft\t22:00:30\tSLEEP-S2\t30\tROC-LOC\n"
    "S2\tLeft\t22:00:45\tMCAP-A1\t6\tEEG-Fp2-F4\n"
    "S3\tUnknown Position\t22:01:00\tSLEEP-S3\t30\tROC-LOC\n"
    "R\tSupine\t22:01:30\tSLEEP-REM\t30\tROC-LOC\n"
    "R\tSupine\t22:02:00\tSLEEP-REM\t30\tROC-LOC\n"
)


@pytest.fixture
def synthetic_txt_path(tmp_path: Path) -> Path:
    p = tmp_path / "rbd_test1.txt"
    p.write_text(SYNTHETIC_TXT, encoding="latin-1")
    return p


def test_parse_remlogic_txt_does_not_return_empty_dataframe(synthetic_txt_path):
    loader = CAPSleepLoader(data_dir=synthetic_txt_path.parent)
    df = loader.parse_remlogic_txt(synthetic_txt_path)
    # To jest dokladnie to, co stara wersja psula: 0 wierszy zawsze.
    assert len(df) > 0


def test_parse_remlogic_txt_extracts_correct_stage_sequence(synthetic_txt_path):
    loader = CAPSleepLoader(data_dir=synthetic_txt_path.parent)
    df = loader.parse_remlogic_txt(synthetic_txt_path)
    # 5 wierszy SLEEP-* w fixture (MCAP-A1 jest wykluczony)
    assert len(df) == 5
    assert list(df["stage_clean"]) == ["WAKE", "N2", "N3", "REM", "REM"]


def test_parse_remlogic_txt_excludes_mcap_microstructure_rows(synthetic_txt_path):
    loader = CAPSleepLoader(data_dir=synthetic_txt_path.parent)
    df = loader.parse_remlogic_txt(synthetic_txt_path)
    # Gdyby MCAP-A1 wyciekl, mielibysmy 6 wierszy i wpis z duration=6.
    assert (df["duration_clean"] == 30).all()


def test_parse_remlogic_txt_handles_position_with_embedded_space(synthetic_txt_path):
    loader = CAPSleepLoader(data_dir=synthetic_txt_path.parent)
    df = loader.parse_remlogic_txt(synthetic_txt_path)
    # Pierwszy wiersz ma "Unknown Position" (spacja w srodku) -- jesli
    # separator by to zle rozbil, stage wyszedlby bledny.
    assert df.iloc[0]["stage_clean"] == "WAKE"


def test_parse_remlogic_txt_missing_header_raises(tmp_path):
    p = tmp_path / "no_header.txt"
    p.write_text("just some random text\nwith no header at all\n", encoding="latin-1")
    loader = CAPSleepLoader(data_dir=tmp_path)

    with pytest.raises(ValueError, match="nagłówka|naglowka"):
        loader.parse_remlogic_txt(p)


def test_stage_map_covers_all_r_and_k_codes():
    # Sanity check -- gdyby ktos usunal wpis z STAGE_MAP przez pomylke,
    # nieznana etykieta wyladuje jako "UNKNOWN" zamiast rzucic blad na
    # etapie mapowania -- to test wykrywa braki w samej mapie.
    expected_raw_codes = {"W", "S0", "S1", "S2", "S3", "S4", "REM", "R", "MT"}
    assert expected_raw_codes <= set(CAPSleepLoader.STAGE_MAP.keys())
