"""
Testy dla CAPSleepDataset.parse_txt_annotations -- w szczególności regresja
na bugu, który zerował df_stages dla każdego pacjenta (patrz PATCH_NOTES.md).

Dane syntetyczne poniżej naśladują dokładny format realnego pliku
z physionet.org/content/capslpdb/1.0.0/rbd1.txt (nagłówek, tabulatory,
przeplatane wiersze SLEEP-*/MCAP-*, pole Position ze spacją w środku) --
wartości są wymyślone na potrzeby testu, nie skopiowane z żadnego
prawdziwego rekordu pacjenta.
"""
from pathlib import Path

import pytest

from src.data.cap_loader import CAPSleepDataset

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


def test_parse_txt_annotations_does_not_return_empty_dataframe(synthetic_txt_path):
    ds = CAPSleepDataset(data_dir=synthetic_txt_path.parent)
    df = ds.parse_txt_annotations(synthetic_txt_path)
    # To jest DOKŁADNIE to, co poprzednia wersja psuła: 0 wierszy zawsze.
    assert len(df) > 0


def test_parse_txt_annotations_extracts_correct_stage_sequence(synthetic_txt_path):
    ds = CAPSleepDataset(data_dir=synthetic_txt_path.parent)
    df = ds.parse_txt_annotations(synthetic_txt_path)
    # 5 wierszy SLEEP-* w fixture (MCAP-A1 jest wykluczony -- to nie epoka)
    assert len(df) == 5
    assert list(df["stage"]) == ["WAKE", "N2", "N3", "REM", "REM"]


def test_parse_txt_annotations_excludes_mcap_microstructure_rows(synthetic_txt_path):
    ds = CAPSleepDataset(data_dir=synthetic_txt_path.parent)
    df = ds.parse_txt_annotations(synthetic_txt_path)
    # Gdyby MCAP-A1 wyciekł do wyniku, mielibyśmy 6 wierszy zamiast 5, i
    # dodatkowy wpis z duration=6 (zamiast 30) w środku sekwencji.
    assert (df["duration"] == 30).all()


def test_parse_txt_annotations_handles_position_with_embedded_space(synthetic_txt_path):
    ds = CAPSleepDataset(data_dir=synthetic_txt_path.parent)
    df = ds.parse_txt_annotations(synthetic_txt_path)
    # Pierwszy wiersz ma "Unknown Position" (spacja w środku) -- jeśli split
    # po białych znakach nadal by tu coś psuł, stage wyszedłby błędny.
    assert df.iloc[0]["stage"] == "WAKE"
    assert df.iloc[0]["time"] == "22:00:00"


def test_parse_txt_annotations_warns_on_unknown_stage_label(tmp_path):
    bad_txt = (
        "Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n"
        "ZZ\tUnknown Position\t22:00:00\tSLEEP-S0\t30\tROC-LOC\n"
    )
    p = tmp_path / "bad.txt"
    p.write_text(bad_txt, encoding="latin-1")
    ds = CAPSleepDataset(data_dir=tmp_path)

    with pytest.warns(UserWarning, match="nieznana etykieta"):
        df = ds.parse_txt_annotations(p)
    assert df.iloc[0]["stage"] == "UNKNOWN"


def test_parse_txt_annotations_missing_header_returns_empty_with_warning(tmp_path):
    p = tmp_path / "no_header.txt"
    p.write_text("just some random text\nwith no header at all\n", encoding="latin-1")
    ds = CAPSleepDataset(data_dir=tmp_path)

    with pytest.warns(UserWarning, match="nie znaleziono nagłówka"):
        df = ds.parse_txt_annotations(p)
    assert len(df) == 0
