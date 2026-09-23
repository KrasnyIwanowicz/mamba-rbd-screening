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


# ---------------------------------------------------------------------------
# Wyrownanie hipnogramu do sygnalu (poprawka 2026-09-23): start epoki z kolumny
# Time wzgledem startu EDF, nie z numeru wiersza.
# ---------------------------------------------------------------------------
import datetime
import warnings

import numpy as np

from src.data.cap_loader import subject_group

_HEADER = SYNTHETIC_TXT.split("Sleep Stage\t")[0] + "Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n"


def _txt(rows: list[tuple[str, str]]) -> str:
    event = {"W": "SLEEP-S0", "S2": "SLEEP-S2", "S3": "SLEEP-S3", "R": "SLEEP-REM"}
    return _HEADER + "".join(f"{st}\tSupine\t{t}\t{event[st]}\t30\tROC-LOC\n" for st, t in rows)


def test_onsets_follow_time_column_across_gap_and_midnight(tmp_path):
    p = tmp_path / "rbd9.txt"
    p.write_text(_txt([("S2", "23:59:00"), ("R", "23:59:30"), ("R", "00:00:00"), ("S2", "00:01:00")]), encoding="latin-1")
    loader = CAPSleepLoader(tmp_path)
    with pytest.warns(UserWarning, match="odstepow"):
        df = loader.parse_remlogic_txt(p, recording_start=datetime.time(23, 58, 0))
    # 60 s po starcie EDF; przez polnoc +30 s; luka 00:00:30 pominieta.
    assert df["start_sec"].tolist() == [60.0, 90.0, 120.0, 180.0]


def test_onsets_without_recording_start_are_relative_to_first_epoch(tmp_path):
    p = tmp_path / "n9.txt"
    p.write_text(_txt([("W", "22:10:00"), ("S2", "22:10:30")]), encoding="latin-1")
    df = CAPSleepLoader(tmp_path).parse_remlogic_txt(p)
    assert df["start_sec"].tolist() == [0.0, 30.0]


def test_hypnogram_starting_before_edf_gives_negative_onset(tmp_path):
    p = tmp_path / "n9.txt"
    p.write_text(_txt([("W", "21:59:30"), ("S2", "22:00:00")]), encoding="latin-1")
    df = CAPSleepLoader(tmp_path).parse_remlogic_txt(p, recording_start=datetime.time(22, 0, 0))
    assert df["start_sec"].tolist() == [-30.0, 0.0]


@pytest.mark.parametrize(
    "subject_id,group",
    [("rbd1", "rbd"), ("n16", "n"), ("nfle3", "nfle"), ("narco2", "narco"), ("RBD4", "rbd")],
)
def test_subject_group_does_not_confuse_prefixes(subject_id, group):
    assert subject_group(subject_id) == group


def test_subject_group_rejects_unknown_ids():
    with pytest.raises(ValueError):
        subject_group("patient7")


def _write_edf(path: Path, start: datetime.datetime, chin_volts: np.ndarray, fs: int):
    pyedflib = pytest.importorskip("pyedflib")
    eeg = np.random.default_rng(1).normal(0, 1e-5, len(chin_volts))
    header = dict(dimension="uV", sample_frequency=fs, physical_max=500.0, physical_min=-500.0,
                  digital_max=32767, digital_min=-32768)
    writer = pyedflib.EdfWriter(str(path), 2, file_type=pyedflib.FILETYPE_EDFPLUS)
    writer.setSignalHeaders([{**header, "label": "EMG1-EMG2"}, {**header, "label": "C4-A1"}])
    writer.setStartdatetime(start)
    writer.writeSamples([chin_volts * 1e6, eeg * 1e6])
    writer.close()


def test_load_subject_slices_rem_from_the_right_place_in_the_signal(tmp_path):
    """Regresja end-to-end: stary kod (start = i * 30) wycinalby REM z [30, 90) s,
    gdzie EMG jest ciche; prawdziwy REM (wg Time) lezy w [90, 150) s, gdzie EMG jest glosne."""
    fs = 200
    rng = np.random.default_rng(0)
    chin = rng.normal(0, 2e-6, fs * 300)
    chin[90 * fs:150 * fs] = rng.normal(0, 50e-6, 60 * fs)
    _write_edf(tmp_path / "rbd9.edf", datetime.datetime(2000, 1, 1, 23, 58, 0), chin, fs)
    (tmp_path / "rbd9.txt").write_text(
        _txt([("S2", "23:59:00"), ("R", "23:59:30"), ("R", "00:00:00"), ("S2", "00:00:30")]), encoding="latin-1"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epochs = CAPSleepLoader(tmp_path, target_fs=fs).load_subject("rbd9")

    rem = [e for e in epochs if e.stage == "REM"]
    assert [e.start_sec for e in rem] == [90.0, 120.0]
    rem_rms = [np.sqrt(np.mean(e.emg_chin ** 2)) for e in rem]
    nrem_rms = [np.sqrt(np.mean(e.emg_chin ** 2)) for e in epochs if e.stage == "N2"]
    assert min(rem_rms) > 10 * max(nrem_rms)
    assert all(e.is_rbd for e in epochs)
    assert [e.epoch_idx for e in epochs] == [0, 1, 2, 3]
