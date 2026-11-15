from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from skrub import TableVectorizer

import sempipes  # pylint: disable=unused-import

from experiments.size_sweep.tracking import attach_feature_dump

DATA_DIR = Path("experiments/processing_modes/data/bearing")
RAW_DIR = DATA_DIR / "raw"
CLIPS_DIR = DATA_DIR / "clips"
ASSETS_PATH = DATA_DIR / "assets.csv"
WINDOWS_PATH = DATA_DIR / "windows.csv"
SENSOR_CLIPS_PATH = DATA_DIR / "sensor_clips.csv"
SAMPLE_RATE = 12000
WINDOW_SAMPLES = SAMPLE_RATE // 2
NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
SENSORS = {"drive_end": "DE", "fan_end": "FE"}
SENSOR_PREFIX = {"drive_end": "de", "fan_end": "fe"}
RECORDINGS = {
    "normal": (97, 98, 99, 100),
    "inner_race_007": (105, 106, 107, 108),
    "ball_007": (118, 119, 120, 121),
    "outer_race_007": (130, 131, 132, 133),
    "inner_race_014": (169, 170, 171, 172),
    "ball_014": (185, 186, 187, 188),
    "outer_race_014": (197, 198, 199, 200),
    "inner_race_021": (209, 210, 211, 212),
    "ball_021": (222, 223, 224, 225),
    "outer_race_021": (234, 235, 236, 237),
}
TIME_COLUMNS = {
    "rms_amplitude": "Root-mean-square amplitude of the vibration clip, as a number.",
    "kurtosis": "Kurtosis of the vibration clip, as a number.",
    "crest_factor": "Crest factor of the vibration clip, as a number.",
    "peak_to_peak": "Peak-to-peak amplitude of the vibration clip, as a number.",
}
SPECTRUM_COLUMNS = {
    "dominant_frequency_hz": "Dominant frequency of the vibration clip in hertz, as a number.",
    "envelope_peak_hz": "Frequency in hertz of the strongest peak in the envelope spectrum, as a number.",
    "high_band_energy_ratio": "Fraction of the clip energy above 2000 Hz, as a number between 0 and 1.",
}
MEASUREMENTS = list(TIME_COLUMNS) + list(SPECTRUM_COLUMNS)
CONTEXT = (
    "Each file is one 0.5 second mono vibration clip from an accelerometer on a motor bearing housing. "
    "The sensor column says whether it sits at the drive end or the fan end. "
    "The sample rate is 12000 Hz. The rpm column is the shaft speed. "
    "Do not output the fault name."
)
TIME_PROMPT = f"Measure the vibration clip in the time domain. {CONTEXT}"
SPECTRUM_PROMPT = f"Measure the frequency content of the vibration clip. {CONTEXT}"
SPLIT_SEED = 42
TRAIN_SIZES = (200, 350, 590)
# Seed 42 holds out one recording per class. These recordings are never trained on.
TEST_RECORDINGS = {1, 35, 14, 3, 17, 16, 4, 28, 19, 15}


def _channel(mat: dict, number: int, sensor_code: str) -> np.ndarray | None:
    key = f"X{number:03d}_{sensor_code}_time"
    if key not in mat:
        return None
    return np.asarray(mat[key], dtype=np.float64).ravel()


def _rpm(mat: dict, number: int, load: int) -> int:
    key = f"X{number:03d}RPM"
    if key in mat:
        return int(np.asarray(mat[key]).ravel()[0])
    return NOMINAL_RPM[load]


def _read_recording(number: int, load: int) -> tuple[dict[str, np.ndarray], int] | None:
    from scipy.io import loadmat
    from scipy.signal import resample_poly

    mat = loadmat(RAW_DIR / f"{number}.mat")
    channels = {}
    for sensor, code in SENSORS.items():
        signal = _channel(mat, number, code)
        if signal is None:
            print(f"recording {number} has no {sensor} channel, dropped")
            return None
        channels[sensor] = signal
    rpm = _rpm(mat, number, load)
    if abs(rpm - NOMINAL_RPM[load]) > 15:
        raise RuntimeError(f"recording {number} rpm {rpm} does not match load {load}")
    if len(channels["drive_end"]) > 200_000:
        channels = {sensor: resample_poly(signal, 1, 4) for sensor, signal in channels.items()}
    return channels, rpm


def build_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import soundfile as sf

    recordings = []
    for category, numbers in RECORDINGS.items():
        for load, number in enumerate(numbers):
            loaded = _read_recording(number, load)
            if loaded is not None:
                recordings.append((category, load, loaded[0], loaded[1]))
    scale = max(np.abs(signal).max() for *_, channels, _ in recordings for signal in channels.values()) / 0.99

    rng = np.random.default_rng(0)
    recording_ids = rng.permutation(len(recordings))
    assets, windows, clips = [], [], []
    for recording_id, (category, load, channels, rpm) in zip(recording_ids, recordings, strict=True):
        assets.append({"recording_id": int(recording_id), "motor_load_hp": load, "rpm": rpm})
        n_windows = len(channels["drive_end"]) // WINDOW_SAMPLES
        for offset in range(n_windows):
            windows.append(
                {
                    "recording_id": int(recording_id),
                    "start_s": offset * WINDOW_SAMPLES / SAMPLE_RATE,
                    "category": category,
                    "_channels": channels,
                    "_offset": offset,
                }
            )

    window_ids = rng.permutation(len(windows))
    clip_ids = rng.permutation(len(windows) * len(SENSORS))
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    clip_index = 0
    for window_id, window in zip(window_ids, windows, strict=True):
        window["window_id"] = int(window_id)
        start = window.pop("_offset") * WINDOW_SAMPLES
        channels = window.pop("_channels")
        for sensor in SENSORS:
            path = CLIPS_DIR / f"{clip_ids[clip_index]:05d}.wav"
            clip_index += 1
            sf.write(path, channels[sensor][start : start + WINDOW_SAMPLES] / scale, SAMPLE_RATE, subtype="PCM_16")
            clips.append({"window_id": int(window_id), "sensor": sensor, "filename": path.as_posix()})

    assets_frame = pd.DataFrame(assets).sort_values("recording_id").reset_index(drop=True)
    windows_frame = pd.DataFrame(windows)[["window_id", "recording_id", "start_s", "category"]]
    windows_frame = windows_frame.sort_values("window_id").reset_index(drop=True)
    clips_frame = pd.DataFrame(clips).sort_values("filename").reset_index(drop=True)
    assets_frame.to_csv(ASSETS_PATH, index=False)
    windows_frame.to_csv(WINDOWS_PATH, index=False)
    clips_frame.to_csv(SENSOR_CLIPS_PATH, index=False)
    return assets_frame, windows_frame, clips_frame


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if ASSETS_PATH.exists() and WINDOWS_PATH.exists() and SENSOR_CLIPS_PATH.exists():
        return pd.read_csv(ASSETS_PATH), pd.read_csv(WINDOWS_PATH), pd.read_csv(SENSOR_CLIPS_PATH)
    return build_table()


def _clips_with_rpm(clips: pd.DataFrame, windows: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    speeds = windows[["window_id", "recording_id"]].merge(assets[["recording_id", "rpm"]], on="recording_id")
    joined = clips.merge(speeds[["window_id", "rpm"]], on="window_id", how="inner")
    return joined[["window_id", "sensor", "filename", "rpm"]].reset_index(drop=True)


def _windows_with_measurements(windows: pd.DataFrame, measured: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    sensors_per_window = measured.groupby("window_id")["sensor"].nunique()
    covered = sensors_per_window.reindex(windows["window_id"]).fillna(0)
    if (covered < len(SENSORS)).any():
        raise RuntimeError("A window is missing a sensor clip")
    wide = measured.pivot(index="window_id", columns="sensor", values=MEASUREMENTS)
    wide.columns = [f"{SENSOR_PREFIX[sensor]}_{column}" for column, sensor in wide.columns]
    joined = windows[["window_id", "recording_id"]].merge(wide.reset_index(), on="window_id", how="left")
    joined = joined.merge(assets[["recording_id", "motor_load_hp", "rpm"]], on="recording_id", how="left")
    return joined.reset_index(drop=True)


def _derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    shaft_hz = frame["rpm"] / 60.0
    for prefix in SENSOR_PREFIX.values():
        envelope = pd.to_numeric(frame[f"{prefix}_envelope_peak_hz"], errors="coerce")
        frame[f"{prefix}_envelope_order"] = envelope / shaft_hz
    de_rms = pd.to_numeric(frame["de_rms_amplitude"], errors="coerce")
    fe_rms = pd.to_numeric(frame["fe_rms_amplitude"], errors="coerce")
    frame["de_fe_rms_ratio"] = de_rms / fe_rms.replace(0, np.nan)
    de_kurtosis = pd.to_numeric(frame["de_kurtosis"], errors="coerce")
    fe_kurtosis = pd.to_numeric(frame["fe_kurtosis"], errors="coerce")
    frame["de_fe_kurtosis_diff"] = de_kurtosis - fe_kurtosis
    # Exact rpm and load identify a recording, and the split holds out whole recordings.
    return frame.drop(columns=["window_id", "recording_id", "rpm", "motor_load_hp"])


def sempipes_pipeline(mode: str, artifact_dir: Path | None = None):
    windows = skrub.var("data")
    labels = skrub.var("labels")
    sensor_clips = skrub.var("sensor_clips")
    assets = skrub.var("assets")
    windows = windows.skb.mark_as_X()
    labels = labels.skb.mark_as_y()

    clips = sensor_clips.skb.apply_func(_clips_with_rpm, windows, assets)
    timed = clips.sem_extract_features(
        nl_prompt=TIME_PROMPT,
        input_columns=["filename", "sensor", "rpm"],
        name="extract_time_stats",
        output_columns=TIME_COLUMNS,
        generate_via_code=mode != "llm",
    )
    measured = timed.sem_extract_features(
        nl_prompt=SPECTRUM_PROMPT,
        input_columns=["filename", "sensor", "rpm"],
        name="extract_spectrum",
        output_columns=SPECTRUM_COLUMNS,
        generate_via_code=mode != "llm",
    )
    joined = windows.skb.apply_func(_windows_with_measurements, measured, assets)
    features = joined.skb.apply_func(_derived_features)
    if artifact_dir is not None:
        features = attach_feature_dump(features, artifact_dir, "bearing_features", [])
    encoded = features.skb.apply(TableVectorizer())
    return encoded.skb.apply(HistGradientBoostingClassifier(random_state=0), y=labels)


def _check_no_leak(clips: pd.DataFrame) -> None:
    for filename in clips["filename"]:
        name = Path(filename).name
        if any(category in filename for category in RECORDINGS) or any(sensor in name for sensor in SENSORS):
            raise RuntimeError(f"Clip path {filename} names its class or sensor")


def _train_recordings(windows: pd.DataFrame, train_size: int) -> set[int]:
    """Nested whole-recording prefixes. A smaller size is contained in the next one."""
    rng = np.random.default_rng(SPLIT_SEED)
    groups = []
    recordings = windows.groupby("recording_id", as_index=False).agg(category=("category", "first"))
    pool = recordings.loc[~recordings["recording_id"].isin(TEST_RECORDINGS)].sort_values("recording_id")
    for _, group in pool.groupby("category", sort=True):
        ids = group["recording_id"].to_numpy().copy()
        rng.shuffle(ids)
        groups.append([int(recording_id) for recording_id in ids])
    ordered = []
    cursor = 0
    while any(cursor < len(group) for group in groups):
        for group in groups:
            if cursor < len(group):
                ordered.append(group[cursor])
        cursor += 1
    counts = windows.groupby("recording_id").size().to_dict()
    cumulative = 0
    totals = []
    for recording_id in ordered:
        cumulative += int(counts[recording_id])
        totals.append(cumulative)
    best_end = 1
    best_distance = abs(totals[0] - train_size)
    for end, total in enumerate(totals, start=1):
        distance = abs(total - train_size)
        if distance < best_distance:
            best_end = end
            best_distance = distance
    if train_size == TRAIN_SIZES[-1]:
        best_end = len(ordered)
    return set(ordered[:best_end])


def load_split(train_size: int) -> tuple[dict, dict, pd.Series, pd.Series]:
    if train_size not in TRAIN_SIZES:
        raise ValueError(f"Bearing train size must be one of {TRAIN_SIZES}")
    assets, windows, clips = _load_tables()
    _check_no_leak(clips)
    train_recordings = _train_recordings(windows, train_size)

    def _env(rows: pd.DataFrame) -> dict:
        rows = rows.reset_index(drop=True)
        return {
            "data": rows[["window_id", "recording_id", "start_s"]],
            "labels": rows["category"],
            "sensor_clips": clips[clips["window_id"].isin(set(rows["window_id"]))].reset_index(drop=True),
            "assets": assets,
        }

    train_env = _env(windows[windows["recording_id"].isin(train_recordings)])
    test_env = _env(windows[windows["recording_id"].isin(TEST_RECORDINGS)])
    return train_env, test_env, train_env["labels"], test_env["labels"]
