import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

sys.path.insert(0, str(Path(__file__).parent))
from lib.utils import load_params, remap_wav_paths


P = load_params()
A = P["acoustic"]
F = Path(P["paths"]["features_dir"])
F.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=4)
def get_sound(path):
    return parselmouth.Sound(path)


def measure_formants(seg, mid, max_formant, dur_ms):
    out = {k: np.nan for k in ["F1", "F2", "F3", "F1_25", "F1_75", "F2_25", "F2_75"]}
    try:
        fmt = seg.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=A["n_formants"],
            maximum_formant=max_formant,
        )
        out["F1"] = fmt.get_value_at_time(1, mid)
        out["F2"] = fmt.get_value_at_time(2, mid)
        out["F3"] = fmt.get_value_at_time(3, mid)
        if dur_ms > A["long_vowel_threshold_ms"]:
            for tag, frac in [("25", 0.25), ("75", 0.75)]:
                out[f"F1_{tag}"] = fmt.get_value_at_time(1, seg.duration * frac)
                out[f"F2_{tag}"] = fmt.get_value_at_time(2, seg.duration * frac)
    except Exception:
        pass
    return out


def measure_f0(seg):
    try:
        pitch = seg.to_pitch(
            pitch_floor=A["pitch_floor"], pitch_ceiling=A["pitch_ceiling"]
        )
        vals = pitch.selected_array["frequency"]
        vals = vals[vals > 0]
        return float(np.mean(vals)) if len(vals) else np.nan
    except Exception:
        return np.nan


def measure_scg(seg, phoneme):
    if phoneme not in A["fricatives"]:
        return np.nan
    try:
        return call(seg.to_spectrum(), "Get centre of gravity", 2.0)
    except Exception:
        return np.nan


def measure_token(row):
    snd = get_sound(row["wav_path"])
    seg = snd.extract_part(
        from_time=row["onset"], to_time=row["offset"], preserve_times=False
    )
    mid = seg.duration / 2
    is_male = str(row["gender"]).upper().startswith("M")
    max_formant = A["max_formant_male"] if is_male else A["max_formant_female"]
    dur_ms = seg.duration * 1000

    feats = measure_formants(seg, mid, max_formant, dur_ms)
    feats["f0"] = measure_f0(seg)
    feats["SCG"] = measure_scg(seg, row["phoneme"])
    return pd.Series(feats)


def main():
    tokens = pd.read_csv(F / "tokens.csv")
    tokens["wav_path"] = remap_wav_paths(tokens["wav_path"], P["corpus"]["root"])

    feats = tokens.apply(measure_token, axis=1)
    out = pd.concat([tokens, feats], axis=1)
    out.to_csv(F / "features_acoustic.csv", index=False)

    miss = out.groupby("phoneme")[["F1", "F2", "f0"]].apply(lambda x: x.isna().mean())
    miss.to_csv(F / "missing_report.csv")
    print(f"Wrote {len(out)} acoustic rows; missing-rate report saved.")


if __name__ == "__main__":
    main()
