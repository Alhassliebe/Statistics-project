from pathlib import Path
import yaml


def load_params(path="params.yaml"):
    return yaml.safe_load(open(path))


def remap_wav_paths(wav_paths, corpus_root):
    """Translate Kaggle paths in tokens.csv to local paths.

    Both Kaggle and the local layout end with ``<SPK>/<filename>.wav``,
    so we keep the last two segments and re-root them under ``corpus_root``.
    Local paths pass through unchanged.
    """
    corpus_root = Path(corpus_root)
    out = []
    for p in wav_paths:
        p = Path(p)
        if p.exists():
            out.append(str(p))
        else:
            local = corpus_root / "2/wav_et_textgrids/FRcorp_textgrids_only" / p.parent.name / p.name
            out.append(str(local))
    return out
