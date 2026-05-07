"""Local pipeline. Whisper/XLS-R features come from the Kaggle notebook —
drop them into ``data/features/`` before running this.
"""

configfile: "params.yaml"
F = config["paths"]["features_dir"]
R = config["paths"]["results_dir"]


rule all:
    input:
        f"{F}/features_acoustic_norm.csv",
        f"{R}/.analyse.done"


rule extract_acoustics:
    input: f"{F}/tokens.csv"
    output: f"{F}/features_acoustic.csv"
    shell: "python scripts/extract_acoustics.py"


rule normalise:
    input:
        ac = f"{F}/features_acoustic.csv"
    output:
        f"{F}/features_acoustic_norm.csv"
    shell: "python scripts/normalise.py"


rule analyse:
    input:
        ac = f"{F}/features_acoustic_norm.csv"
    output: touch(f"{R}/.analyse.done")
    shell: "python scripts/analyse.py"
