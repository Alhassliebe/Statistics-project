# Phonetics Project — Local Pipeline

## Pipeline architecture

The pipeline is split between two stages

1. In the Kaggle notebook (`kaggle_neural_extraction.ipynb`), TextGrid annotations are used to segment each WAV file into phoneme-level chunks. These segments are passed through pretrained Whisper and wav2vec 2.0 and hidden representations from selected layers are pooled to produce feature vectors. The resulting features are saved to `data/features/`.
2. The local stage, managed with `Snakefile`, performs the analysis: Praat-based formant extraction, Lobanov normalization, dimensionality reduction (PCA/UMAP), and statistical evaluation. All parameters are defined in `params.yaml`, so changing them triggers recomputation only where needed.

Run the Kaggle notebook first (`kaggle_neural_extraction.ipynb`), download the three
output files, drop them into `data/features/`, then run the local pipeline below.

## Run

```bash
pip install -r requirements.txt
snakemake -j4
```

Or stage by stage:

```bash
python scripts/extract_acoustics.py
python scripts/normalise.py
python scripts/analyse.py
```

`analyse.py` and `normalise.py` skip neural sections cleanly if the `.npz`
files from Kaggle are missing — useful while the GPU run is in progress.
