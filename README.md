# Acoustic vs Neural Representations of French Vowels (L1 French and L1 Russian speakers)

Statistics project (Université Paris Cité, 2026). I compare three ways of representing the same speech sounds and ask how much phonetic structure each one captures:

- **acoustic features** from Praat (formants, Lobanov-normalised),
- **Whisper** (`whisper-medium`) encoder hidden states, layer 20,
- **wav2vec 2.0 XLS-R** (`wav2vec2-large-xlsr-53`) hidden states, layer 12.

The data is a phonetically aligned corpus of French read by native French speakers and Russian learners of French: 78 sentences, 19 speakers, about 23,000 phoneme tokens.

## Key results

Vowel classification with a nearest-centroid classifier, leave-one-speaker-out:

| Representation | Accuracy | Macro F1 |
|---|---|---|
| Acoustic (F1, F2) | 0.797 | 0.650 |
| Whisper, layer 20 | **0.940** | **0.907** |
| XLS-R, layer 12 | 0.851 | 0.801 |

- All differences between representations are significant (McNemar tests).
- Representational similarity (Mantel test): Whisper and XLS-R are very close to each other (r = 0.94), XLS-R is closer to acoustics (r = 0.73) than Whisper is (r = 0.55).
- Front/back vowel clusters are recovered by acoustics and XLS-R (ARI 0.59) but not by Whisper, whose geometry is organised more by vowel height.
- XLS-R recovers speaker gender almost perfectly, Whisper barely encodes it.
- The largest L1/L2 differences are on /y/, a vowel that does not exist in Russian.
- Mid vowels /e/ and /o/ are the hardest for all three representations.

Full analysis with mixed-effects models, permutation tests and figures: [`Muravyeva_report.pdf`](Muravyeva_report.pdf).


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
