# Tomato-leaf field screening pipeline

A mobile field unit photographs tomato leaves. Captures are **noisy** and **unevenly lit**, and **bandwidth is limited**. This project builds and measures the whole chain:

```
field capture ─► [GATE 1 exposure] ─► restore ─► enhance ─► JPEG (transmit) ─► decode
             ─► segment + morphology ─► [GATE 2 mask area] ─► CNN ─► P(diseased)
             ─► decision agent: CLEAR / FLAG / RE-IMAGE  (asymmetric cost)
```

**Live screening lab:** <https://sanskrit-adk.github.io/plant-screening-pipeline/>. Upload a leaf photo, or pick a held-out PlantVillage test leaf, and watch all seven phases run in your browser. Experiment results are at [`results.html`](https://sanskrit-adk.github.io/plant-screening-pipeline/results.html).

Missing a diseased leaf (CLEAR when diseased) costs **50**, flagging a healthy leaf costs **1**, and a re-image costs **2** plus the second decision. Every stage can be switched off on its own, and the experiments measure what each stage is worth.

---

## 1. Setup in VS Code

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

Then press **Ctrl+Shift+P → "Python: Select Interpreter" → `.venv`**.

**Optional, for an NVIDIA GPU:** install the CUDA build of PyTorch *before* `requirements.txt`, using the command from <https://pytorch.org/get-started/locally/>. For example:
`pip install torch --index-url https://download.pytorch.org/whl/cu124`.
`train.device: auto` picks CUDA, then Apple MPS, then CPU.

`.vscode/launch.json` has one **debugpy** launch configuration per experiment (Run and Debug panel, F5). Each one runs `python -m src.experiments` from the project root.

---

## 2. Data sources (`data.source` in `config.yaml`)

The dataset is **PlantVillage**. It has about 54k leaf images on a plain background in 38 classes. Only the **10 tomato classes** are used, as a **binary** task.

**Dataset links**

| resource | link |
|---|---|
| Hugging Face (what `source: hf` loads; includes the segmented masks) | <https://huggingface.co/datasets/mohanty/PlantVillage> |
| Official GitHub repo (layout used by `source: folder`) | <https://github.com/spMohanty/PlantVillage-Dataset> |
| TensorFlow Datasets page (class list and size info) | <https://www.tensorflow.org/datasets/catalog/plant_village> |
| Original dataset paper: Hughes & Salathé (2015), *An open access repository of images on plant health to enable the development of mobile disease diagnostics* | <https://arxiv.org/abs/1511.08060> |
| Deep-learning paper on the dataset: Mohanty, Hughes & Salathé (2016), *Frontiers in Plant Science* 7:1419 | <https://doi.org/10.3389/fpls.2016.01419> |

The tomato classes map to the binary labels like this:

| label | classes |
|---|---|
| 0 healthy | `Tomato___healthy` |
| 1 diseased | Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot, Spider_mites Two-spotted_spider_mite, Target_Spot, Tomato_mosaic_virus, Tomato_Yellow_Leaf_Curl_Virus |

Classes are matched case-insensitively. A name that starts with "tomato" and contains "healthy" is 0; any other tomato class is 1. All healthy images are kept, plus at most `pos_to_neg_ratio` (2.0) × as many diseased images, sampled with the seeded RNG.

| source | what it does |
|---|---|
| `hf` (default) | Hugging Face [`mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage), colour images. It uses the dataset's own **leaf-grouped train/test split**. Validation is carved out of train with `GroupShuffleSplit` on `leaf_id`, so no leaf appears in two splits. The repo is a legacy *loading-script* dataset, which `datasets>=4` no longer runs. When `load_dataset` fails, the loader reads the same repo files directly: `data.zip` (**about 2 GB, downloaded once** into the HF cache), `splits/color_{train,test}.txt` and `leaf_grouping/leaf-map.json`. It uses the same leaf-id logic as the official script. The zip also contains `raw/segmented/`, so ground-truth leaf masks are available for IoU. |
| `folder` | A local copy of the [GitHub repo](https://github.com/spMohanty/PlantVillage-Dataset) layout: `data/raw/color/<ClassName>/*.jpg` plus, optionally, `data/raw/segmented/<ClassName>/<stem>_final_masked.jpg`. Segmented images give GT masks (`gray > 10`). Groups are the filename part before `___`, split with `GroupShuffleSplit` into train/val/test. That prefix is a per-photo UUID. For true per-leaf grouping, place `leaf-map.json` (from the HF repo) at `data/raw/leaf_grouping/leaf-map.json` and it is used automatically. |
| `synthetic` | About 600 fake leaves: a green ellipse on grey paper, 3–7 brown lesions on diseased leaves, 3×3 blur, ~2:1 diseased:healthy, with GT masks. **Only for smoke tests.** |

Images are converted to RGB, resized to 128×128 (`INTER_AREA`) and cached as uint8 in `data/cache/tomato*.npz`. The cache is rebuilt automatically when data settings change. Split fractions are val 0.15 and test 0.15. The `hf` source keeps the official test split.

You can switch the source without editing the config: add `--source synthetic` (or `folder` / `hf`) to any command.

---

## 3. Running

Always run from the project root. **Start with the quick smoke test:**

```bash
python -m src.experiments --exp all --quick --source synthetic   # ~3 min, no download
python -m src.experiments --exp all --quick                      # quick run on real PlantVillage
```

Full experiments:

```bash
python -m src.experiments --exp figures       # illustrative figures (fast)
python -m src.experiments --exp denoise       # objective PSNR/SSIM + end-to-end per denoiser
python -m src.experiments --exp compression   # JPEG sweep, one model trained on lossless
python -m src.experiments --exp compression --retrain   # retrain per JPEG quality
python -m src.experiments --exp ablation      # 6 configs x 3 seeds = 18 CNNs
python -m src.experiments --exp propagation   # harsh lighting x gates on/off
python -m src.experiments --exp all           # figures, denoise, compression, ablation, propagation
python -m src.experiments --exp ablation --config my_config.yaml
```

`--quick` uses 200 train / 80 val / 80 test images, 2 epochs, patience 2, one seed, JPEG qualities [75, 30, 5], denoisers [none, median, nlm], σ [20] and `latency_n` 10.

**Runtime:** the ablation trains **18 CNNs** (6 configs × 3 seeds), which takes about **1–2 hours on CPU**. To save time, set `experiments.seeds: [42]`. Denoising trains one CNN per method, and the compression sweep trains one (or 8 with `--retrain`).

---

## 4. Files

| file | purpose |
|---|---|
| `config.yaml` | every setting (seed, data, degradation, pipeline, training, costs, gates, experiments) |
| `src/utils.py` | paths (`ROOT`, `RESULTS`, `FIGURES`), `load_config`, `set_seed`, `get_device`, `ensure_dirs` |
| `src/data.py` | the three data sources, tomato binary mapping, balancing, leaf-grouped splits, caching |
| `src/degrade.py` | reproducible field degradation (illumination field, noise, salt & pepper) with a noise-free reference |
| `src/stages.py` | restore / enhance / compress / segment_mask / apply_mask / iou |
| `src/pipeline.py` | `PipelineConfig`, safety gates, `run_one`, parallel `process`, `benchmark_latency` |
| `src/model.py` | `SmallCNN`, training with early stopping, temperature scaling, latency |
| `src/decision.py` | cost-sensitive CLEAR / FLAG / RE-IMAGE agent, threshold search, baselines |
| `src/experiments.py` | all experiments, figures and the CLI |
| `src/export_web.py` | trains the full pipeline and exports the model, samples and parity data for the website |
| `tools/web_parity.mjs` | Node check that the JavaScript port reproduces the Python pipeline |
| `docs/` | the GitHub Pages site (screening lab + results page) |
| `results/` | CSV / JSON outputs |
| `figures/` | PNG outputs |

---

## 5. Method details

### 5.1 Degradation (`src/degrade.py`)
For a clean image *I* and seed `rng = default_rng([seed, image_idx, attempt])` (attempt 0 = first photo, 1 = re-image):

```
B   = GaussianBlur(I, blur_ksize)                      (optional)
G   = normalise_[gmin,gmax]( a·linear_gradient(θ) + (1−a)·radial_vignette(cx,cy) ),  a ~ U(0.3,0.7)
REF = clip(G ⊙ B)                                      noise-free, unevenly lit reference
D   = clip(G ⊙ B + N(0, σ²))  then salt & pepper with probability sp_prob
```

The defaults are σ = 20, G ∈ [0.4, 1.2] and no blur or salt & pepper. Each image always gets the same degradation for a given attempt, whatever the pipeline config, so configurations are compared on identical inputs. `REF` lets restoration be scored **objectively**.

### 5.2 Stages (`src/stages.py`)
- **Restoration (objective):** `none`, `gaussian` (5×5), `median` (5), `bilateral` (7, 50, 50), `nlm` (`fastNlMeansDenoisingColored`, h = hColor = 12, template 7, search 21) or `wavelet` (BayesShrink, soft, YCbCr, `rescale_sigma`).
- **Enhancement (subjective):** `none`, `clahe` (clip 2.0, 8×8 tiles, LAB L channel), `histeq` (YCrCb Y channel) or `gamma` (0.7 LUT). It only touches luminance, so leaf colour is kept.
- **Compression:** JPEG at quality *q*. The receiver sees the decoded image. `jpeg_q: null` means lossless: PNG level 9 is encoded only to measure the size. The compression ratio is `H·W·3 / bytes`.
- **Segmentation + morphology:** HSV saturation (0 where V < 25) → 5×5 Gaussian → Otsu → open (5×5 ellipse) → close (7×7 ellipse, ×2) → largest connected component → hole filling (flood fill from a padded corner, invert, OR). `apply_mask` zeroes the background, crops to the bounding box (+4 px) and resizes back to 128.

### 5.3 Safety gates (`src/pipeline.py`)
- **Gate 1 (exposure):** the mean grey level of the raw capture must be within [35, 235].
- **Gate 2 (segmentation):** the mask area fraction must be within [0.05, 0.95]. It always passes when segmentation is off.

### 5.4 CNN (`src/model.py`)
- **Architecture:** 4 × [Conv3×3 (no bias) → BN → ReLU → MaxPool2] with 32-64-128-128 channels → global average pool → Dropout 0.3 → Linear(128, 2).
- **Input:** normalised as `(x/255 − 0.5)/0.5`. Augmentation is random horizontal/vertical flips plus rot90.
- **Training:** Adam (lr 1e-3, batch 64) with class-weighted cross-entropy (weights `N/(2·count)`). Early stopping on validation loss (patience 4, max 15 epochs) restores the best weights.
- **Calibration:** temperature scaling. T is fitted on validation by LBFGS over log T and clamped to [0.05, 20].

### 5.5 Decision rule (`src/decision.py`)
Given a calibrated *p* = P(diseased):

- E[CLEAR] = p · C_FN and E[FLAG] = (1 − p) · C_FP, so the **Bayes threshold** is C_FP / (C_FP + C_FN) = 1/51 ≈ 0.0196.
- **First photo:** if gate 1 or gate 2 fails, or t_low ≤ p₁ ≤ t_high → **RE-IMAGE**. Otherwise **FLAG** if p₁ > t_high, else **CLEAR**.
- **Re-image:** a failed gate → **FLAG** (a failed gate never clears). Otherwise FLAG if p₂ > Bayes threshold.
- **Threshold choice:** (t_low, t_high) are chosen on **validation** by exhaustive search over unique({0, Bayes, 1} ∪ 101 quantiles of p₁). The search minimises mean cost and breaks ties toward a lower re-image rate. Test is simulated **once**.
- **Baselines** (`results/baseline_policies.json`): argmax (0.5) and the single Bayes threshold, both without re-imaging.

### 5.6 Ablation protocol
Six configurations are compared: `full`, `no_restoration`, `no_enhancement`, `no_compression` (lossless), `no_segmentation` and `cnn_only` (all stages off).

- **Training:** every configuration × seed is **retrained** from scratch. Temperature and thresholds are re-chosen on validation for each one.
- **Inputs:** the test set and its degradations are identical across configs.
- **Cost difference:** per-sample test cost is averaged across seeds, and the difference to `full` gets a paired bootstrap 95% CI (2000 resamples). `measurable_change` is true when the CI excludes 0.

### 5.7 Compression protocol
- **Model:** one CNN is trained on the lossless full pipeline. With `--retrain`, one CNN is trained per quality.
- **Sweep:** for each q in [lossless, 95, 75, 50, 30, 15, 10, 5], temperature and thresholds are re-chosen on validation (processed at that q), and test accuracy, sensitivity, cost and bytes are measured. PSNR is computed between compressed and uncompressed preprocessed images (200 test images).
- **Knee:** the first compression ratio (ascending) where CNN accuracy **or** sensitivity drops more than `degrade_tolerance_pp` (1.0 pp) below lossless.
- **Recommended operating point:** the highest ratio before the knee.

---

## 6. Results (real PlantVillage tomato, seed 42)

These come from `python -m src.experiments --exp all` on the Hugging Face data. The run took 110 minutes on CPU.

**Test set:** 990 held-out leaves (660 diseased, 330 healthy), grouped by leaf. The test set was used **once**, after early stopping, temperature and thresholds had been chosen on validation. **Settings:** degradation σ = 20, illumination gain 0.4–1.2; default pipeline NLM → CLAHE → JPEG q50 → segmentation → CNN.

### 6.1 Stage ablation (each model retrained; Δ cost with a paired-bootstrap 95% CI)

| config | cost / leaf | Δ vs full [95% CI] | sensitivity | specificity | CNN acc | AUC | bytes / image | ms / image |
|---|---|---|---|---|---|---|---|---|
| **full** | **0.259** | – | 99.5% | 71.5% | 95.2% | 0.990 | 3 041 | 88.7 |
| no restoration | 0.283 | +0.024 [−0.22, +0.27] | 99.5% | 64.2% | 96.5% | 0.994 | 5 916 | 7.1 |
| no enhancement | 0.268 | +0.009 [−0.21, +0.22] | 99.7% | 56.1% | 94.7% | 0.988 | 2 152 | 91.5 |
| no compression | 0.223 | −0.035 [−0.22, +0.12] | 100% | 40.3% | 93.7% | 0.985 | 35 066 | 102.3 |
| no segmentation | 0.086 | **−0.173 [−0.37, −0.02]** | 100% | 77.9% | 98.9% | 0.999 | 3 041 | 95.7 |
| CNN only (raw) | 0.107 | −0.152 [−0.34, 0.00] | 100% | 71.5% | 97.8% | 0.999 | 42 681 | 7.6 |

- **Only one change is measurable:** removing segmentation. Its CI excludes 0, and removing it *lowers* the cost. PlantVillage leaves sit on a plain, uniform background, which the CNN handles well. The Otsu mask (IoU 0.74 against the GT masks) sometimes cuts away diseased tissue.
- **Restoration, enhancement and compression are within noise.** Their CIs straddle 0. Because one missed leaf costs 50, a single FN moves the cost by 0.05 per leaf, so the CIs are wide.
- **Compression costs nothing measurable but saves 11.5× bandwidth.**

### 6.2 Accuracy vs compression ratio (trained once on lossless, then tested at each JPEG quality)

| JPEG q | lossless | 95 | 75 | 50 | **30** | 15 | 10 | 5 |
|---|---|---|---|---|---|---|---|---|
| ratio vs raw RGB | 1.4:1 (PNG) | 5.1:1 | 11.2:1 | 16.2:1 | **21.4:1** | 29.5:1 | 34.4:1 | 42.6:1 |
| bytes / image | 35 066 | 9 576 | 4 387 | 3 041 | **2 301** | 1 666 | 1 430 | 1 153 |
| PSNR (dB) | ∞ | 37.5 | 30.5 | 27.1 | **25.3** | 23.7 | 22.8 | 21.0 |
| CNN accuracy | 93.7% | 94.8% | 94.3% | 94.4% | **93.1%** | 84.2% | 85.1% | 67.7% |
| sensitivity | 100% | 100% | 99.7% | 100% | **100%** | 100% | 99.2% | 99.8% |
| cost / leaf | 0.223 | 0.148 | 0.236 | 0.163 | **0.187** | 0.252 | 0.438 | 0.500 |

- **Knee:** q15 (29.5:1), where accuracy falls more than 1 percentage point below lossless.
- **Recommended operating point:** **q30, 21:1, 2.3 KB per leaf**, with no loss of accuracy or sensitivity.
- Beyond the knee, the temperature rises (T = 1.9 at q15, 11.4 at q5). The agent then simply flags almost everything, so sensitivity stays high while the cost climbs.

### 6.3 Accuracy vs denoising method

| method | PSNR σ=10 / 20 / 30 (dB) | SSIM σ=20 | ms / image | CNN acc (retrained, σ=20) | sensitivity | cost / leaf | total ms / image |
|---|---|---|---|---|---|---|---|
| none | 28.2 / 22.2 / 18.7 | 0.54 | 0.0 | **96.5%** | 99.5% | 0.283 | 10.8 |
| Gaussian 5×5 | 26.8 / 25.9 / 24.9 | 0.62 | 0.1 | 91.9% | 100% | **0.177** | 7.8 |
| median 5×5 | 25.6 / 24.9 / 24.1 | 0.52 | 0.3 | 91.1% | 100% | 0.376 | 7.5 |
| bilateral | **31.4 / 27.7** / 22.6 | **0.74** | 0.6 | 95.4% | 99.7% | 0.220 | 9.4 |
| non-local means | 27.4 / 27.0 / 24.0 | 0.66 | 80 | 95.2% | 99.5% | 0.259 | 95.1 |
| wavelet | 29.3 / 26.8 / **25.5** | 0.67 | 6 | 93.5% | 99.7% | 0.304 | 13.2 |

- **Higher PSNR does not mean better decisions.** Bilateral wins on PSNR and SSIM at σ ≤ 20, yet the CNN trained on *un-denoised* images has the highest accuracy, because it learns to ignore the noise.
- **NLM accounts for 82 of the full pipeline's 89 ms.** Bilateral gives about the same accuracy in about 9 ms, so it is the better field choice.

### 6.4 Decision layer and error propagation

**Policies on the full pipeline's test set:**

| policy | cost / leaf | FN | FP | sensitivity |
|---|---|---|---|---|
| argmax at 0.5 | 1.682 | 33 | 15 | 95.0% |
| Bayes threshold C_FP/(C_FP+C_FN) = 0.0196 | 0.203 | 0 | 201 | 100% |
| validation-tuned CLEAR / FLAG / RE-IMAGE | 0.259 | 3 | 94 | 99.5% |

- **Cost-aware thresholds cut the cost about 6.5×** compared with argmax.
- **On this test set the fixed Bayes threshold did slightly better than the validation-tuned one.** We report the validation-tuned result because it is the protocol we committed to.
- **The search chose an empty RE-IMAGE band** (t_low = t_high = 0.073), so re-imaging only happens when a safety gate fails.

**Error propagation:**

- **Harsh lighting** (gain down to 0.12) raises the cost from 0.259 to 0.287. All of the extra cost is extra false positives (94 → 118).
- **The gates fired on 6 of 990 normal and 8 of 990 harsh images.** None of those leaves was then missed. Gates add a small re-image cost and did not change the FN count here.

### 6.5 Latency budget (full pipeline, CPU, ms per image)

| restore (NLM) | enhance (CLAHE) | JPEG enc+dec | segment | CNN | **total** |
|---|---|---|---|---|---|
| 82.2 | 0.4 | 0.3 | 1.1 | 4.7 | **88.7** |

**Caveat:** a single seed (`experiments.seeds: [42]`). Set `seeds: [42, 43, 44]` in `config.yaml` to add between-seed standard deviations, which roughly doubles the runtime because trained models are cached in `results/model_cache/`.

---

## 7. Outputs → assignment questions

| question | output |
|---|---|
| What is each stage worth? Is the change measurable? | `results/ablation_summary.csv` (`delta_cost_vs_full`, CI, `measurable_change`), `results/ablation_runs.csv`, `figures/ablation.png` |
| Does the cost-aware agent beat naive policies? | `results/baseline_policies.json` (argmax 0.5 vs Bayes threshold vs CLEAR/FLAG/RE-IMAGE) |
| How far can we compress before diagnosis degrades? | `results/compression_sweep.csv`, `results/compression_summary.json` (knee, recommended q), `figures/compression_sweep.png`, `figures/jpeg_artifacts.png` |
| Which denoiser, and does a better PSNR mean better decisions? | `results/denoise_objective.csv` (PSNR/SSIM/ms vs σ), `results/denoise_end_to_end.csv`, `figures/denoise.png` |
| Restoration vs enhancement (objective vs subjective) | `figures/restoration_vs_enhancement.png` |
| How does segmentation work? | `figures/segmentation_steps.png` (with IoU when GT masks exist) |
| How do errors propagate, and do the safety gates help? | `results/error_propagation.csv` (cost, FN, gate failures, P(disease) when gates fail, misses among gate failures, segmentation IoU) |
| Latency budget per stage | `ms_restore … ms_cnn, ms_total` columns in the ablation and denoise CSVs |

---

## 8. The website (`docs/`, GitHub Pages)

`docs/index.html` is an interactive **screening lab**. Everything runs client-side in plain JavaScript, with no server and no framework:

| phase | course topic | in the browser |
|---|---|---|
| 1 | fundamental steps: acquisition | field degradation (illumination field + Gaussian noise), safety gate 1 |
| 2 | restoration (objective) | Gaussian / median / bilateral / non-local means, scored by PSNR and SSIM against the noise-free reference |
| 3 | enhancement (subjective) | CLAHE / histogram equalisation / gamma, with histograms, RMS contrast and entropy |
| 4 | compression | real JPEG through the browser's encoder: bytes, ratio, PSNR, error map |
| 5 | morphology + segmentation | saturation → Otsu → opening → closing → largest component → hole fill, safety gate 2 |
| 6 | recognition, CNN basics | SmallCNN forward pass with feature maps of every block, logits, temperature-scaled softmax |
| 7 | utility-based agent | expected cost of CLEAR / FLAG, RE-IMAGE band on a log-odds axis, a second photo when asked; editing the costs re-runs the threshold search on the validation set |

| file | role |
|---|---|
| `docs/js/imgproc.js` | port of `src/degrade.py` and `src/stages.py`, following OpenCV's algorithms, kernels, border modes and rounding |
| `docs/js/cnn.js` | SmallCNN inference (BatchNorm folded into the convolutions) |
| `docs/js/agent.js` | port of `src/decision.py`, including the threshold search |
| `docs/js/app.js`, `docs/css/app.css` | page logic and styling |
| `docs/model/` | exported weights, temperature, thresholds, test metrics and validation probabilities |
| `docs/samples/` | one held-out test photo per tomato class |
| `docs/results.html` | experiment results page |

**Rebuilding the model and checking the port:**

```bash
python -m src.export_web        # train the full pipeline on real data, write docs/model, docs/samples, results/web_check
node tools/web_parity.mjs       # compare every JS stage and the end-to-end accuracy with Python
```

To preview locally, serve `docs/` over HTTP (`python -m http.server -d docs 8000`) and open <http://localhost:8000>. Opening the file directly does not work, because browsers block `fetch` from `file://`.

---

## 9. Reproducibility

- **Seeding:** `set_seed` fixes Python, NumPy and PyTorch (CPU and CUDA), sets `PYTHONHASHSEED`, and sets `cudnn.deterministic = True` and `benchmark = False`.
- **Data loading:** the DataLoader shuffles with a seeded `torch.Generator` and `num_workers = 0`.
- **Degradation:** keyed on `(seed, global image index, attempt)`, so it is identical across configs and runs, and independent of joblib scheduling.
- **Splits:** leaf-grouped (no leaf in two splits) and derived from `seed`. The data cache stores the settings that produced it and is rebuilt if they change.
- **No test leakage:** all model selection (early stopping, temperature, thresholds) uses validation only.
- **Remaining non-determinism:** bit-level differences can remain across hardware and library versions (e.g. GPU vs CPU kernels, OpenCV builds). Latency numbers are machine-dependent.
