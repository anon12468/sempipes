# SemPipes - Optimizable Semantic Data Operators for Tabular Machine Learning Pipelines

SemPipes is a research project to extend Python-based machine learning pipelines with semantic data operators. It is heavily relying on the awesome work from the [skrub](https://github.com/skrub-data/skrub) project!

## Experiments map

### §5.1 Semantic operators improve expert pipelines (Table 2)

#### Extraction of clinical features — `micromodels`

- Folder: [experiments/micromodels/](experiments/micromodels/)
- Original / reimplementation: [reimplemented.py](experiments/micromodels/reimplemented.py)
- SemPipes: [sempipes.py](experiments/micromodels/sempipes.py)
- Optimized: [sempipes_optimised.py](experiments/micromodels/sempipes_optimised.py), [_sempipes_state.json](experiments/micromodels/_sempipes_state.json)

#### Blocking for entity resolution — SIGMOD’22

- Folder: [experiments/sigmod/](experiments/sigmod/)
- **baseline-sigmod:** [baseline/baseline.py](experiments/sigmod/baseline/baseline.py) → [baseline_sempipes.py](experiments/sigmod/baseline/baseline_sempipes.py) → [baseline_sempipes_optimized.py](experiments/sigmod/baseline/baseline_sempipes_optimized.py)
- **rutgers:** [rutgers/execute.py](experiments/sigmod/rutgers/execute.py) → [execute_sempipes_medium.py](experiments/sigmod/rutgers/execute_sempipes_medium.py) → [execute_sempipes_medium_optimized.py](experiments/sigmod/rutgers/execute_sempipes_medium_optimized.py); [trajectories.zip](experiments/sigmod/rutgers/trajectories.zip)
- **sustech:** [sustech/EntityBlocking.py](experiments/sigmod/sustech/EntityBlocking.py), [EntityBlocking_sempipes.py](experiments/sigmod/sustech/EntityBlocking_sempipes.py), [FeatureExtracting_sempipes.py](experiments/sigmod/sustech/FeatureExtracting_sempipes.py), [Main_optimizable.py](experiments/sigmod/sustech/Main_optimizable.py)

#### Feature engineering — movie revenue (`kaggle-movie-a/b`)

- Folder: [experiments/tmdb_box_office_prediction/](experiments/tmdb_box_office_prediction/)
- Original: [kaggle.py](experiments/tmdb_box_office_prediction/kaggle.py), [kaggle2.py](experiments/tmdb_box_office_prediction/kaggle2.py)
- SemPipes: [kaggle_sempipes.py](experiments/tmdb_box_office_prediction/kaggle_sempipes.py), [kaggle_sempipes2.py](experiments/tmdb_box_office_prediction/kaggle_sempipes2.py)
- Optimized: [kaggle_sempipes_optimised.py](experiments/tmdb_box_office_prediction/kaggle_sempipes_optimised.py), [kaggle_sempipes_optimised2.py](experiments/tmdb_box_office_prediction/kaggle_sempipes_optimised2.py)

#### Feature engineering — house prices (`kaggle-house-a/b`)

- Folder: [experiments/house_prices_advanced_regression_techniques/](experiments/house_prices_advanced_regression_techniques/)
- Original: [kaggle.py](experiments/house_prices_advanced_regression_techniques/kaggle.py), [kaggle2.py](experiments/house_prices_advanced_regression_techniques/kaggle2.py)
- SemPipes: [kaggle_sempipes.py](experiments/house_prices_advanced_regression_techniques/kaggle_sempipes.py), [kaggle_sempipes2.py](experiments/house_prices_advanced_regression_techniques/kaggle_sempipes2.py)
- Optimized (CV): [kaggle_sempipes_optimised_cv.py](experiments/house_prices_advanced_regression_techniques/kaggle_sempipes_optimised_cv.py), [kaggle_sempipes_optimised2_cv.py](experiments/house_prices_advanced_regression_techniques/kaggle_sempipes_optimised2_cv.py)

#### Feature engineering — Scrabble (`kaggle-scrabble`)

- Folder: [experiments/scrabble_player_rating/](experiments/scrabble_player_rating/)
- Original: [kaggle.py](experiments/scrabble_player_rating/kaggle.py), [kaggle2.py](experiments/scrabble_player_rating/kaggle2.py)
- SemPipes: [kaggle_sempipes.py](experiments/scrabble_player_rating/kaggle_sempipes.py), [kaggle_sempipes2.py](experiments/scrabble_player_rating/kaggle_sempipes2.py)
- Optimized: [kaggle_sempipes_optimised.py](experiments/scrabble_player_rating/kaggle_sempipes_optimised.py), [kaggle_sempipes_optimised2.py](experiments/scrabble_player_rating/kaggle_sempipes_optimised2.py)

#### Data annotation for model debugging — `hibug`

- Folder: [experiments/hibug/](experiments/hibug/)
- Original: [hibug.py](experiments/hibug/hibug.py)
- SemPipes: [sempipes.py](experiments/hibug/sempipes.py)
- Optimized: [sempipes_optimized.py](experiments/hibug/sempipes_optimized.py), [sempipes_state.json](experiments/hibug/sempipes_state.json)

#### Data augmentation for fairness — `sivep`

- Folder: [experiments/sivep/](experiments/sivep/)
- Original: [oversampling.py](experiments/sivep/oversampling.py)
- SemPipes: [sempipes.py](experiments/sivep/sempipes.py)
- Optimized: [sempipes_optimised.py](experiments/sivep/sempipes_optimised.py), [_sempipes_state.json](experiments/sivep/_sempipes_state.json)

---

### §5.2 Optimization effectiveness of LLMs and search strategies (Figures 3–5)

- Folder: [experiments/colopro/](experiments/colopro/)
- Runner: [minibench.py](experiments/colopro/minibench.py)
- Aggregated results: [results/minibench.csv](experiments/colopro/results/minibench.csv)

---

### §5.3 Limitations of agentic pipeline generation (Tables 3 and 4)

For each task, the `*_agents/` folder holds the agents' generated solutions and logs, with one subfolder per agent (`aide/`, `claude/`, `cursor/`, `mle-star/`, `mlzero/`, `swe/`). It also holds the task description that every agent received (`goal.txt`).

#### Blocking for entity resolution (SIGMOD)

- Task code: [experiments/sigmod/](experiments/sigmod/)
- SemPipes: [rutgers/execute_sempipes_medium.py](experiments/sigmod/rutgers/execute_sempipes_medium.py), [execute_sempipes_medium_optimized.py](experiments/sigmod/rutgers/execute_sempipes_medium_optimized.py); contest baseline: [baseline/](experiments/sigmod/baseline/)
- Agent runs: [sigmod_agents/](experiments/sigmod/sigmod_agents/)

#### High enrollment prediction (BEAVER)

- Task code: [experiments/beaver_enrollment/](experiments/beaver_enrollment/)
- Baseline: [baseline.py](experiments/beaver_enrollment/baseline.py)
- SemPipes: [pipeline.py](experiments/beaver_enrollment/pipeline.py), [enrollment_pipeline.py](experiments/beaver_enrollment/enrollment_pipeline.py); [trajectories.zip](experiments/beaver_enrollment/trajectories.zip)
- Agent runs: [beaver_agents/](experiments/beaver_enrollment/beaver_agents/)

#### Parking violation prediction (NYC)

- Task code: [experiments/nyc_penalties/](experiments/nyc_penalties/)
- Baseline: [baseline.py](experiments/nyc_penalties/baseline.py)
- SemPipes: [sempipes.py](experiments/nyc_penalties/sempipes.py), [nyc_pipeline.py](experiments/nyc_penalties/nyc_pipeline.py); [trajectories.zip](experiments/nyc_penalties/trajectories.zip)
- Agent runs: [nyc_agents/](experiments/nyc_penalties/nyc_agents/)

#### Ablation: Cursor with and without baseline code (Table 4)

The Cursor runs are split first by setting, then by backing LLM:
- `gemini_flash`: Gemini-2.5-Flash
- `gemini_pro`: Gemini-2.5-Pro (only without baseline)
- `gemini_3.8_flash`: Gemini-3.8-Flash
- `gpt-5.6-sol`: GPT-5.6-Sol

The Cursor columns of Table 3 are the `gemini_flash` and `gemini_pro` runs without baseline.

| Task | Cursor w/o baseline | Cursor w/ baseline | Baseline code given to Cursor |
|---|---|---|---|
| SIGMOD | [no_baseline/](experiments/sigmod/sigmod_agents/cursor/no_baseline/) | [baseline/](experiments/sigmod/sigmod_agents/cursor/baseline/) | [rutgers/execute.py](experiments/sigmod/rutgers/execute.py) |
| BEAVER | [no_baseline/](experiments/beaver_enrollment/beaver_agents/cursor/no_baseline/) | [baseline/](experiments/beaver_enrollment/beaver_agents/cursor/baseline/) | [baseline.py](experiments/beaver_enrollment/baseline.py) |
| NYC | [no_baseline/](experiments/nyc_penalties/nyc_agents/cursor/no_baseline/) | [baseline/](experiments/nyc_penalties/nyc_agents/cursor/baseline/) | [baseline.py](experiments/nyc_penalties/baseline.py) |

---

### §5.4 Comparison to specialized approaches (Table 5, Figure 6)

#### Feature generation vs CAAFE and AutoGluon (Table 5)

- Folder: [experiments/caafe/](experiments/caafe/)
- CAAFE baseline: [caafe.py](experiments/caafe/caafe.py)
- AutoGluon baseline (`AutoMLPipelineFeatureGenerator` + `ArithmeticFeatureGenerator`): [autogluon_tracked.py](experiments/caafe/autogluon_tracked.py) (random forest), [autogluon_tabpfn_tracked.py](experiments/caafe/autogluon_tabpfn_tracked.py) (TabPFN)
- SemPipes: [sempipes.py](experiments/caafe/sempipes.py), [sempipes_tabpfn.py](experiments/caafe/sempipes_tabpfn.py)
- Datasets / generated code: [data/](experiments/caafe/data/), [sempipes/](experiments/caafe/sempipes/)

#### Zero-shot feature extraction (Figure 6)

- Folder: [experiments/feature_extraction/](experiments/feature_extraction/)
- Modalities: [text_legal/](experiments/feature_extraction/text_legal/), [image_medical/](experiments/feature_extraction/image_medical/), [audio_env50/](experiments/feature_extraction/audio_env50/)
- Extracted features: [results/](experiments/feature_extraction/results/)

---

### §5.5 Impact of instruction quality (Table 6)

- Folder: [experiments/prompt_levels/](experiments/prompt_levels/)
  - House (`kaggle-house-a`): [house/](experiments/prompt_levels/house/)
  - Midwest (`skrub-midwest`): [midwest/](experiments/prompt_levels/midwest/)
  - TMDB (`kaggle-movie-b`): [tmdb/](experiments/prompt_levels/tmdb/)
- SIGMOD–rutgers: [experiments/sigmod/rutgers/](experiments/sigmod/rutgers/)

---

### §5.6 Sensitivity to the number of prompt examples and data split sizes (Figures 7 and 8)

Both experiments use the `kaggle-house-a` (`house`), `kaggle-movie-b` (`tmdb`) and `skrub-midwest` (`midwest`) pipelines.

#### Number of prompt examples (Figure 7)

- Folder: [experiments/prompt_samples/](experiments/prompt_samples/)
- Runner: [run.py](experiments/prompt_samples/run.py) (sets the number of examples with `--n-prompt-samples`)
- Pipelines: [house_pipeline.py](experiments/prompt_samples/house_pipeline.py), [tmdb_pipeline.py](experiments/prompt_samples/tmdb_pipeline.py), [midwest_pipeline.py](experiments/prompt_samples/midwest_pipeline.py)

#### Training and validation set size (Figure 8)

- Folder: [experiments/split_size/](experiments/split_size/)
- Runner: [run.py](experiments/split_size/run.py). It holds out a fixed 25% test set and runs the `train` or `val` sweep, which uses 10%, 25%, 50% and 100% of the training or the validation split.
- Pipelines: [pipelines.py](experiments/split_size/pipelines.py)

---

### §5.7 Per-tuple LLM processing vs. code synthesis (Figure 9)

- Folder: [experiments/size_sweep/](experiments/size_sweep/)
- Pipelines: each runs with per-tuple LLM processing (`mode="llm"`) or with code synthesis, at three training set sizes.
  - `kaggle-movie-b`: [tmdb_pipeline.py](experiments/size_sweep/tmdb_pipeline.py)
  - `scrabble`: [scrabble_pipeline.py](experiments/size_sweep/scrabble_pipeline.py)
  - `BC5CDR`: [bc5cdr_pipeline.py](experiments/size_sweep/bc5cdr_pipeline.py)
  - `bearing`: [bearing_pipeline.py](experiments/size_sweep/bearing_pipeline.py)

---

### Additional experiments

#### Global program synthesis (§5.6 of the initial submission)

- Folder: [experiments/globalprogramsynthesis/](experiments/globalprogramsynthesis/)
- Synthesis attempts and errors: `gemini25pro_intermediate_*.py`, `gemini25pro_intermediate_*_errors.txt`, `gemini25pro_leakage_*.py`

#### Agent failures

- [aide_failures_SIGMOD/](experiments/aide_failures_SIGMOD/)
- [swe_failures_SIGMOD/](experiments/swe_failures_SIGMOD/)
- [aide_failures_dataintegration/](experiments/aide_failures_dataintegration/)
- [aide_failures_fairness/](experiments/aide_failures_fairness/)
- [aide_failures_singlecelldata/](experiments/aide_failures_singlecelldata/)
- [single_cell_analysis/](experiments/single_cell_analysis/)
