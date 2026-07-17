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

### §5.3 Limitations of agentic pipeline generation (Table 3)

#### Blocking for entity resolution (SIGMOD)

- Task code: [experiments/sigmod/](experiments/sigmod/)
- SemPipes: [rutgers/execute_sempipes_medium.py](experiments/sigmod/rutgers/execute_sempipes_medium.py), [execute_sempipes_medium_optimized.py](experiments/sigmod/rutgers/execute_sempipes_medium_optimized.py); contest baseline: [baseline/](experiments/sigmod/baseline/)
- Example agent artifacts: [sigmod_agents/](experiments/sigmod/sigmod_agents/)

#### High enrollment prediction (BEAVER)

- Task code: [experiments/beaver_enrollment/](experiments/beaver_enrollment/)
- Baseline: [baseline.py](experiments/beaver_enrollment/baseline.py)
- SemPipes: [pipeline.py](experiments/beaver_enrollment/pipeline.py), [enrollment_pipeline.py](experiments/beaver_enrollment/enrollment_pipeline.py); [trajectories.zip](experiments/beaver_enrollment/trajectories.zip)
- Example agent artifacts: [beaver_agents/](experiments/beaver_enrollment/beaver_agents/)

#### Parking violation prediction (NYC)

- Task code: [experiments/nyc_penalties/](experiments/nyc_penalties/)
- Baseline: [baseline.py](experiments/nyc_penalties/baseline.py)
- SemPipes: [sempipes.py](experiments/nyc_penalties/sempipes.py), [nyc_pipeline.py](experiments/nyc_penalties/nyc_pipeline.py); [trajectories.zip](experiments/nyc_penalties/trajectories.zip)
- Example agent artifacts: [nyc_agents/](experiments/nyc_penalties/nyc_agents/)

---

### §5.4 Comparison to specialized approaches (Table 4, Figure 6)

#### Feature generation vs CAAFE (Table 4)

- Folder: [experiments/caafe/](experiments/caafe/)
- CAAFE baseline: [caafe.py](experiments/caafe/caafe.py)
- SemPipes: [sempipes.py](experiments/caafe/sempipes.py), [sempipes_tabpfn.py](experiments/caafe/sempipes_tabpfn.py)
- Datasets / generated code: [data/](experiments/caafe/data/), [sempipes/](experiments/caafe/sempipes/)

#### Zero-shot feature extraction (Figure 6)

- Folder: [experiments/feature_extraction/](experiments/feature_extraction/)
- Modalities: [text_legal/](experiments/feature_extraction/text_legal/), [image_medical/](experiments/feature_extraction/image_medical/), [audio_env50/](experiments/feature_extraction/audio_env50/)
- Extracted features: [results/](experiments/feature_extraction/results/)

---

### §5.5 Impact of instruction quality (Table 5)

- Folder: [experiments/prompt_levels/](experiments/prompt_levels/)
  - House (`kaggle-house-a`): [house/](experiments/prompt_levels/house/)
  - Midwest (`skrub-midwest`): [midwest/](experiments/prompt_levels/midwest/)
  - TMDB (`kaggle-movie-b`): [tmdb/](experiments/prompt_levels/tmdb/)
- SIGMOD–rutgers: [experiments/sigmod/rutgers/](experiments/sigmod/rutgers/)

---

### §5.6 Benefits over global program synthesis

- Folder: [experiments/globalprogramsynthesis/](experiments/globalprogramsynthesis/)
- Synthesis attempts and errors: `gemini25pro_intermediate_*.py`, `gemini25pro_intermediate_*_errors.txt`, `gemini25pro_leakage_*.py`

---

### Additional agent failures

- [aide_failures_SIGMOD/](experiments/aide_failures_SIGMOD/)
- [swe_failures_SIGMOD/](experiments/swe_failures_SIGMOD/)
- [aide_failures_dataintegration/](experiments/aide_failures_dataintegration/)
- [aide_failures_fairness/](experiments/aide_failures_fairness/)
- [aide_failures_singlecelldata/](experiments/aide_failures_singlecelldata/)
- [single_cell_analysis/](experiments/single_cell_analysis/)
