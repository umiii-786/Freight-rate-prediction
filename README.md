# Freight Rate Prediction - MLOps pipeline

Predicts the posted rate (`posted_rate`) of freight loads. Converted from the notebook in
`notebooks/` into a reproducible pipeline: **DVC** (data + pipeline versioning),
**MLflow** (experiment tracking), central **logging**, custom **exception handling**.

## Pipeline (dvc.yaml)

```
data/external ──► data_ingestion ──► data_preprocessing ──► build_features
                                                                  │
                                      ┌───────────────────────────┤
                                      ▼                           ▼
                               evaluate_models              train_model ──► predict_model
                              (time-based CV, 3 models)    (final fit)      (validation + Dec)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the 4 CSV files in data/external/ then follow BUILD_GUIDE.md (dvc add, dvc stage add ...)
dvc repro                                              # run / re-run the pipeline
mlflow ui --backend-store-uri sqlite:///mlflow.db      # compare experiments
dvc metrics show                                       # CV metrics of the selected model
```

## Layout

| Path | Purpose |
|---|---|
| `src/config.py`, `params.yaml` | paths, constants, all tunable values |
| `src/logging_config.py`, `src/exception.py` | logging + error handling |
| `src/data/` | ingestion (schema checks), preprocessing (cleaning) |
| `src/features/` | custom transformer, sklearn preprocessor, feature building |
| `src/model/` | model factory, time-based CV, evaluate / train / predict stages |
| `src/tracking.py` | MLflow setup (URI + credentials come from env vars) |
| `reports/` | metrics, comparison table/plot, predictions |
