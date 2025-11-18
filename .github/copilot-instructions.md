# Copilot Instructions for PS3 - Claims Modeling Exercise

## Project Overview
This is a **Python package for actuarial claims modeling** - an academic exercise (Problem Set 3) building machine learning models for insurance claim prediction using French Motor Third-Party Liability (MTPL) data.

## Architecture & Data Flow
- **`ps3/data/`**: Data loading and preprocessing pipeline
  - `_load_transform.py`: Fetches OpenML data, applies domain-specific transforms (claim capping at 100k, exposure limits)
  - `_sample_split.py`: Hash-based train/test splitting using policy IDs (deterministic)
- **`ps3/preprocessing/`**: Custom sklearn transformers  
  - `_winsorizer.py`: Quantile-based outlier clipping (incomplete implementation)
- **`analyses/`**: Main analysis notebooks/scripts
  - `ps3_script.py`: Primary modeling workflow with GLM baselines and TODO tasks

## Critical Setup Commands
```bash
# Environment setup (mamba preferred over conda)
mamba env create
conda activate ps3
pre-commit install
pip install --no-build-isolation -e .

# Testing (current pytest config requires coverage adjustment)
pytest tests/ --no-cov  # Bypass coverage requirement
```

## Key Patterns & Conventions

### Data Processing
- **Target variable**: `PurePremium = ClaimAmountCut / Exposure` (claims per unit exposure)
- **Sample weights**: Always use `Exposure` as sample weights in model fitting
- **Categorical encoding**: Use `dask_ml.Categorizer` for GLMs, `OneHotEncoder` for sklearn pipelines
- **ID-based splitting**: Deterministic train/test split via hashed policy IDs (prevents data leakage)

### Model Architecture
- **GLM baseline**: Tweedie distribution (power=1.5) with L1 regularization via `glum`
- **Feature engineering**: Splines for continuous vars (`BonusMalus`, `Density`), one-hot for categoricals
- **Pipeline pattern**: `ColumnTransformer` → `GeneralizedLinearRegressor` chains

### Code Organization
- Use **sklearn-style transformers** (inherit from `BaseEstimator`, `TransformerMixin`)
- **Import style**: Local imports from `ps3.data`, `ps3.preprocessing`
- **TODO pattern**: Active development areas marked with `# TODO:` comments

## Development Workflow
- **Pre-commit hooks**: black, isort, mypy configured (see `pyproject.toml`)
- **Test structure**: Parametrized tests with `pytest.mark.parametrize`
- **Coverage**: Configured for `ps3` package only (`--cov=ps3`)

## Domain-Specific Context
- **Insurance modeling**: Claims frequency/severity separation is standard practice
- **Tweedie GLM**: Industry standard for pure premium modeling (handles zero-inflated, continuous positive claims)
- **Exposure weighting**: Essential for insurance data - represents policy duration/size

## Current TODOs
1. Complete `Winsorizer.fit()` method (missing `return self`)
2. Fix `Winsorizer.transform()` to clip values rather than filter
3. Implement spline pipeline in `analyses/ps3_script.py`
4. Add sample split function call in analysis script

## Testing Notes
- Run tests with `pytest tests/ --no-cov` due to missing pytest-cov dependency
- Winsorizer tests expect clipping behavior, not filtering
- Use `np.random.RandomState(0)` for reproducible test data