# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform

# %%
# load data
df = load_transform()

df.head()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]

# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable?


#%%
# TODO: use your create_sample_split function here
# df = create_sample_split(...)
df = create_sample_split(df, "IDpol")
train = np.where(df["sample"] == 1)
test = np.where(df["sample"] == 0)
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]

# Create a sub-pipeline for numeric features with scaling and splines
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("spline", SplineTransformer(knots="quantile", include_bias=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
        ("num", numeric_pipeline, numeric_cols)
    ]
)
preprocessor.set_output(transform="pandas")

# Create the full pipeline with preprocessor and GLM estimator
model_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('estimate', GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True))
    ]
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.


model_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('estimate', LGBMRegressor(objective='tweedie', tweedie_variance_power=1.5))
    ]
)

model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
lgb_pipeline = Pipeline([
    ('regressor', LGBMRegressor(objective='tweedie', tweedie_variance_power=1.5))
])

param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate':[0.01, 0.02, 0.05, 0.1, 0.2]
}

cv = GridSearchCV(lgb_pipeline, param_grid, cv=5)
cv.fit(X_train_t, y_train_t, regressor__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)

# Let's compare the sorting of the pure premium predictions

#%%
# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()
#%%
#start of PS4
#task 1
print(df["BonusMalus"].unique())

# %%

bonus_group = np.sort(df["BonusMalus"].unique())
avg_claim = df.groupby("BonusMalus")["PurePremium"].mean()


plt.bar(bonus_group, avg_claim)
plt.xlabel("BonusMalus group")
plt.ylabel("Avg Claim")
plt.title("Avg claim for each BonusMalus group")
plt.show()

#%%

#print(X_train_t.shape)
#print(X_train_t.columns)
#print(np.where(X_train_t.columns == "BonusMalus"))

#preprocessor.fit(df[predictors].iloc[train])
#transformed_features = preprocessor.transform(df[predictors].iloc[train])
#n_features = transformed_features.shape[1]

#monotoncity_contrains = np.zeros(n_features)
#print(np.where(transformed_features.columns.str.contains("BonusMalus")))
#print(len(monotoncity_contrains))
# %%

monotoncity_contrains = np.zeros(61)
monotoncity_contrains[49:55] = 1

constrained_lgbm = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('estimate', LGBMRegressor(objective='tweedie',
                                    tweedie_variance_power=1.5,
                                    monotone_constraints = monotoncity_contrains))
    ]
)


param_grid = {
    'estimate__n_estimators': [100, 200, 300],
    'estimate__learning_rate':[0.01, 0.02, 0.05, 0.1, 0.2]
}

cv_constrained = GridSearchCV(constrained_lgbm, param_grid, cv=5)
cv_constrained.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)
# %%
# exercise 2

best_est = cv_constrained.best_estimator_

#retrain best model 
# Create properly preprocessed data for evaluation that matches the pipeline
X_train_pipeline = df[predictors].iloc[train]
X_test_pipeline = df[predictors].iloc[test]

# Get the preprocessor from the best estimator
pipeline_preprocessor = best_est.named_steps['preprocessor']

# Fit and transform the data using the same preprocessor as the pipeline
X_train_transformed = pipeline_preprocessor.fit_transform(X_train_pipeline)
X_test_transformed = pipeline_preprocessor.transform(X_test_pipeline)

best_est.fit(X_train_pipeline, y_train_t, 
            estimate__sample_weight=w_train_t,
            estimate__eval_set=[(X_train_transformed, y_train_t), (X_test_transformed, y_test_t)],
            estimate__eval_names=["train", "test"],
            estimate__eval_metric="rmse")

lgb.plot_metric(best_est['estimate'])
plt.show()
# %%
#exercise 3
from ps3.evaluation import evaluate_pred

exposure = df["Exposure"].iloc[test]
constrained = evaluate_pred(df_test["pp_t_lgbm_constrained"], y_test_t,exposure)
unconstrained = evaluate_pred(df_test["pp_t_lgbm"], y_test_t,exposure)
print(constrained.head())
print(unconstrained.head())
# %%
#exercise 4

import dalex as dx

exp_const = dx.Explainer(
    best_est, X_test_t, y_test_t, label="LGBM constrained"
)

exp_unconst = dx.Explainer(
    cv.best_estimator_, X_test_t, y_test_t, label="LGBM unconstrained"
)

pdp_unconst = exp_unconst.model_profile(type='partial')
pdp_const = exp_const.model_profile(type='partial')

pdp_const.plot()
pdp_unconst.plot()
#%%

pdp_const.plot(pdp_unconst)

# %%
#exercise 5

exp_glm = dx.Explainer(
    t_glm1, X_test_t, y_test_t, label="GLM"
)

row = X_test_t.iloc[[0]]

shap_lgbm = exp_const.predict_parts(
    row,
    type="shap"
)

shap_glm = exp_glm.predict_parts(
    row,
    type="shap"
)

shap_lgbm.plot()
shap_glm.plot()
shap_lgbm.plot(shap_glm)

# %%
