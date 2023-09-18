
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path
import os
import glob
import datetime
import json
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split


# Dataset

# The datasets contain info on the prices of used cars in the UK.
filelist = glob.glob("car-data/*.csv")

# Preprocessing

# These are the only preprocessing steps you should perform for now.

# The `car-data` directory must be in the current working directory (most likely in the same spot as
# the `assignment13.py` file).
# 1. Read in all the datasets and combine them into one large dataset.
# 2. Rename the `tax(£)` column to be `tax` so it all aligns.
# 3. Include a column called `make` which should be the car's make (file name without suffix).
# 4. Do not change the data or column names otherwise.
df = pd.DataFrame()
for f in filelist:
    newdf = pd.read_csv(f)
    newdf['make'] = f.split('.')[0][9:]
    df = pd.concat([df,newdf])
    
df = df.reset_index().drop(columns = "index")
df["realtax"] = df["tax"].fillna(0) + df["tax(£)"].fillna(0)
df = df.drop(["tax", "tax(£)"],axis=1).rename(columns={"realtax":"tax"})




# Question 1

# Let's do some cleaning!
# * Drop the rows that contain `NaN` values.
# * Filter out cars that were built after 2023.
# * Remove any leading/trailing whitespace on the string columns
# * Convert the 4 string columns to categorical unordered columns.
# * Sort the *columns* in ascending order

# Sort the cleaned dataframe by the values of `make`, `model`, `price`, and `mileage` in that order
# in ascending order and submit the first 100 rows.

cleaned = df.dropna()
cleaned = cleaned.loc[cleaned["year"] < 2023]
cleaned = cleaned.applymap(lambda x: x.strip() if isinstance(x, str) else x)
strings = cleaned[["model","transmission","fuelType","make"]]
cleaned[["model","transmission","fuelType","make"]] = cleaned[
    ["model","transmission","fuelType","make"]].apply(lambda x: x.astype('category'))

cleaned = cleaned.sort_index(axis=1)
q1 = cleaned.sort_values(["make","model","price","mileage"]).head(100)


# Question 2

# How much memory usage was saved by *cleaning and converting* the string columns to categorical
# ones in kilobytes (kb) (base 10)?

conversion = (df[["model","transmission","fuelType","make"]].memory_usage())/1000 - (cleaned[["model","transmission","fuelType","make"]].memory_usage())/1000
q2 = sum(conversion[1:5])

# Question 3

# For each `make`, what is the average car price?
# Submit a Series where the index is the car make and the values are the mean price in ascending
# order.

q3 = cleaned.groupby("make")["price"].mean().sort_values()


# Question 4

# A bit more cleaning...
# We don't want to use car models that only have a very few data points.
# Filter out models that had less than 10 data points and be sure to remove those unused categories
# as well.

# Submit a Series of the counts for each model for all models that had at least 10 occurrences in
# descending order with the model as the index.
# Be sure to filter out models who had less than 10 data points before moving on to question 5.

models = cleaned["model"].value_counts()
models = models.loc[models >= 10]
clean = cleaned.loc[cleaned["model"].isin(models.index)]
q4 = models


# Question 5

# Now before we do any more analysis (we probably could have done this at the very beginning), let's
# split the data after question 4 into a training and test set.
# This is important to prevent test data leakage.
# Use the `train_test_split` function with `random_state=1` to select 90% of the data to be in the
# training set.
# The remaining rows should be in the test set.

# For each car model, find the proportion of values that ended up in the training set.
# Submit a Series with the model as the index and the values as the proportion that ended up in the
# training set sorted by index ascending.

# The resulting train and test set will be used for the other problems.
train_df, test_df = train_test_split(clean, test_size=.1, random_state=1)
train_mods = train_df["model"].value_counts()
q5 = (train_mods / models).dropna()


# Question 6

# Let's train a model using the statsmodels formula api.
# Create an Ordinary Least Squares model to predict the `price` based on `transmission`, `mileage`,
# `make`, and `mpg` (with an intercept).
# Remember to use only the training set to fit the model.
# Do not change the names of the columns.

# Submit a Series with the parameter names as the index and the parameter estimates as the values
# sorted by index.
# Note how the categorical variables are automatically converted to be one hot encoded (indicator
# variables).

reg = smf.ols("price ~ transmission + mileage + make + mpg + 1", train_df).fit()

q6 = reg.params.sort_index()


# Question 7

# Next, let's train a Lasso model using statsmodels *without* formulas.
# Fit a Lasso Regression model with `price` as the response and `transmission`, `mileage`, `make`,
# `mpg`, and `model` as the predictors.
# Using the training set, you will need to create dummy variables for the categorical columns
# (including dropping the first category).
# Do not change the names of the columns.
# You will also need to add a constant.
# Use `alpha=15` for fitting the Lasso regression.
# Submit a Series with the parameter names as the index and the parameter estimates as the values
# sorted by index

# *Note: Since `model` contains many categories, this will results in a large number of dummy
# columns.
# Lasso regresssion can be useful way to select a fewer number of columns that have a large effect
# on the response (variable selection) and to deal with multicollinearity.*

cat_cols = ["transmission", "make", "model"]
train_df6 = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)

train_df6 = sm.add_constant(train_df6)

X = train_df6.drop(["price", "engineSize","fuelType","tax","year"], axis=1)
y = train_df["price"]


lasso_model = sm.OLS(y, X).fit_regularized(alpha=15, L1_wt=1.0)
params = pd.Series(lasso_model.params, index=lasso_model.params.index).sort_index()


q7 = params


# Question 8

# Use scikit-learn to fit a Linear Regression model that predicts the car price based on the
# numerical variables `engineSize`, `mileage`, `mpg`, `tax`, and `year` (make the columns in that
# order).
# Again use only the train set for fitting the model.

# Submit a Series where the index is the parameter name (column name) and the values are the
# parameter coefficients (do not include the intercept) sorted by index.

from sklearn.linear_model import LinearRegression

X8 = train_df[["engineSize", "mileage", "mpg", "tax", "year"]]
y8 = train_df["price"]

linreg_model = LinearRegression().fit(X8, y8)

params8 = pd.Series(linreg_model.coef_, index=X8.columns).sort_index()
q8 = params8


# Question 9

# Train a scikit-learn Lasso Regression model with `alpha=1` and `max_iter=2000` on the training
# data using `transmission`, `fuelType`, `tax`, and `mpg` as predictors and `price` as the response.
# Use `OneHotEncoding` on the categorical variables (no dropping) and standardize the numerical
# variables using `StandardScalar`.
# Also, if using the `ColumnTransformer` (recommended), set `verbose_feature_names_out=False` to
# make the column names more readable.
# Make sure to wrap the data transformations and Lasso Regression estimator into a pipeline so we
# can easily make predictions on the test data.

# Calculate the residuals on the **test data set**.
# Submit a Series where the index matches the test set's index and the values are the corresponding
# residuals.

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso

transformer = ColumnTransformer([
    ("onehot", OneHotEncoder(sparse=False, handle_unknown="ignore"), ["transmission", "fuelType"]),
    ("scaler", StandardScaler(), ["tax", "mpg"]),
])


estimator = Lasso(alpha=1, max_iter=2000)

pipe = Pipeline([
    ("transformer", transformer),
    ("estimator", estimator),
])

pipe.fit(train_df[["transmission", "fuelType", "tax", "mpg"]], train_df["price"])
residuals = test_df["price"] - pipe.predict(test_df[["transmission", "fuelType", "tax", "mpg"]])
residuals_series = pd.Series(residuals, index=test_df.index)
q9 = residuals_series


# Question 10

# When we fit a Linear Regression model on some data, we get back point estimates of the parameter
# values - one value per parameter.
# But sometimes we want to know more than just the point estimates, we may want to know the
# distribution of the parameters or even the distribution of a function of the parameters.
# If the parameter of interest is *nice* we might be able to pull out some theory to find
# approximate distributions, but this may not be feasible for some parameters of interest.

# Another option is to use some resampling techniques to get an empirical distribution of the
# parameters of interest (like histograms).
# This general idea is called *bootstrapping*.
# Given some large number $N$, a specific type of *bootstrapping* is to sample the entire dataset
# *with replacement* $N$ times to create $N$ different datasets data are the same size as the
# original dataset.
# For each of those datasets, you can fit your model and get $N$ different versions of the parameter
# estimates.
# With a large enough $N$, you can find an approximate distribution of the parameters or functions
# of the parameters.

# To see this in action, first filter the training dataset to be cars whose `make` is toyota.
# We will fit a Linear Regression model on the toyota data to predict `price` using the
# `engineSize`, `mileage`, `mpg`, and `tax` columns.
# The parameter of interest is the ratio of the parameters for `engineSize` and `mpg` which would
# look like `coef_engineSize` / `coef_mpg`.
# Use 1000 bootstrap samples to estimate the **median** of the distribution of `coef_engineSize` /
# `coef_mpg`.

# Submit the estimate of the median of the parameter of interest.

# *Hint: you could keep track of all 1000 versions of the parameter estimates. For a given dataset,
# the estimate of the ratio can be calculated by taking the ratio of the two coefficients*. Then
# you'll have 1000 samples of the parameter of interest.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample


toyota_df = train_df[train_df['make'] == 'toyota']

X10 = toyota_df[['engineSize', 'mileage', 'mpg', 'tax']]
y10 = toyota_df['price']


model = LinearRegression().fit(X10, y10)

ratio = model.coef_[0] / model.coef_[2]

bootstrapped_ratios = []
for i in range(1000):
    X_resampled, y_resampled = resample(X10, y10)
    model_resampled = LinearRegression().fit(X_resampled, y_resampled)
    ratio_resampled = model_resampled.coef_[0] / model_resampled.coef_[2]
    bootstrapped_ratios.append(ratio_resampled)

median_ratio = np.median(bootstrapped_ratios)

q10 = median_ratio
