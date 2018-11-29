import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=12345)

# Average CV score on the training set was:-6648.872981389077
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=17),
    SelectPercentile(score_func=f_regression, percentile=2),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.001, max_depth=9, min_child_weight=17, n_estimators=100, nthread=1, subsample=0.5)),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ElasticNetCV(l1_ratio=1.0, tol=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
