from typing import Any, Callable, Union
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations


from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer


class IterativeGroupClassifier(BaseEstimator, ClassifierMixin):
    """Iterative Group Classifier

    Takes in a train and test dataset and imputes both based on the specified
    rules. For the Kaggle challenge, primarily operates by having separate
    imputation models for different values of product_code. Also allows for a
    backup model to be used when the most important feature is missing.
    """

    def __init__(
        self,
        iterative_regressor,
        classifier,
        backup_est=None,
        use_backup="partial",
    ):
        self.iterative_regressor = iterative_regressor
        self.classifier = classifier
        if backup_est is None:
            self.backup_est = clone(classifier)
        else:
            self.backup_est = backup_est
        self.use_backup = use_backup

    def _iterative_impute(
        self,
        X: pd.DataFrame,
        group_col: str = None,
        fit: bool = True,
        max_iter: int = 5,
        impute_cols: list[str] = None,
    ) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        X : pd.DataFrame
            Data to impute
        group_col : str, optional
            If specified, separate imputing models fit to each group, by
            default None
        fit : bool, optional
            _description_, by default True
        max_iter : int, optional
            _description_, by default 5
        impute_cols : list[str], optional
            If specified, will only impute these columns, by default None

        Returns
        -------
        pd.DataFrame
            _description_
        """
        # Specify impute cols
        self.impute_cols = impute_cols
        if self.impute_cols is None:
            self.impute_cols = list(X.columns)

        # Perform initial imputations using mean
        complete = X.copy()
        masks = np.zeros(X.shape, dtype=bool)
        for i in range(X.shape[1]):
            masks[:, i] = X.iloc[:, i].notna()
            if ((~masks[:, i]).sum() == 0) | (
                X.columns[i] not in self.impute_cols
            ):
                continue
            complete.iloc[~masks[:, i], i] = X.iloc[masks[:, i], i].mean()
            # complete.iloc[~masks[:, i], i] = X.iloc[masks[:, i], i].median()

        # Iteratively train models
        if fit:
            # Very cool: <https://stackoverflow.com/a/27809959>
            self.models: dict = defaultdict(lambda: defaultdict(dict))
        for i in range(max_iter):
            for j in range(X.shape[1]):
                col = X.columns[j]
                if ((~masks[:, j]).sum() == 0) | (col not in self.impute_cols):
                    continue

                if group_col is None:

                    # Masks for fitting and imputing
                    fit_mask = masks[:, j]
                    pred_mask = ~masks[:, j]
                    if (fit_mask.sum() == 0) | (pred_mask.sum() == 0):
                        # Skip if nothing to fit or predict
                        continue
                    # Fit model
                    curr_pred = [
                        feat for feat in complete.columns if feat not in [col]
                    ]
                    if fit:
                        est = clone(self.iterative_regressor)
                        est.fit(
                            complete.loc[fit_mask, curr_pred],
                            complete.loc[fit_mask, col],
                        )
                        # Save model
                        self.models[i][col] = est

                    # Re-impute missing
                    complete.loc[pred_mask, col] = self.models[i][col].predict(
                        complete.loc[pred_mask, curr_pred]
                    )
                else:
                    for group in complete[group_col].unique():
                        # Masks for fitting and imputing
                        fit_mask = masks[:, j] & (complete[group_col] == group)
                        pred_mask = (~masks[:, j]) & (
                            complete[group_col] == group
                        )

                        if (fit_mask.sum() == 0) | (pred_mask.sum() == 0):
                            # Skip if nothing to fit or predict
                            continue

                        # Fit model
                        curr_pred = [
                            feat
                            for feat in complete.columns
                            if feat not in [col, group_col]
                        ]
                        if fit:
                            est = clone(self.iterative_regressor)
                            est.fit(
                                complete.loc[fit_mask, curr_pred],
                                complete.loc[fit_mask, col],
                            )
                            # Save model
                            self.models[i][col][group] = est

                        # Re-impute missing
                        complete.loc[pred_mask, col] = self.models[i][col][
                            group
                        ].predict(complete.loc[pred_mask, curr_pred])

        return complete

    def _quick_predict(self, est, data):
        if hasattr(est, "predict_proba"):
            return est.predict_proba(data)[:, 1]
        else:
            return est.predict(data)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        return_predictions: bool = True,
        group_col: str = "product_code",
        use_groups: bool = True,
        fs_func: Callable = f_classif,
        **impute_params
    ) -> Union[np.ndarray, Any]:
        """Fit Classifier

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data
        y_train : pd.Series
            Training labels
        X_test : pd.DataFrame
            Test data
        return_predictions : bool, optional
            Whether to return predictions, by default True
        group_col : str, optional
            Feature to group by while imputing, by default "product_code"
        use_groups : bool, optional
            Whether to use groups while imputing, by default True
        fs_func : function, optional
            Function used for feature selection, by default f_classif

        Returns
        -------
        Union[np.ndarray, Any]
            If return_predictions, then predictions. Otherwise, the classifier.
        """

        # Combine X datasets
        train_missing = X_train.isna()
        test_missing = X_test.isna()
        n_train = X_train.shape[0]  # For splitting out later
        X = pd.concat([X_train, X_test])

        # Impute
        iterative_group_col = None
        if use_groups:
            iterative_group_col = group_col
        complete = self._iterative_impute(
            X, group_col=iterative_group_col, **impute_params
        )

        # Re-split
        X_train = complete.iloc[:n_train, :]
        X_test = complete.iloc[n_train:, :]

        # Fit model
        if self.use_backup == "None":
            self.classifier.fit(X_train.drop(columns=group_col), y_train)
        else:
            # Get best col
            f_stat, _ = fs_func(X_train.drop(columns=group_col), y_train)
            self.backup_col = X_train.columns[f_stat.argmax()]

            mask = train_missing[self.backup_col]

            # Fit primary
            self.classifier.fit(
                X_train[~mask].drop(columns=group_col), y_train[~mask]
            )

            # Fit backup
            self.backup_est.fit(
                X_train[mask].drop(columns=[group_col, self.backup_col]),
                y_train[mask],
            )

        # Get predictions if specified
        if return_predictions:
            return self._predict(
                X_test, mask=test_missing[self.backup_col], group_col=group_col
            )
        else:
            return self.classifier

    def _predict(
        self,
        X_test: pd.DataFrame,
        mask: pd.Series,
        group_col: str = "product_code",
    ) -> np.ndarray:
        """Make predictions on already imputed test set

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data, already imputed
        mask : pd.Series
            Whether most important feature is missing
        group_col : str, optional
            Feature to group on, by default "product_code"

        Returns
        -------
        np.ndarray
            Predictions
        """
        # Predict
        if self.backup_est is None:
            return self._quick_predict(
                self.classifier, X_test.drop(columns=group_col)
            )
        else:
            y_pred = np.zeros(X_test.shape[0])

            # Make primary predictions
            y_pred[~mask] = self._quick_predict(
                self.classifier, X_test[~mask].drop(columns=group_col)
            )

            # Make backup predictions
            y_pred[mask] = self._quick_predict(
                self.backup_est,
                X_test[mask].drop(columns=[group_col, self.backup_col]),
            )
            if self.use_backup == "partial":
                # Add in primary predictions
                y_pred[mask] += self._quick_predict(
                    self.classifier, X_test[mask].drop(columns=group_col)
                )

                # Divide by 2
                y_pred[mask] /= 2
            return y_pred

    def predict(self, X_test, group_col="product_code") -> np.ndarray:
        """Make predictions test set

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data
        group_col : str, optional
            Feature to group on, by default "product_code"

        Returns
        -------
        np.ndarray
            Predictions
        """
        # Impute
        missing_mask = X_test[self.backup_col].isna()
        X_test = self._iterative_impute(X_test, group_col=group_col, fit=False)

        return self._predict(X_test, mask=missing_mask, group_col=group_col)


# Define feature selection function
def fs_func(X, y):
    X_ = SimpleImputer().fit_transform(X)
    return f_classif(X_, y)


def cv_eval(est, X, y, group_col, metrics, test_groups=1, **fit_params):
    """Custom Cross validation

    Parameters
    ----------
    est : _type_
        _description_
    X : _type_
        _description_
    y : _type_
        _description_
    group_col : _type_
        _description_
    metrics : _type_
        _description_
    test_groups : int, optional
        How many product_codes to have in the test fold at a time, by default 1
    use_groups : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """

    # Copy and scale
    X_ = X.copy()

    # Run tests
    row_list = []
    estimators = []
    all_group_sets = combinations(X_[group_col].unique(), test_groups)
    for group_set in all_group_sets:
        row = {"group": group_set}

        # Get train and test groups
        test_mask = X_[group_col].isin(group_set)
        X_train, X_test = X_[~test_mask].copy(), X_[test_mask].copy()
        y_train, y_test = y[~test_mask].copy(), y[test_mask].copy()

        # Train and make predictions
        curr_est = clone(est)
        y_pred = curr_est.fit(
            X_train,
            y_train,
            X_test,
            return_predictions=True,
            group_col=group_col,
            **fit_params
        )  # [:, 1]

        # Evaluate
        for met in metrics:
            val = met(y_test, y_pred)
            row[met.__name__] = val
        row_list.append(row)
        estimators.append(curr_est)

    return estimators, pd.DataFrame(row_list)
