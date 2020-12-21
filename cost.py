import pandas as pd
import numpy as np


def get_k_th_score_params(grid_, k: int):
    idx_ = list(grid_.cv_results_['rank_test_score']).index(k)
    param_ = grid_.cv_results_['params'][idx_]
    score_ = grid_.cv_results_['mean_test_score'][idx_]
    return score_, param_


def plot_true_pred_figure(y_true, y_pred, title=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    from sklearn.metrics import r2_score
    r2_ = r2_score(y_true, y_pred)
    from matplotlib import pyplot as plt
    plt.scatter(y_true, y_pred)
    max_v = np.max([y_pred, y_true])
    plt.plot([0, max_v], [0, max_v], color='red')
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    title = str(r2_) + " " + str(title) if title else str(r2_)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    """
    模型初选. 观察score前n名的params, 可以发现这些模型均使用了random_forest模型.
    :return 网格搜索结果 grid
    """
    # read data
    raw_data = pd.read_csv("dataset/cost/train.csv")
    labels = ['charges']
    X, y = raw_data.drop(labels, axis=1), raw_data[labels]

    # train val split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # processor
    from sklearn.compose import make_column_transformer, make_column_selector
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
    from sklearn.preprocessing import QuantileTransformer

    processor_lin = make_column_transformer(
        (OneHotEncoder(), make_column_selector(dtype_include=object)),
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        remainder='passthrough')
    processor_nlin = make_column_transformer(
        (OrdinalEncoder(), make_column_selector(dtype_include=object)),
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        remainder='passthrough')

    # target_transformers
    from sklearn.preprocessing import FunctionTransformer

    target_transformers = [
        None,  # None means identical transform
        QuantileTransformer(n_quantiles=300, output_distribution="normal"),
        FunctionTransformer(func=np.log, inverse_func=np.exp),
        FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
    ]

    # model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR

    models = [
        RandomForestRegressor(),
        LinearRegression(),
        Ridge(),
        SVR(),
    ]

    # pipe together
    from sklearn.pipeline import Pipeline
    from sklearn.compose import TransformedTargetRegressor

    tt = TransformedTargetRegressor(regressor=SVR(), transformer=None)
    pipe = Pipeline([
        ('processor', processor_lin),
        ('ttr', tt),
    ])

    # grid search CV fit
    from sklearn.model_selection import GridSearchCV

    param_grid = dict(
        processor=[processor_lin, processor_nlin],
        ttr__transformer=target_transformers,
        ttr__regressor=models,
    )

    grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
    grid.fit(X_train, y_train)
    score = grid.score(X_test, y_test)
    print(score)
    print(grid.best_params_)

    plot_true_pred_figure(y_test, grid.predict(X_test), "best params on test set")
    plot_true_pred_figure(y_train, grid.predict(X_train), "best params on train set")
