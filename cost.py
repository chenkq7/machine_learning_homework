import pandas as pd
import numpy as np

raw_train_data = pd.read_csv("dataset/cost/train.csv")

nominal_columns = ['sex', 'smoker', 'region']
ratio_columns = ['children', 'age', 'bmi']
feature_columns = nominal_columns + ratio_columns
label_columns = ['charges']

if __name__ == '__main__':
    # dataset independent preprocess
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    oe = OrdinalEncoder()
    nominal_encoded = oe.fit_transform(raw_train_data[nominal_columns])
    X = np.column_stack((raw_train_data[ratio_columns], nominal_encoded))
    y = raw_train_data[label_columns]

    # train val split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # dataset dependent preprocess
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # model select
    from sklearn import linear_model
    from sklearn.ensemble import RandomForestRegressor
    # svr = make_pipeline(StandardScaler(), linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7))
    svr = make_pipeline(StandardScaler(), RandomForestRegressor())

    # fit
    svr.fit(X_train, y_train)

    # metrics
    from sklearn.metrics import r2_score
    score = svr.score(X_test, y_test)
    pred = svr.predict(X_test)
    print(pred[:10])
    print(y_test[:10])
    print(score)
    print(r2_score(y_test, pred))
