import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Logistic regression is a classifier which tries to predict 'Price Binary'
# Price binary is a boolean; 0 = Price went down that day, 1 = Price went up that day

def display(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            print(df)


if __name__ == '__main__':
    df = pd.read_csv('df_final.csv', parse_dates=['Date'], low_memory=False)
    df.drop(['Date'], 1, inplace=True)

    df_label = df['Price Binary']
    df.drop(['Price Binary'], 1, inplace=True)

    names = df.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    scaled_df.fillna(-99999, inplace=True)

    X = np.array(scaled_df)
    y = np.array(df_label)
    X = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=200).fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    print(clf.predict(X[:20, :]))
    print(accuracy)
