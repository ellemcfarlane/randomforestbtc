from flask import Flask
from flask import request
from random_forest_regressor import RandomForestRegressor
from random_forest_regressor import RandomTree
import pickle
from getdata import scraper
import json
import pandas as pd

app = Flask(__name__)

#pre-caching the html files so we don't waste memory opening and closing files constantly:
handle = open("template/index.html", "r")
index_html = handle.read()
handle.close()
handle = open("template/style.css", "r")
stylesheet_css = handle.read()
handle.close()
handle = open("template/manual.html", "r")
manual_html = handle.read()
handle.close()

getdata = scraper()
today = getdata.gettoday()

#loading the pickle data
with open('../final_forest_drop0.pkl', 'rb') as file:
    regressor = pickle.load(file)


@app.route("/")
def index():
    return index_html

@app.route("/manual", methods=['GET'])
def manual():
    if request.args.get('market_price'): #is there an incoming get request
        frame = {
            'market_price':float(request.args['market_price']), 'avg_block_size':float(request.args['avg_block_size']), 'blocks_size':float(request.args['blocks_size']), 'cost_per_txn':float(request.args['cost_per_txn']), 'difficulty':float(request.args['difficulty']), 'txn_vol':float(request.args['txn_vol']), 'hash_rate':float(request.args['hash_rate']),
            'market_cap':float(request.args['market_cap']), 'confirm_time':float(request.args['confirm_time']), 'miners_revenue':float(request.args['miners_revenue']), 'n_transaction':float(request.args['n_transaction']), 'n_transaction_exclude_popular':float(request.args['n_transaction_exclude_popular']),
            'txn_per_block':float(request.args['txn_per_block']), 'output_vol':float(request.args['output_vol']), 'total_bitcoins':float(request.args['total_bitcoins']), 'trade_volume':float(request.args['trade_volume']), 'txn_fees':float(request.args['txn_fees'])
        }
        return manual_html.replace("<!--replacethis-->", "<div class='prediction current'><h2 id='price'>" + str(regressor.predict([frame])[0]) + "</h2> <span>Price Prediction</span></div>")
    else:
        return manual_html

@app.route("/style.css")
def stylesheet():
    return stylesheet_css

@app.route("/predict")
def predict():
    return str(regressor.predict(today)[0])

@app.route("/currentstats")
def currentstats():
    return json.dumps(today)

@app.route("/currentprice")
def currentprice():
    return str(today[0]['market_price'])

@app.route("/manualprediction", methods=['GET'])
def manualprediction():
    frame = {
            'market_price':float(request.args['market_price']), 'avg_block_size':float(request.args['avg_block_size']), 'blocks_size':float(request.args['blocks_size']), 'cost_per_txn':float(request.args['cost_per_txn']), 'difficulty':float(request.args['difficulty']), 'txn_vol':float(request.args['txn_vol']), 'hash_rate':float(request.args['hash_rate']),
            'market_cap':float(request.args['market_cap']), 'confirm_time':float(request.args['confirm_time']), 'miners_revenue':float(request.args['miners_revenue']), 'n_transaction':float(request.args['n_transaction']), 'n_transaction_exclude_popular':float(request.args['n_transaction_exclude_popular']),
            'txn_per_block':float(request.args['txn_per_block']), 'output_vol':float(request.args['output_vol']), 'total_bitcoins':float(request.args['total_bitcoins']), 'trade_volume':float(request.args['trade_volume']), 'txn_fees':float(request.args['txn_fees'])
    }
    return str(regressor.predict([frame])[0])

@app.route("/graphdata")
def graphdata():
    dataset = pd.read_csv('../raw_csvs/bitcoin_truncated.csv', low_memory=True)
    dataset['market_price'] = dataset['market_price'].shift(-1)
    dataset.drop(len(dataset) - 1, axis=0, inplace=True)
    y_orig = dataset.loc[:, 'market_price'].values
    X_orig = dataset.drop(columns=['market_price', 'date', 'Unnamed: 0', 'Unnamed: 0.1']).to_dict('records')
    dataset = dataset.sample(frac=1, random_state=0)
    y = dataset.loc[:, 'market_price'].values
    dataset.drop(columns=['market_price', 'date', 'Unnamed: 0', 'Unnamed: 0.1'], inplace=True)
    X = dataset.to_dict('records')
    train_sz = int(.8*len(X))
    X_train = X[:train_sz]
    y_train = y[:train_sz]
    X_test = X[train_sz:]
    y_test = y[train_sz:]
    y_pred = regressor.predict(X_test)
    y_train_pred = regressor.predict(X_train)
    y_full = regressor.predict(X_orig)
    x_test = range(len(y_pred))
    x_train = range(len(y_train_pred))
    x_full = range(len(y_pred) + len(y_train_pred))
    #time to stitch the lists together in a useful manner
    return json.dumps([feedtoformat(x_train, y_train), feedtoformat(x_train, y_train_pred), feedtoformat(x_test, y_test), feedtoformat(x_test, y_pred), feedtoformat(x_full, y_orig), feedtoformat(x_full, y_full)])

#feed data into format for chartjs
def feedtoformat(set1, set2):
    toreturn = []
    for i in range(len(set1)):
        toreturn.append({"x":set1[i], "y":set2[i]})
    return toreturn

app.run()
