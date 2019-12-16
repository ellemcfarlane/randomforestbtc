from flask import Flask
import pickle
from random_forest_regressor import RandomForestRegressor
from random_forest_regressor import RandomTree
from getdata import scraper

app = Flask(__name__)

#pre-caching the html files so we don't waste memory opening and closing files constantly:
handle = open("template/index.html", "r")
index_html = handle.read()
handle.close()
handle = open("template/style.css", "r")
stylesheet_css = handle.read();
handle.close()

getdata = scraper();
today = getdata.gettoday()

#loading the pickle data
with open('../final_forest.pkl', 'rb') as file:
    regressor = pickle.load(file)


@app.route("/")
def index():
    return index_html

@app.route("/style.css")
def stylesheet():
    return stylesheet_css

@app.route("/predict")
def predict():
    return str(regressor.predict(today)[0])


app.run()
