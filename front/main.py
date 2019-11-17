from flask import Flask

app = Flask(__name__)

#pre-caching the html files so we don't waste memory opening and closing files constantly:
handle = open("template/index.html", "r")
index_html = handle.read()
handle.close()
handle = open("template/style.css", "r")
stylesheet_css = handle.read();
handle.close()

@app.route("/")
def index():
    return index_html

@app.route("/style.css")
def stylesheet():
    return stylesheet_css

app.run()
