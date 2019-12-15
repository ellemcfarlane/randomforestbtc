import urllib.request
import json

class scraper:    
    def s_market_price():
        return json.loads(urllib.request.urlopen("https://blockchain.info/ticker").read())
    
    def s_avg_block_size():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/avg-block-size?format=json").read())
    
    def s_cost_per_txn():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/cost-per-transaction?format=json").read())

    def s_difficulty():
        return urllib.request.urlopen("https://blockchain.info/q/getdifficulty").read()

    def s_txn_vol():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/n-transactions?format=json").read())

    def s_hash_rate():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/hash-rate?format=json").read())

    def s_market_cap():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/market-cap?format=json").read())

    def s_confirm_time():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/median-confirmation-time?format=json").read())

    def s_miners_revenue():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/miners-revenue?format=json").read())

    def s_n_transactions():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/n-transactions-total?format=json").read())

    def s_n_transaction_exclude_popular():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/n-transactions-excluding-popular?format=json").read())

    def s_txn_per_block():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/n-transactions-per-block?format=json").read())

    def s_output_vol():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/output-volume?format=json").read())

    def s_total_bitcoins():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/total-bitcoins?format=json").read())

    def s_trade_volume():
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/trade-volume?format=json").read())

    def s_txn_fees():
        return json.loads(urllib.request.urlopen("https://bitcoinfees.earn.com/api/v1/fees/recommended").read())


