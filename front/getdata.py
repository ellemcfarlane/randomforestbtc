from blockchain import statistics
import pandas as pd
import urllib.request
import json
import numpy as np

class scraper:
    def s_avg_block_size(self):
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/avg-block-size?format=json").read())

    def s_cost_per_txn(self):
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/cost-per-transaction?format=json").read())

    def s_txn_vol(self):
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/n-transactions?format=json").read())

    def s_market_cap(self):
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/market-cap?format=json").read())

    def s_confirm_time(self):
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/median-confirmation-time?format=json").read())

    def s_n_transaction_exclude_popular(self):
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/n-transactions-excluding-popular?format=json").read())

    def s_txn_per_block(self):
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/n-transactions-per-block?format=json").read())

    def s_output_vol(self):
        return json.loads(urllib.request.urlopen("https://api.blockchain.info/charts/output-volume?format=json").read())



    def gettoday(self):
        df_names = {
         0:'market_price', 1:'avg_block_size', 2:'blocks_size', 3:'cost_per_txn', 4:'difficulty', 5:'txn_vol', 6:'hash_rate',
         7:'market_cap', 8:'confirm_time', 9:'miners_revenue', 10:'n_transaction', 11:'n_transaction_exclude_popular',
         12:'txn_per_block', 13:'output_vol', 14:'total_bitcoins', 15:'trade_volume', 16:'txn_fees'
        }
        stats = statistics.get()
        dataframes = [
         stats.market_price_usd, self.s_avg_block_size()['values'][-1]['y'], stats.blocks_size, self.s_cost_per_txn()['values'][-1]['y'], stats.difficulty, self.s_txn_vol()['values'][-1]['y'], stats.hash_rate,
         self.s_market_cap()['values'][-1]['y'], self.s_confirm_time()['values'][-1]['y'], stats.miners_revenue_usd, stats.number_of_transactions, self.s_n_transaction_exclude_popular()['values'][-1]['y'],
         self.s_txn_per_block()['values'][-1]['y'], self.s_output_vol()['values'][-1]['y'], stats.total_btc, stats.trade_volume_usd, stats.total_fees_btc
        ]
        toreturn = pd.DataFrame(dataframes)
        toreturn = toreturn.transpose()
        toreturn = toreturn.rename(columns = df_names)
        print(len(toreturn))
        return toreturn

