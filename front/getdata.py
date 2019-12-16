from blockchain import statistics
import pandas as pd
import urllib.request
import json
import numpy as np

class scraper:

    daily = []

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



    def __init__(self):
        stats = statistics.get()
        self.daily = [
         stats.market_price_usd, self.s_avg_block_size()['values'][-1]['y'], stats.blocks_size, self.s_cost_per_txn()['values'][-1]['y'], stats.difficulty, self.s_txn_vol()['values'][-1]['y'], stats.hash_rate,
         self.s_market_cap()['values'][-1]['y'], self.s_confirm_time()['values'][-1]['y'], stats.miners_revenue_usd, stats.number_of_transactions, self.s_n_transaction_exclude_popular()['values'][-1]['y'],
         self.s_txn_per_block()['values'][-1]['y'], self.s_output_vol()['values'][-1]['y'], stats.total_btc, stats.trade_volume_usd, stats.total_fees_btc
        ]

    def gettoday(self):
        df_names = {
         'market_price':self.daily[0], 'avg_block_size':self.daily[1], 'blocks_size':self.daily[2], 'cost_per_txn':self.daily[3], 'difficulty':self.daily[4], 'txn_vol':self.daily[5], 'hash_rate':self.daily[6],
         'market_cap':self.daily[7], 'confirm_time':self.daily[8], 'miners_revenue':self.daily[9], 'n_transaction':self.daily[10], 'n_transaction_exclude_popular':self.daily[11],
         'txn_per_block':self.daily[12], 'output_vol':self.daily[13], 'total_bitcoins':self.daily[14], 'trade_volume':self.daily[15], 'txn_fees':self.daily[16]
        }
        #toreturn = pd.DataFrame(self.daily)
        #toreturn = toreturn.transpose()
        #toreturn = toreturn.rename(columns = df_names)
        return df_names

