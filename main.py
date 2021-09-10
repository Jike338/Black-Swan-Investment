import backtrader as bt
import datetime
from strategy import TestStrategy
cerebro = bt.Cerebro()
cerebro.broker.set_cash(1000000)


data = bt.feeds.YahooFinanceCSVData(
    dataname='data.csv',
    # Do not pass values before this date
    fromdate=datetime.datetime(2000, 1, 1),
    # Do not pass values after this date
    todate=datetime.datetime(2000, 12, 31),
    reverse=False)

cerebro.adddata(data)
cerebro.addstrategy(TestStrategy)

print('Starting portfolio value" %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Final portfolio value: %.2f' % cerebro.broker.getvalue())