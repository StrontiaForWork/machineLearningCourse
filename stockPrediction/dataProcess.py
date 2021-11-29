import pandas as pd
import numpy as np
import sqlalchemy
import tushare as ts

engine = sqlalchemy.create_engine('mysql+pymysql://root:root@localhost:3306/stock')

# sql='''
# SELECT stock_code ,deal_date, pclose FROM tb_fkline
# WHERE stock_code IN ('sh600829')
# ORDER BY deal_index
# '''

# # 创建股票表
# sql = '''
# CREATE TABLE `daily_price` (
#   `id` int NOT NULL primary key AUTO_INCREMENT,
#   `ts_code` char(12) NOT NULL ,
#   `trade_date` char(12) NOT NULL,
#   `open_price` float(7,2) NULL,
#   `high_price` float(7,2) NULL,
#   `low_price` float(7,2) NULL,
#   `close_price` float(7,2) NULL,
#   `adj_close_price` float(7,4) NULL,
#   `change` float(5,2) NULL,
#   `pct_chg` float(5,2) NOT NULL,
#   `vol` float(10,2) NOT NULL,
#   `amount` float(10,2) NOT NULL
# ) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;
# '''
#
# tm = pd.read_sql(sql,engine)

# 获取股票数据
pro = ts.pro_api('1288ed74a1b0b450b1b67f57773fce56905763e070754173f8bf1974')
# 查询当前所有正常上市交易的股票列表
df = pro.daily(ts_code='002230.SZ') #, start_date='19990829', end_date='20211128'
print(df)
# 将股票数据存入MySQL
df.to_sql(name='daily_price',con=engine,if_exists='replace',index=False)
