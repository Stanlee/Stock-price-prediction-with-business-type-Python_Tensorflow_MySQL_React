#!/usr/bin/python
#-*-coding:utf-8-*-
# MySQL에 연결하는 connection 객체를 생성하며 table을 read, update, delete함.
# Date : 2018년 6월 2일
# Author : 이인현

import pandas as pd
from pandas import Series, DataFrame
import pymysql
from sqlalchemy import *
import pandas as pd
import datetime

pymysql.install_as_MySQLdb()
# import MySQLdb
from DataFrame_Control1 import *
from Cyboss_Control1 import *
from sqlalchemy.sql import select

class MySQL_Sim_Control:
    def __init___(self):
        self.conn = 0
        self.engine = 0  # pandas sql 사용하기위해
        self.conn_pandas = 0  # pandas sql 사용하기위한 connection 객체
        self.user = 0
        self.password = 0
        self.db = 0
        self.table_name = 0
        self.stockcode = 0
        self.DataFrame = 0
        self.SQL_SAFE_UPDATES = 0;  ### 설정 중요

    # DB 연결
    def set_connection_obj_db(self, host=0, user=0, password=0, db=0, charset=0):
        self.conn = pymysql.connect(host=host, user=user, password=password, db=db, charset=charset)
        print("Conn successfully!!")

    def set_off_connection(self):
        self.conn.close()

    def set_connection_obj_pandas(self, host=0, user=0, password=0, db=0, charset=0):
        self.engine = create_engine(
            "mysql+mysqldb://{}:{}@{}/".format(user, password, host) + db + '?charset=' + charset)
        self.conn_pandas = self.engine.connect()
        print("Conn successfully!!")

    def set_off_connection_pandas(self):
        self.conn_pandas.close()

    # 종목코드 세팅
    def set_stockcode(self, stockcode):
        self.stockcode = stockcode

    # 종목에 해당하는 데이터프레임 세팅
    def set_dataframe(self, dataframe):
        self.dataframe = dataframe

    # table 이름 설정
    def set_table_name(self, table_name):
        self.table_name = table_name

    # table 이름 받기
    def get_table_name(self):
        return self.table_name

    # 데이터베이스 내 테이블 전체 읽기
    def read_table(self, table_name):
        curs = self.conn.cursor()
        sql = "select * from {}".format(table_name)
        curs.execute(sql)
        rows = curs.fetchall()
        print(rows)
        pass

    # 데이터베이스 내 테이블 데이터 프레임으로 전체 읽기
    def read_table_by_df(self, table_name):
        dataframe = pd.read_sql_table(table_name, self.conn_pandas)
        return dataframe

    # 데이터베이스 내 테이블 저장 (기존 데이터의 row 아래로 concat)
    def save_db(self, dataframe, db, table):
        try:
            # Save dataframe to database
            dataframe.to_sql(name=table, con=self.engine, if_exists='append')
            print("Saved successfully!!")
            self.conn_pandas.commit()
        except:
            pass
        #    traceback.print_exc()
        # finally:
        #     self.conn_pandas.close()

    # 종목코드와 종목명 전체를 읽어와서 MYSQL에 저장함.
    def update_in_MySQL_code_management(self):
        # 4. 종목코드명 전체와 종목명 전체를 읽어와서 MySQL로 저장함.
        stock_code_obj = Stock_Code()
        stock_code = Stock_Code()
        codeList1, codeLength1 = stock_code.getCodeList(1)  # 1코스피
        codeList2, codeLength2 = stock_code.getCodeList(2)  # 2코스닥

        codeList1 = list(codeList1)
        codeList2 = list(codeList2)

        # print(type(codeList1))
        # print(type(codeList2))

        # codeList = []
        codeList1.extend(codeList2)
        #print('codeList:{}'.format(codeList1))
        #print('codeListLen:{}'.format(len(codeList1)))
        # codeList.extend(codeList2) # 코스피 코스닥 코드명 합치기
        # print('codeList:{}'.format(codeList))
        # print(len(codeList))

        stockname_total = []
        code_total = []
        stockname = 0

        # 종목코드명을 종목명으로 변경하여 출력
        for code in codeList1:
            stockname = stock_code_obj.code_to_name(code)
            code_total.append(code)
            stockname_total.append(stockname)

        dataframe_daeshin = pd.DataFrame({'code': code_total, 'stockname': stockname_total})
        # print("dataframe_daeshin:{}".format(dataframe_daeshin))
        # dataframe_daeshin_set = set(dataframe_daeshin['code'])

        # mysql obj를 생성하여 코드를 종목명으로 변경하는 함수 생성 (데이터프레임 방식으로 한번에 읽기)
        mysql_control_obj = MySQL_Sim_Control()
        mysql_control_obj.set_connection_obj_pandas(host='yourhost', user='user', password='password', db='tensor', charset='utf8')
        dataframe_db = mysql_control_obj.read_table_by_df("code_management")

        total_dataframe = pd.concat([dataframe_daeshin, dataframe_db], ignore_index=True)
        total_dataframe = total_dataframe.drop_duplicates(['code', 'stockname'], keep='last')
        total_dataframe = total_dataframe.drop_duplicates(['code'], keep='first')
        total_dataframe = total_dataframe.sort_values(by=['code'], ascending=True)
        #print("total_dataframe:{}".format(total_dataframe))

        total_dataframe.to_sql('code_management', con=mysql_control_obj.conn_pandas, if_exists='replace', index=False)

        mysql_control_obj.set_off_connection_pandas()

        # 종목코드와 종목명 전체를 읽어와서 MYSQL에 저장함.

    def update_in_MySQL_stock_price(self, type=0, stockname='all', from_date=0, to_date=0):

        ######################################### 1. 대신증권의 시세를 읽어옴 ##########################################
        stock_code = Stock_Code()
        mysql_control_obj = MySQL_Sim_Control()
        mysql_control_obj.set_connection_obj_pandas(host='yourhost', user='user', password='password', db='tensor', charset='utf8')
        if (type == 1):
            codeList, codeLength = stock_code.getCodeList(1)  # 1코스피
        elif (type == 2):
            codeList, codeLength = stock_code.getCodeList(2)  # 2코스닥
        else:
            df_from_MySQL_manage_stock =  pd.read_sql_query("select code from manage_stock", mysql_control_obj.conn_pandas)
            codeList = df_from_MySQL_manage_stock['code'].tolist()
            print(codeList)

        for code in codeList:
            stock_chart = CpStockChart()
            stock_chart.RequestFromTo(code=code, fromDate=from_date, toDate=to_date, caller=stock_chart)
            name, data_df = stock_chart.get_name_and_df()
            print(name)
            print(data_df)

            # 데이터프레임 콘트롤1으로 가공
            dataframe_control1_obj = DataFrame_Control1()
            dataframe_control1_obj.set_stock_item(name)
            dataframe_control1_obj.set_dataframe(data_df)
            dataframe_control1_obj.dataframe_sorting(column='기준일')
            name_cyboss = dataframe_control1_obj.get_stock_item()
            data_df_cyboss = dataframe_control1_obj.get_dataframe()
            #print('name_cybos:{}'.format(name_cyboss))
            #print('data_df_cyboss:{}'.format(data_df_cyboss))

            data_df_cyboss['code'] = code
            data_df_cyboss['name'] = name
            #print('data_df_cyboss:{}'.format(data_df_cyboss))

            data_df_cyboss = data_df_cyboss.rename(index=str,
                                                   columns={'code': 'code', 'name': 'name', '기준일': 'date', '시가': 'open', '고가': 'high', '저가': 'low',
                                                            '종가': 'close', '거래량':'volume'})

            data_df_cyboss.reindex(columns=['code', 'name', 'date', 'open', 'high', 'low', 'close', 'volume'])
            print('data_df_cyboss:{}'.format(data_df_cyboss))
            data_df_cyboss['date'] = data_df_cyboss['date'].apply(str)
            data_df_cyboss['date'] = pd.to_datetime(data_df_cyboss['date'], format='%Y-%m-%d', errors='ignore').dt.date
            #print('data_df_cyboss:{}'.format(data_df_cyboss))

            ############################## 2. 해당 종목에 대해 My SQL에 저장되어 있는 정보를 읽어옴.#####################
            ### 1) 시세 테이블 stockmanagement

            df_from_MySQL = pd.read_sql_query("select * from stockprice where code = '{}'".format(code),
                                              mysql_control_obj.conn_pandas)
            print('df_from_MySQL:{}'.format(df_from_MySQL))

            ##########################################################################################################
            ########################### 4. 대신증권에서 읽어온 정보에 대하여 MySQL에 저장함.  ########################
            ###########################  해당 종목의 내용을 모두 drop 한 뒤 새로운 데이터프레임 write ################
            ##########################################################################################################
            try:
                pd.read_sql_query("DELETE FROM tensor.stockprice WHERE code = '{}'".format(code),
                                  mysql_control_obj.conn_pandas)
                data_df_cyboss.to_sql('stockprice', con=mysql_control_obj.conn_pandas, if_exists='append',
                                       index=False)
                # print('saved successfully 1')
                mysql_control_obj.save_db(data_df_cyboss, 'tensor', 'stockprice')
                # mysql_control_obj.set_off_connection_pandas()
            # # 해당 종목에 대한 지울 내용이 없을 때
            except:
                data_df_cyboss.to_sql('stockprice', con=mysql_control_obj.conn_pandas, if_exists='append',
                                       index=False)
                # print('saved successfully 2')
                mysql_control_obj.save_db(data_df_cyboss, 'tensor', 'stockprice')
                # mysql_control_obj.set_off_connection_pandas()

    def update_type_price(self, stocktype_typename):
        mysql_control_obj = MySQL_Sim_Control()
        mysql_control_obj.set_connection_obj_pandas(host='yourhost', user='user', password='password', db='tensor', charset='utf8')
        code = pd.read_sql_query(
            "select code from manage_stock where stocktype_typename = '{}'".format(stocktype_typename),
            mysql_control_obj.conn_pandas)

        codeList = code['code'].tolist()
        #print(codeList)

        data_total = pd.DataFrame()

        for code in codeList:
            print(code)
            data_df = pd.read_sql_query(
                "select name,date,open,high,low,close,volume from stockprice where code = '{}'".format(code),
                mysql_control_obj.conn_pandas)
            # print(data_df)
            data_total = data_total.append(data_df)
            print(data_total)

        data_total = data_total[['open','high','low','close','volume','date']]
        print('data_total:{}'.format(data_total))
        data_total_grouped = data_total.groupby('date').mean()
        print('data_total_grouped_index:{}'.format(data_total_grouped.index))
        data_total_grouped['date'] = data_total_grouped.index
        data_total_grouped.reset_index(drop=True, inplace=True)
        data_total_grouped['typename'] = stocktype_typename
        data_total_grouped = data_total_grouped[['typename', 'date', 'open', 'high', 'low', 'close', 'volume']]
        print('data_total_grouped:{}'.format(data_total_grouped))

        try:
            pd.read_sql_query("DELETE FROM tensor.typeprice WHERE typename = '{}'".format(stocktype_typename),
                              mysql_control_obj.conn_pandas)
            data_total_grouped.to_sql('typeprice', con=mysql_control_obj.conn_pandas, if_exists='append',
                                  index=False)
            # print('saved successfully 1')
            mysql_control_obj.save_db(data_total_grouped, 'tensor', 'typeprice')
            # mysql_control_obj.set_off_connection_pandas()
        # # 해당 종목에 대한 지울 내용이 없을 때
        except:
            data_total_grouped.to_sql('typeprice', con=mysql_control_obj.conn_pandas, if_exists='append',
                                  index=False)
            # print('saved successfully 2')
            mysql_control_obj.save_db(data_total_grouped, 'tensor', 'typeprice')
            # mysql_control_obj.set_off_connection_pandas()


if __name__ == "__main__":
    ### MYSQL 업데이트 하기###
    now = datetime.datetime.now()
    today = now.strftime('%Y%m%d')
    print('today:{}'.format(today))
    mysql_control_obj = MySQL_Sim_Control()
    mysql_control_obj.update_in_MySQL_code_management()
    #mysql_control_obj.update_in_MySQL_stock_price(type=1, from_date='20170101', to_date=today)
    #mysql_control_obj.update_in_MySQL_stock_price(type=2, from_date='20170101', to_date=today)
    mysql_control_obj.update_in_MySQL_stock_price(type=3, from_date='20170101', to_date=today)  # 3 관심종목만
    mysql_control_obj.update_type_price('음식료품')
    mysql_control_obj.update_type_price('건설업')
    mysql_control_obj.update_type_price('의약품')