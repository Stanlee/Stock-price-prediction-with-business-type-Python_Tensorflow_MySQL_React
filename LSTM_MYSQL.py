#-*-coding:utf-8-*-
#!/usr/bin/python

import numpy as np # linear algebra
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keraspp import skeras
import math, time
import os
import pandas as pd

import pymysql
from sqlalchemy import *
import datetime
pymysql.install_as_MySQLdb()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from matplotlib import rcParams, style
plt.rcParams['font.family'] = 'HYGungSo-Bold'

# dataset
class LSTM_Each:
    def __init__(self, symbol_name):
        self.price_dataset = 0
        self.symbol_name = symbol_name
        self.stock_price_close = 0
        self.stock_price_open = 0
        self.stock_price_high = 0
        self.stock_price_low = 0
        self.stock_price_volume = 0
        self.stock_price_total=0
        self.trainX = 0
        self.trainY =0
        self.testX =0
        self.testY =0
        self.model = 0
        self.min_max_scaler_open = preprocessing.MinMaxScaler()
        self.min_max_scaler_close = preprocessing.MinMaxScaler()
        self.min_max_scaler_low = preprocessing.MinMaxScaler()
        self.min_max_scaler_high = preprocessing.MinMaxScaler()
        self.min_max_scaler_volume = preprocessing.MinMaxScaler()
        self.score_evalutation = pd.DataFrame()
        self.score_evalutation_new = pd.DataFrame()

    def get_score_evalutation(self):
        return self.score_evalutation

    def get_price_dataset(self):
        return self.score_evalutation

    def set_price_dataset(self, price_dataset):
        self.price_dataset = price_dataset

    def set_trainX(self, trainX):
        self.trainX = trainX
        print(self.trainX.shape)
    def set_trainY(self, trainY):
        self.trainY = trainY
    def set_testX(self, testX):
        self.testX = testX
    def set_testY(self, testY):
        self.testY = testY

    def normalize_data(self, df):
        df['open'] = self.min_max_scaler_open.fit_transform(df.open.values.reshape(-1, 1))
        df['low'] = self.min_max_scaler_low.fit_transform(df.low.values.reshape(-1, 1))
        df['high'] = self.min_max_scaler_high.fit_transform(df.high.values.reshape(-1, 1))
        df['volume'] = self.min_max_scaler_volume.fit_transform(df.volume.values.reshape(-1, 1))
        df['close'] = self.min_max_scaler_close.fit_transform(df.close.values.reshape(-1, 1))
        return df

    def model_score(model, X_train, y_train, X_test, y_test):
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
        return trainScore[0], testScore[0]

    # 데이터 준비
    def load_data(self, stock, seq_len):
        amount_of_features = len(stock.columns)  # 5
        data = stock.as_matrix()
        sequence_length = seq_len + 1  # index starting from 0
        result = []

        for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
            result.append(data[index: index + sequence_length])  # index : index + 22days

        result = np.array(result)
        row = round(0.9 * result.shape[0])  # 90% split
        train = result[:int(row), :]  # 90% date, all features

        x_train = train[:, :-1]             # 마지막 열인 close를 제외한 값만 추출
        y_train = train[:, -1][:, -1]       # 마지막 열인 close만 추출

        x_test = result[int(row):, :-1]         # 마지막 열인 close를 제외한 값만 추출
        y_test = result[int(row):, -1][:, -1]   # 마지막 열인 close만 추출
        #print('y_test:{}'.format(y_test))

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
        #print('x_test:{}'.format(x_test))

        return [x_train, y_train, x_test, y_test]

# Step 2 Build Model
    def build_model(self, layers):
        d = 0.3
        model = Sequential()

        model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))

        model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))

        model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
        model.add(Dense(1, kernel_initializer="uniform", activation='linear'))

        # adam = keras.optimizers.Adam(decay=0.2)

        start = time.time()
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print("Compilation Time : ", time.time() - start)

        return model

    def denormalize(self, normalized_value):
        #df = df.close.values.reshape(-1, 1)
        normalized_value = normalized_value.reshape(-1, 1)
        new = self.min_max_scaler_close.inverse_transform(normalized_value)
        return new

    def model_score(self, model, X_train, y_train, X_test, y_test,symbol_name):
        print("symbol_name:",symbol_name)
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

        return trainScore[0], testScore[0]

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
        mysql_control_obj.set_connection_obj_pandas(host='assist2018-mysql.ccnyonovq9it.ap-northeast-2.rds.amazonaws.com', user='team6', password='t6@assist', db='tensor', charset='utf8')
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
        mysql_control_obj.set_connection_obj_pandas(host='assist2018-mysql.ccnyonovq9it.ap-northeast-2.rds.amazonaws.com', user='team6', password='t6@assist', db='tensor', charset='utf8')
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

    def result_prediction(self, price_dataset_stock_2, code, result, prediction):
        result_df = pd.DataFrame(result)
        prediction_df = pd.DataFrame(prediction)
        count = len(result)

        #print('result_df:{}'.format(result_df))
        #print('prediction_df:{}'.format(prediction_df))
        #print('price_dataset_stock_2:{}'.format(price_dataset_stock_2))
        result_prediction_df = pd.DataFrame()
        result_prediction_df['result'] = result_df[0]
        result_prediction_df['prediction'] = prediction_df[0]
        result_prediction_df['code'] = code
        result_prediction_df = result_prediction_df[['code', 'result', 'prediction']]
        result_prediction_df =result_prediction_df.reset_index(drop=True)
        price_dataset_stock_2 = price_dataset_stock_2.reset_index(drop=True)

        result_prediction_df = pd.concat([price_dataset_stock_2, result_prediction_df], axis=1)
        print('result_prediction_df:{}'.format(result_prediction_df))

        try:
            pd.read_sql_query("DELETE FROM tensor.result_prediction WHERE code = '{}'".format(code),
                              mysql_control_obj.conn_pandas)
            result_prediction_df.to_sql('result_prediction', con=mysql_control_obj.conn_pandas, if_exists='append',
                                  index=False)
            # print('saved successfully 1')
            mysql_control_obj.save_db(result_prediction_df, 'tensor', 'result_prediction_df')
            # mysql_control_obj.set_off_connection_pandas()
        # # 해당 종목에 대한 지울 내용이 없을 때
        except:
            result_prediction_df.to_sql('result_prediction', con=mysql_control_obj.conn_pandas, if_exists='append',
                                  index=False)
            # print('saved successfully 2')
            mysql_control_obj.save_db(result_prediction_df, 'tensor', 'result_prediction')
            # mysql_control_obj.set_off_connection_pandas()

if __name__ == "__main__":

    #mysql_control_obj2 = MySQL_Sim_Control()
    #mysql_control_obj2.update_type_price('음식료품')

    # mysql 읽어오기
    now = datetime.datetime.now()
    today = now.strftime('%Y%m%d')
    print('today:{}'.format(today))
    mysql_control_obj = MySQL_Sim_Control()
    mysql_control_obj.set_connection_obj_pandas(host='assist2018-mysql.ccnyonovq9it.ap-northeast-2.rds.amazonaws.com', user='team6', password='t6@assist', db='tensor', charset='utf8')
    price_dataset = mysql_control_obj.read_table_by_df('stockprice')
    print(price_dataset)
    #type_price_dataset = mysql_control_obj.read_table_by_df('typeprice')
    #print(type_price_dataset)
    codes = price_dataset['code']
    codes = list(set(codes))
    print('codes:{}'.format(codes))

    train_total = []
    test_total = []

    windows = [1, 5, 10, 22]

    for window in windows:
        for code in codes:
            flag = 1
            print('price_dataset:{}'.format(price_dataset))
            price_dataset_stock = price_dataset.loc[price_dataset.code == code]
            print('price_dataset:{}'.format(price_dataset_stock))
            price_dataset_stock_1 = price_dataset_stock[['open', 'low', 'high', 'volume','close']]
            price_dataset_stock_2 = price_dataset_stock[['date', 'code', 'name']]
            price_dataset_stock_copy = price_dataset_stock_1.copy(deep=True)  # deep copy를 하여 원본 데이터를 보존한다.
            symbol_name = code
            print(code)


            window = 1
            # 입력 X=t 대한 출력 Y=t+1로 reshape
            lstm_obj = LSTM_Each(symbol_name)
            lstm_obj.set_price_dataset(price_dataset_stock_copy)
            price_dataset_norm = lstm_obj.normalize_data(lstm_obj.price_dataset)
            #print('price_dataset_norm:{}'.format(price_dataset_norm))

            X_train, y_train, X_test, y_test = lstm_obj.load_data(price_dataset_norm, window)
            #print(X_train[0], y_train[0]) # 22일의 time series의 X_train과 22일째의 y_train값 한 세트

            #print('X_train_shape:{}'.format(X_train.shape))
            #print('y_train_shape:{}'.format(y_train.shape))
            #print('X_test_shape:{}'.format(X_test.shape))
            #print('y_test_shape:{}'.format(y_test.shape))

            #print(X_train[0], y_train[0])

            model = lstm_obj.build_model([5, window, 1])
            h = model.fit(X_train, y_train, batch_size=512, epochs=30, validation_split=0.1, verbose=1)

            fig, ax = plt.subplots(nrows=1, ncols=1)
            skeras.plot_loss(h)
            plt.title('History of training')
            #plt.show()
            #skeras.save_history_history(symbol_name, h, fold='/home/inhyunlee/Desktop/Stock1')
            #fig.savefig('//home/inhyunlee/Desktop/Stock1/{}_training_history.png'.format(symbol_name))

            diff = []
            ratio = []
            predicted_test_norm = model.predict(X_test)  # test set의 X(normalized)값에 대해서 예측한 y(normalized)값을 p라고 명함.
            print(predicted_test_norm.shape)
            # for each data index in test data
            for u in range(len(y_test)):   # y_test의 날짜 만큼 예측 값을 기록함.
                # pr = prediction day u
                pr = predicted_test_norm[u][0]
                # (y_test day u / pr) - 1
                ratio.append((y_test[u] / pr) - 1)      # 비율
                diff.append(abs(y_test[u] - pr))        # 차이

            predicted_test = lstm_obj.denormalize(predicted_test_norm)
            newy_test = lstm_obj.denormalize(y_test)
            #print(newy_test)
            #print('predicted_test:{}'.format(predicted_test))
            #print('newy_test:{}'.format(newy_test))

            train, test = lstm_obj.model_score(model, X_train, y_train, X_test, y_test, code)

            train_total.append(train)
            test_total.append(test)

            #print(train_total)
            #print(test_total)
            total_dataframe = {'train_score': train_total, 'test_score': test_total}

            total_dataframe = pd.DataFrame(total_dataframe)
            print(total_dataframe)

            temp_dir_score = "/home/inhyunlee/Desktop/{}.xlsx".format(window)
            writer = pd.ExcelWriter(temp_dir_score, engine='xlsxwriter')
            total_dataframe.to_excel(writer, 'Sheet1', index=False)
            writer.save()

            #fig, ax = plt.subplots(nrows=1, ncols=1)
            #ax.set_title(folder_name)
            #ax.plot(predicted_test, color='red', label='Prediction')
            #ax.plot(newy_test, color='blue', label='Actual')
            #ax.legend(loc='best')
            #plt.show()
            #fig.savefig('/home/inhyunlee/Desktop/{}/{}.png'.format(window, symbol_name))
            print('code : {}'.format(code))
            print('Prediction:{}'.format(predicted_test))
            print('Actual:{}'.format(newy_test))

            ## 날짜의 개수 역순 출력
            date_count = len(newy_test)
            print('date_count:{}'.format(date_count))
            price_dataset_stock_2 = price_dataset_stock_2[-date_count:]
            #print('price_dataset_stock_2:{}'.format(price_dataset_stock_2))

            mysql_control_obj.result_prediction(price_dataset_stock_2, code, predicted_test, newy_test)