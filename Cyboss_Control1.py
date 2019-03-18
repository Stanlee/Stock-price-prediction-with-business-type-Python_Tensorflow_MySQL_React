# !/usr/bin/python
# -*-coding:utf-8-*-
# 특정 종목 혹은 모든 종목에 대하여 특정 기간내의 시가, 고가, 저가, 종가를 읽어온다.
# 읽어온 결과를 각 폴더별로 엑셀파일로 저장함.
# Date : 2018년 2월 12일
# Author : 이인현
# 기능1 : Stock_Code 클래스: Cyboss에 접속하여 코드를 받아옴.
# 기능2 : CpStockChart 클래스 : 시세를 받아오는 objStockChart를 생성함.
# 기능3 : CpStockChart 클래스 RequestFromTo 함수: 종목 코드값을 기준으로 기간내에 기준일, 시가, 고가, 저가, 종가 정보를 수신.
#                                                 수신한 정보를 기반으로 Column명을 재배열하며, 기준일 index를 오름차순으로 정렬함.(최신데이터가 하단에 추가)

import win32com.client
import pandas as pd

g_objCpStatus = win32com.client.Dispatch('CpUtil.CpCybos')
instCpCodeMgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")

# Class 종목이름을 리스트로 저장하는 클래스 (1번 코스피, 2번 코스닥)
class Stock_Code:
    def __init__(self):
        self.instCpCodeMgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")

    # 모든종목의 code명을 받아오는 함수
    def getCodeList(self, type):
        codeList = self.instCpCodeMgr.GetStockListByMarket(type)
        # print(codeList)
        codeLength = len(codeList)
        #print(code_Length)
        return codeList, codeLength

    # 코드에 해당하는 종목명을 name에 저장함.
    def code_to_name(self, code):
        name = instCpCodeMgr.CodeToName(code)
        return name

# code명에 해당하는 종목의 시가,고가,저가,종가,거래량을 받아오는 함수
class CpStockChart:
    def __init__(self):
        self.objStockChart = win32com.client.Dispatch("CpSysDib.StockChart")
        self.name = 0 # 종목명
        self.data_df = 0 # 종목 시가,저가,고가,종가 데이터프레임

    def get_name_and_df(self):
        return self.name, self.data_df

    def RequestFromTo(self, code, fromDate, toDate, caller):

        # 코드에 해당하는 종목명을 name에 저장함.
        name = instCpCodeMgr.CodeToName(code)
        print('code:{}, name:{}, from:{}, to:{}'.format(code, name, fromDate, toDate))

        bConnect = g_objCpStatus.IsConnect
        if (bConnect == 0):
            print("PLUS가 정상적으로 연결되지 않음. ")
            return False

        self.objStockChart.SetInputValue(0, code)  # 종목코드
        self.objStockChart.SetInputValue(1, ord('1'))  # 기간으로 받기
        self.objStockChart.SetInputValue(2, toDate)  # To 날짜 (전날까지만 보유하고 있음. 당일정보는 별도의 클래스 활용)
        self.objStockChart.SetInputValue(3, fromDate)  # From 날짜
        # self.objStockChart.SetInputValue(4, 500)  # 최근 500일치
        self.objStockChart.SetInputValue(5, [0, 2, 3, 4, 5, 8])  # 날짜,시가,고가,저가,종가,거래량 8
        self.objStockChart.SetInputValue(6, ord('D'))  # '차트 주기 - 일간 차트 요청
        self.objStockChart.SetInputValue(9, '1')  # 수정주가 사용
        self.objStockChart.BlockRequest()

        rqStatus = self.objStockChart.GetDibStatus()
        rqRet = self.objStockChart.GetDibMsg1()
        # print("통신상태", rqStatus, rqRet)
        # if rqStatus != 0:
        #     exit()

        len = self.objStockChart.GetHeaderValue(3)
        #print('len:{}'.format(len))

        caller.dates = []
        caller.opens = []
        caller.highs = []
        caller.lows = []
        caller.closes = []
        caller.vols = []
        for i in range(len):
            caller.dates.append(self.objStockChart.GetDataValue(0, i))
            caller.opens.append(self.objStockChart.GetDataValue(1, i))
            caller.highs.append(self.objStockChart.GetDataValue(2, i))
            caller.lows.append(self.objStockChart.GetDataValue(3, i))
            caller.closes.append(self.objStockChart.GetDataValue(4, i))
            caller.vols.append(self.objStockChart.GetDataValue(5, i))

        data_dict = {'기준일': caller.dates,
                     '시가': caller.opens,
                     '고가': caller.highs,
                     '저가': caller.lows,
                     '종가': caller.closes,
                     '거래량': caller.vols
        }
        self.name = name
        self.data_df = pd.DataFrame(data_dict)
        self.data_df = self.data_df.reindex(columns=['기준일','시가','고가','저가','종가', '거래량'])
        self.data_df = self.data_df.sort_values(by=['기준일']) # 오름차순으로 정렬 향후 수동 추가시 최근일자의 내용을 가장 하단에 추가하기 위함.

if __name__ == "__main__":
    pass