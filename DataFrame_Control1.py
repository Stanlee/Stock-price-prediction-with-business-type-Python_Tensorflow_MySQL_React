#!/usr/bin/python
#-*-coding:utf-8-*-
from Cyboss_Control1 import *

class DataFrame_Control1:
    def __init___(self, stock_item='', dataframe=pd.DataFrame()):
        self.stock_item = stock_item
        self.dataframe = dataframe
        self.new_dataframe = 0

    def set_stock_item(self, stock_item):
        self.stock_item = stock_item

    def set_dataframe(self, dataframe):
        self.dataframe = dataframe

    def get_stock_item(self):
        return self.stock_item

    def get_dataframe(self):
        return self.dataframe

    def get_new_dataframe(self):
        return self.new_dataframe

    def dataframe_sorting(self, column, ascending=True):
        self.dataframe = self.dataframe.sort_values(by=column, ascending=ascending)

if __name__ == "__main__":
    pass