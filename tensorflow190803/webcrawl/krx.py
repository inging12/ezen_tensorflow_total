import pandas as pd

class KrxCrawler:
    def __init__(self, param):
        self.param = param

    def scrap(self):
        code = pd.read_html(self.param, header='0')[0]
        print(code)