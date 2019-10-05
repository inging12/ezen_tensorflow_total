from textMining.model import SamsungReport

if __name__ == '__main__':
    #f = SamsungReport()
    #f.read_file()
    #print(SamsungReport.extract_hangeul(f))
    sam =SamsungReport()
    #sam.download()
    #print(sam.extract_noun())
    #print(sam.read_stopword())
    #sam.find_freq()
    print(sam.draw_wordCloud())
