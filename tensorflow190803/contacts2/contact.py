class Contact2:
    def __init__ (self, to_do, to_eat):
        self.to_do = to_do
        self.to_eat = to_eat

    def to_string(self):
        t = ' 할 것: {} \n 먹을 것 {} \n'.format(self.to_do,self.to_eat)
        return t