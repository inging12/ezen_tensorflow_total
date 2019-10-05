from contacts2.model import Contacts2Model

class Contact2Controller:
    def __init__(self):
        self.model = Contacts2Model

    @staticmethod
    def print_menu():
        print('1. 입력')
        print('2. 출력')
        print('0. 종료')
        menu = input('메뉴 선택 : \n')
        return menu

    def run(self):
        contacts2 = []
        while 1:
            menu = self.print_menu()
            print('메뉴 : %s' % menu)
            if menu == '1':
                to_do = input('할 것\n')
                to_eat = input('먹을 것\n')
                t = self.model.set_info(to_do, to_eat)
                contacts2.append(t)
            elif menu == '2':
                print(self.model.get_info(contacts2))
            elif menu == '0':
                print('시스템 종료')
                break

