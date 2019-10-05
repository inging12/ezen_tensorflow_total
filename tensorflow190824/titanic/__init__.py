from titanic.controller import TitanicController
from titanic.view import TitanicView

if __name__ == '__main__':
    def print_menu():
        print('0. EXIT')
        print('1. LEARNING MACHINE')
        print('2. VIEW : plot_survived_dead')
        print('3. TEST ACCURACY')
        print('4. SUBMIT')
        return input('CHOOSE ONE \n')

    while 1 :
        menu = print_menu()
        print('MENU : %s' % menu)
        if menu == '0':
            print('** EXIT **')
            break
        elif menu == '1':
            print('** CREATE TRAIN **')
            ctrl = TitanicController()
            t = ctrl.create_train()
            print('** t 모델 **')
            print(t)
            break
        elif menu == '2':
            view = TitanicView()
            t = view.create_train()
            #view.plot_survived_dead(t)
            #view.plot_sex(t)
            view.bar_chart(t,'Pclass')
            break
        elif menu == '3':
            ctrl = TitanicController()
            t = ctrl.create_train()
            ctrl.test_all()
            break

        elif menu == '4':
            ctrl = TitanicController()
            t = ctrl.create_train()
            ctrl.submit()
            break

    '''
    # 19.08.24
    ctrl = TitanicController()
    t = ctrl.create_train()


    """
    --------------------- train head & column -------------------------
       PassengerId  Pclass  ...     Fare Embarked
    0          892       3  ...   7.8292        Q
    1          893       3  ...   7.0000        S
    2          894       2  ...   9.6875        Q
    3          895       3  ...   8.6625        S
    4          896       3  ...  12.2875        S
    """
    '''