from calculator.model import CalculatorModel

class CalculatorController:
    def __init__(self, num1, num2):
        self.calc = CalculatorModel(num1, num2)

    def exec(self, op):
        if op == '+':
            return self.calc.add()

        elif op == '-':
            return self.calc.sub()

        elif op == '*':
            return self.calc.mul()

        elif op == '/':
            return self.calc.div()

