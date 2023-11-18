"""
Programming 2023
Seminar 8
Intoduction to class inheritance
"""
# pylint:disable=missing-class-docstring,too-few-public-methods
# Vehicle
    # Attributes:
        # max_speed
        # colour
    # Methods:
        # move


class Vehicle:
    def __init__(self, max_speed: int, colour: str) -> None:
        self._max_speed = max_speed
        self._colour = colour

    def move(self):
        pass


# Car
    # Attribute:
        # max_speed
        # colour
        # fuel
    # Methods:
        # move
        # stay


class Car(Vehicle):
    def __init__(self, max_speed: int, colour: str, fuel: float):
        super().__init__(max_speed, colour)
        self._fuel = fuel

    def move(self) -> None:
        while self._fuel:
            self._fuel -= 1
            print('Мошино запрвлено сполна.')
        print('Мошино не заправлено.')

    def stay(self) -> None:
        if self._fuel:
            return
        print('popa.')


LADA = Car(100, 'eggplant', 95)
LADA.move()
LADA.stay()


# Bicycle
    # Attributes:
        # number_of_wheels
        # colour
        # max_speed
    # Methods:
        # move
        # freestyle


class Bicycle(Vehicle):
    def __init__(self, number_of_wheels: int, max_speed: int, colour: str) -> None:
        super().__init__(max_speed, colour)
        self._number_of_wheels = number_of_wheels

    def move(self):
        print(f'I go with a speed of {self._number_of_wheels * 2} because I have {self._number_of_wheels} wheels.')

    def freestyle(self):
        pass


stels = Bicycle(2, 30, 'yellow')
print(stels._colour)
stels.move()
stels.freestyle()

lst = [stels, LADA]
for i in lst:
    i.move()

# Aircraft
    # Attributes:
        # number_of_engines
        # colour
        # max_speed
    # Methods:
        # move

class Aircraft:
    ...


# for...
