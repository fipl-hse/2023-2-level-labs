"""
Programming 2023
Seminar 8
Intoduction to class inheritance
"""
# pylint:disable=missing-class-docstring,too-few-public-methods
class Vehicle:
    def __init__(self, max_speed: int, color: str) -> None:
        self._max_speed = max_speed
        self._color = color

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
    def __init__(self, max_speed: int, color: str, fuel: float):
        super().__init__(max_speed, color)
        self._fuel = fuel

    def move(self):
        while self._fuel > 0:
            self._fuel -= 1
            print(f'Moving...{self._fuel} litres of fuel for {}')
        print('No fuel left')


    def stay(self):
        if self._fuel:
            return None
        print('The car is staying')
LADA = Car( )


# Bicycle
    # Attributes:
        # number_of_wheels
        # colour
        # max_speed
    # Methods:
        # move
        # freestyle

class Bicycle(Vehicle):

    def __init__(self, color, max_speed, number_of_wheels):
        self._number_of_wheels = number_of_wheels
        self._color = color
        self._max_speed = max_speed

    def move(self):
        pass

    def freestyle(self):
        pass


# stels = Bicycle('yellow', 30, 2)
# print(stels.colour)
# stels.move()
# stels.freestyle()


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
