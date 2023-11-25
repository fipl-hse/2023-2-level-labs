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
    def __init__(self, max_speed, colour: str):
        self.max_speed = max_speed
        self.colour = colour

    def move(self):
        return print("moving")

# Car
    # Attribute:
        # max_speed
        # colour
        # fuel
    # Methods:
        # move
        # stay


class Car(Vehicle):
    def __init__(self, max_speed, colour, fuel: str):
        super().__init__(max_speed, colour)
        self.fuel = fuel

    def stay(self):
        return print("staying")


LADA = Car(200, "purple", "96")
print(LADA.fuel, LADA.max_speed, LADA.colour, LADA.stay())


# Bicycle
    # Attributes:
        # number_of_wheels
        # colour
        # max_speed
    # Methods:
        # move
        # freestyle

class Bicycle(Vehicle):
    def __init__(self, max_speed, colour, number_of_wheels: int):
        super().__init__(max_speed, colour)

    def freestyle(self):
        return print("freestyling")


stels = Bicycle(30, 'yellow', 2)
print(stels.colour)
stels.move()
stels.freestyle()


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
