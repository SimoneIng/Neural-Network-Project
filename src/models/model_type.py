from enum import Enum

class ModelType(Enum):
    MLP = 1
    CNN = 2

    def __str__(self):
        return self.name