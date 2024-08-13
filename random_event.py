import numpy as np

class RandomEvent:
    def __init__(self,x,px) -> None:
        self.x = x
        self.px = px

    def __str__(self) -> str:
        return f'Event(x={self.x}, p(x)={self.px})'

    @property
    def information(self):
        """
        How much surprise one gets when a specific event happens
        """
        return -np.log(self.px)