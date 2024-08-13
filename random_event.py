import numpy as np

class RandomEvent:
    def __init__(self,x=0,prob=0) -> None:
        self.x = x
        self.prob = prob

    def __str__(self) -> str:
        return f'Event(x={self.x}, p(x)={self.prob:.2f})'
    
    def __repr__(self) -> str:
        return str(self)

    @property
    def information(self):
        """
        How much surprise one gets when a specific event happens
        """
        return -np.log(self.prob)