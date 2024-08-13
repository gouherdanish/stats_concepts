import numpy as np
from typing import Any

class RandomEvent:
    """
    Defines a random event whose probability is given
        `x` represents the event
        `prob` represents the probability of that event

    Usage:
        event1 = RandomEvent(x='Getting a six when a fair die is thrown', prob=1/6)
        event2 = RandomEvent(x='Getting a head when a coin is flipped', prob=1/2)

        print(event1.information)   # 1.791759469228055
        print(event2.information)   # 0.693147180559945
    """
    def __init__(self,x:Any='Sample Event',prob:float=1.0) -> None:
        self.x = x
        self.prob = prob
            
    def __str__(self) -> str:
        return f"Event(x='{self.x}', p(x)={self.prob:.2f})"
    
    def __repr__(self) -> str:
        return str(self)

    @property
    def information(self):
        """
        How much surprise one gets when a specific event happens
        """
        return -np.log(self.prob)