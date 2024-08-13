import numpy as np
from typing import List
from random_event import RandomEvent

class RandomVariable:
    """
    Random variable assigns a numerical value to a randomized experiment
    e.g. 
        Experiment - Coin Flips
        X       = number of heads in 3 consecutive coin flips = {0,1,2,3}
        p(X)    = {1/8,3/8,3/8,1/8}
    """
    def __init__(self,events: List[RandomEvent]) -> None:
        self.events = events
        self._X     = np.array([event.x for event in self.events])
        self._pX    = np.array([event.px for event in self.events])

    def __str__(self) -> str:
        return str(self.distribution())
    
    def distribution(self):
        return {k:v for k,v in zip(self._X,self._pX)}
    
    def information(self):
        return np.array([event.information for event in self.events])

    def expectation(self):
        return sum(self._pX * self._X)
    
    def entropy(self):
        """
        Measures the average surprise or uncertainty or randomness in a probability distribution
        """
        return sum(self._pX * self.information())

if __name__=='__main__':
    # Number of heads in 3 consective coin flips and their probabilities
    events = [
        RandomEvent(x=0,prob=0.125),
        RandomEvent(x=1,prob=0.375),
        RandomEvent(x=2,prob=0.375),
        RandomEvent(x=3,prob=0.125),
    ]
    X = RandomVariable(events)
    print(f'Random Variable, X = {X}')
    print(f'Information, I(X) = {X.information()}')
    print(f'Expection, E(X) = {X.expectation()}')
    print(f'Entropy, S(X) = {X.entropy()}')