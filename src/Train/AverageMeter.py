'''
 # @ author: bella | bob
 # @ date: 2024-09-18 21:34:20
 # @ license: MIT
 # @ description:
 '''

class AverageMeter:
    
    def __init__(self) -> None:
        self.reset()
        pass
    
    def reset(self) -> None:
        self.value: float = 0
        self.average: float = 0
        self.sum: float = 0
        self.count: int = 0
        pass
    
    def update(self, value: float, n: int = 1) -> None:
        self.value: float = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count
        pass
    
    def __str__(self) -> str:
        return f"value: {self.value}, average: {self.average}, sum: {self.sum}, count: {self.count}"
        pass
    
    def __repr__(self) -> str:
        return self.__str__()
        pass
    
    pass