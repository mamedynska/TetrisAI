import random
from block import Block

class TetrisBlock(Block):
    Tetrisblocks = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],
        [[4, 5, 9, 10], [2, 6, 5, 9]],
        [[6, 7, 9, 10], [1, 5, 6, 10]],
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]

    Colors = [    
        (0, 0, 0),
        (120, 40, 180),
        (100, 180, 180),
        (80, 35, 20),
        (80, 135, 20),
        (180, 35, 20),
        (180, 35, 120)
    ]

    def __init__(self,coordinates):
        self.coordinates = coordinates
        self.rotation = 0
        self.shape = random.randint(0, len(self.Tetrisblocks) - 1)
        self.color = random.randint(0, len(self.Colors) - 1)

    def representation(self):
        return self.Tetrisblocks[self.shape][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.Tetrisblocks[self.shape])

    def reset(self):
        for _ in range(self.coordinates):
            self.coordinates.y = 0