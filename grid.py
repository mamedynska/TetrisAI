from coordinates import Coordinates
from tetrisBlock import TetrisBlock


class Grid:

    #boolean coordArray[x][y];
    #figure = None

    def __init__(self,x,y):

        self.maxX = x
        self.maxY = y
        #self.coordArray = [[0]*x for i in range(y)]

        self.coordArray = []
        for i in range(self.maxY):
            newLine = []
            for j in range(self.maxX):
                newLine.append(0)
            self.coordArray.append(newLine)

        self.newBlock = None

        self.score = 0
        self.isGameOver = False

    def spawnNewBlock(self):
        self.newBlock = TetrisBlock(Coordinates(3,0))

    def moveBlock(self,direction):
        tmpBlockX = self.newBlock.coordinates.x
        if direction == "r":
            self.newBlock.coordinates.x += 1
        elif direction == "l":
            self.newBlock.coordinates.x -= 1
        
        if self.checkCollisions():
            self.newBlock.coordinates.x = tmpBlockX

    def rotateBlock(self):
        old_rotation = self.newBlock.rotation
        self.newBlock.rotate()
        if self.checkCollisions():
            self.newBlock.rotation = old_rotation


    def checkCollisions(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.newBlock.representation():
                    if i + self.newBlock.coordinates.y > self.maxY - 1 or \
                            j + self.newBlock.coordinates.x > self.maxX - 1 or \
                            j + self.newBlock.coordinates.x < 0 or \
                            self.coordArray[i + self.newBlock.coordinates.y][j + self.newBlock.coordinates.x] > 0:
                        return True
        return False

    def addToGrid(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.newBlock.representation():
                    self.coordArray[i + self.newBlock.coordinates.y][j + self.newBlock.coordinates.x] = self.newBlock.color

    def endOfGrid(self):
        self.addToGrid()
        self.clearLines()
        self.spawnNewBlock()
        if self.checkCollisions():
            self.isGameOver = True

    #Returns False if the block couldn't fall
    #else True (if the block successfully fell)
    def fallBlock(self):
        tmpBlockY = self.newBlock.coordinates.y
        self.newBlock.coordinates.y += 1
        if self.checkCollisions():
            self.newBlock.coordinates.y = tmpBlockY
            self.endOfGrid()
            return False
        return True


    def fallBlockUntilTheEnd(self):
        while self.fallBlock():
            self.fallBlock()


    def clearLines(self):
        lines = 0
        for i in range(1, self.maxY):
            zeros = 0
            for j in range(self.maxX):
                if self.coordArray[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.maxX):
                        self.coordArray[i1][j] = self.coordArray[i1 - 1][j]

        self.score += lines ** 2

    def linesToClear(self):
        lines = 0
        for i in range(1, self.maxY):
            zeros = 0
            for j in range(self.maxX):
                if self.coordArray[i][j]:
                    zeros += 1
            if zeros == 0:
                lines += 1
        return lines