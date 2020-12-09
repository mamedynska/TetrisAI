from block import Block
from grid import Grid
from tetrisBlock import TetrisBlock
import tetris_ai

import pygame


class Tetris:

    #Grid board = new Grid(10,15);

    def __init__(self):
        self.board = Grid(10,15)

    def displayGame(self):

        pygame.init()

        frameRate = 25
        coordSize = 20

        x = 100
        y = 60

        gameWindow = pygame.display.set_mode((400,600))
        clock = pygame.time.Clock()
        pygame.display.set_caption("Tetris")

        while True:
            self.game(gameWindow,clock)

        pygame.quit()

    def game(self, gameWindow, clock):
        coordSize = 20
        frameRate = 25

        x = 100
        y = 60

        while not self.board.isGameOver:
            if self.board.newBlock is None:
                self.board.spawnNewBlock()
                if self.board.checkCollisions():
                    self.board.isGameOver = True

            gameWindow.fill((255,255,255))

            self.keyPressed()

            for i in range(self.board.maxY):
                for j in range(self.board.maxX):
                    pygame.draw.rect(gameWindow, (120,120,120), [x + coordSize * j, y + coordSize * i, coordSize, coordSize], 1)
                    if self.board.coordArray[i][j] > 0:
                        pygame.draw.rect(gameWindow, self.board.newBlock.color,
                                        [x + coordSize * j + 1, y + coordSize * i + 1, coordSize - 2, coordSize - 1])


            if self.board.newBlock is not None:
                for i in range(4):
                    for j in range(4):
                        p = i * 4 + j
                        if p in self.board.newBlock.representation():
                            pygame.draw.rect(gameWindow, self.board.newBlock.color,
                                            [x + coordSize * (j + self.board.newBlock.coordinates.x) + 1,
                                            y + coordSize * (i + self.board.newBlock.coordinates.y) + 1,
                                            coordSize - 2, coordSize - 2])


            font = pygame.font.SysFont('Calibri', 25, True, False)
            text = font.render("Score: " + str(self.board.score), True, (0,0,0))
            gameWindow.blit(text, [0, 0])

            pygame.display.flip()
            clock.tick(frameRate)
        
        self.reset()

#TODO :  
#   - get_holes(self, board) : return nb_holes
#   - get_bumpiness_and_height : return total_bumpiness, total_height
#   - 
#
#
    def get_states(self):
        lines_cleared = self.check_cleared_rows()
        holes = self.get_holes()
        bumpiness, height = self.get_bumpiness_and_height()
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    #TODO
    def check_cleared_rows(self):
        if self.board.fullLines is not None:
            if self.board.fullLines != self.board.linesToClear():
                return self.board.fullLines
        return self.board.linesToClear()

    #TODO
    def get_holes(self):
        #   num_holes = 0
        #     for col in zip(*board):
        #         row = 0
        #         while row < self.height and col[row] == 0:
        #             row += 1
        #         num_holes += len([x for x in col[row + 1:] if x == 0])
        #     return num_holes

        ########################

        #Soit on a : (1er cas)
        #   X    X    X
        #      trou
        #ou (2eme cas)
        #        X
        #   X  trou   X

        holes = 0
        #boucles de parcours x en y
        for i in range(self.board.maxY):
            for j in range(self.board.maxX):

                    #On repère là où il y a un bloc
                    #(plus opti que de vérifier là où il n'y a pas de bloc)
                    #Et si le bloc n'est pas posé tout en bas (i < maxY)
                    #==> Je cherche à verifier si ce bloc GENE un potentiel trou!
                    if self.board.coordArray[i][j] > 0 and i < self.board.maxY:
                        yToCheck = i + 1

                        #On regarde si le bloc d'en dessous est vide
                        if (self.board.coordArray[yToCheck][j] == 0):
                            #On vérifie si ce vide est un trou
                            
                            #On regarde s'il faut vérifier les 2 autres emplacements
                            if j > 0 and j < self.board.maxX:
                                #2eme cas
                                if (self.board.coordArray[yToCheck][j + 1] > 0) and (self.board.coordArray[yToCheck][j - 1] > 0):
                                    holes += 1
                                #1er cas
                                elif (self.board.coordArray[i][j + 1] > 0) and (self.board.coordArray[i][j - 1] > 0):
                                    holes +=1

                            #OU un seul, selon si le vide est situé en bord de plateau
                            else:
                                if j == 0:
                                    #2eme cas
                                    if (self.board.coordArray[yToCheck][j + 1] > 0):
                                        holes += 1
                                    #1er cas
                                    elif (self.board.coordArray[i][j + 1] > 0):
                                        holes +=1

                                elif j == self.board.maxX:
                                    #2eme cas
                                    if (self.board.coordArray[yToCheck][j - 1] > 0):
                                        holes += 1
                                    #1er cas
                                    elif(self.board.coordArray[i][j - 1] > 0):
                                        holes +=1

        return holes



    #TODO
    def get_bumpiness_and_height(self):
        # height : '''Sum and maximum height of the board'''
        totalHeight = 0
        totalBumpiness = 0
        heights = []
        for x in range(self.board.maxX):
            y = 0
            while y < self.board.maxY:
                if self.board.coordArray[y][x] == 0:
                    y += 1
            heights.append(y)
            totalHeight += self.board.maxY - y

        # bumpiness : '''Sum of the differences of heights between pair of columns'''
        for i in range(len(heights) - 1):
            totalBumpiness += abs(heights[i] - heights[i + 1])

        return totalBumpiness, totalHeight


    def get_next_states(self):
        states = {}
        for i in range(4):
            for x in range(self.board.maxX):
                while self.board.fallBlock():
                    pass
                states[(self.board.newBlock.coordinates.x, i)] = self.get_states()
            self.board.newBlock.reset()
        return states

    def reset(self):
        self.board = Grid(10,15)
        return self.get_states()
        

    def keyPressed(self):
        # for event in list(pygame.event.get()) + tetris_ai.run_ai():
        canFall = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.board.isGameOver = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.board.newBlock.rotate()

                if event.key == pygame.K_DOWN:
                    canFall = self.board.fallBlock()

                if event.key == pygame.K_LEFT:
                    self.board.moveBlock("l")

                if event.key == pygame.K_RIGHT:
                    self.board.moveBlock("r")

                if event.key == pygame.K_SPACE:
                    self.board.fallBlockUntilTheEnd()
                    canFall = False
    
        if canFall is not None and not canFall:
            self.board.endOfGrid()


    def step(self, action):
        x, rotations = action
        for _ in range(rotations):
            self.board.rotateBlock()

        self.board.newBlock.coordinates.x = x            

        while self.board.fallBlock():
            pass

        self.board.endOfGrid()

        if not self.board.isGameOver:
            self.board.spawnNewBlock()
        else:
            self.board.score -= 2

        return self.board.score, self.board.isGameOver