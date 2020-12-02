from block import Block
from grid import Grid
from tetrisBlock import TetrisBlock
import tetris_ai

import pygame


class Tetris:

    #Grid board = new Grid(10,15);
    #int score = 0
    #int counter = 0

    def __init__(self):
        self.board = Grid(10,15)
        self.score = 0
        self.counter = 0
        self.downButton = False

    def game(self):

        pygame.init()

        frameRate = 25
        coordSize = 20

        x = 100
        y = 60

        gameWindow = pygame.display.set_mode((400,600))
        clock = pygame.time.Clock()
        pygame.display.set_caption("Tetris")

        while not self.board.isGameOver:
            if self.board.newBlock is None:
                self.board.spawnNewBlock()
                if self.board.checkCollisions():
                    self.board.isGameOver = True

            self.counter += 1
            if self.counter > 100000:
                self.counter = 0

            if self.counter % (frameRate // 2) == 0 or self.downButton:
                if not self.board.isGameOver:
                    self.board.fallBlock()


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
        
        pygame.quit()


#TODO :  
#   - get_holes(self, board) : return nb_holes
#   - get_bumpiness_and_height : return total_bumpiness, total_height
#   - 
#
#
    def get_states(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    #TODO
    def check_cleared_rows(board):
        pass
        # to_delete = []
        # for i, row in enumerate(board[::-1]):
        #     if 0 not in row:
        #         to_delete.append(len(board) - 1 - i)

        #On ne veut pas la supprimer donc on retourne juste le nombre de block à supprimer dans notre algo
        ## if len(to_delete) > 0:
        ##     board = self.remove_row(board, to_delete)
        # return len(to_delete), board

    #TODO
    def get_holes(board):
        pass
        #   num_holes = 0
        #     for col in zip(*board):
        #         row = 0
        #         while row < self.height and col[row] == 0:
        #             row += 1
        #         num_holes += len([x for x in col[row + 1:] if x == 0])
        #     return num_holes

    #TODO
    def get_bumpiness_and_height(board):
        pass
        # board = np.array(board)
        # mask = board != 0
        # invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        # heights = self.height - invert_heights
        # total_height = np.sum(heights)
        # currs = heights[:-1]
        # nexts = heights[1:]
        # diffs = np.abs(currs - nexts)
        # total_bumpiness = np.sum(diffs)
        # return total_bumpiness, total_height


    #TODO méthode qui reset le plateau du tetris
    def reset():

        return self.get_states(self, board)

    def keyPressed(self):
        # for event in list(pygame.event.get()) + tetris_ai.run_ai():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.board.isGameOver = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.board.newBlock.rotate()

                if event.key == pygame.K_DOWN:
                    self.downButton = True

                if event.key == pygame.K_LEFT:
                    self.board.moveBlock("l")

                if event.key == pygame.K_RIGHT:
                    self.board.moveBlock("r")

                if event.key == pygame.K_SPACE:
                    self.board.fallBlockUntilTheEnd()
        
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    self.downButton = False

