import pygame

import random
import os

import neat

import pickle

import math

pygame.font.init()

WIN_SIZE = 600
NUMBER_OF_SQUARES = 5
NUMBER_OF_SQUARES_SQUARED = NUMBER_OF_SQUARES**2
SQUARE_SIZE = WIN_SIZE//NUMBER_OF_SQUARES
FONT = pygame.font.SysFont('comicsansms',30)
FONT_TRAINING = pygame.font.SysFont('comicsansms',90)

WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0,255,0)
RED = (255,0,0)

win = pygame.display.set_mode((WIN_SIZE,WIN_SIZE))
pygame.display.set_caption("SnakeAI")

class snake_body:
    def __init__(self, x, y, moving_side):
        self.x = x
        self.y = y
        self.moving_side = moving_side

    def draw_head(self,size):
        # draws the head and eyes of the snake based on the current moving direction
        pygame.draw.rect(win, (0, 200, 0), (self.x*size, self.y*size, size, size))
        if self.moving_side == 'up':
            pygame.draw.circle(win, (255, 255, 255), (self.x*size + size // 4, self.y*size + size // 4),size // 4)
            pygame.draw.circle(win, (0, 0, 0), (self.x*size + size // 4, self.y*size + size // 4), size // 8)
            pygame.draw.circle(win, (255, 255, 255), (self.x*size + size - size // 4, self.y*size + size // 4),size // 4)
            pygame.draw.circle(win, (0, 0, 0), (self.x*size + size - size // 4, self.y*size + size // 4),size // 8)
        if self.moving_side == 'down':
            pygame.draw.circle(win, (255, 255, 255), (self.x*size + size // 4, self.y*size + size - size // 4),size // 4)
            pygame.draw.circle(win, (0, 0, 0), (self.x*size + size // 4, self.y*size + size - size // 4),size // 8)
            pygame.draw.circle(win, (255, 255, 255),(self.x*size + size - size // 4, self.y*size + size - size // 4),size // 4)
            pygame.draw.circle(win, (0, 0, 0),(self.x*size + size - size // 4, self.y*size + size - size // 4),size // 8)
        if self.moving_side == 'left':
            pygame.draw.circle(win, (255, 255, 255), (self.x*size + size // 4, self.y*size + size - size // 4),size // 4)
            pygame.draw.circle(win, (0, 0, 0), (self.x*size + size // 4, self.y*size + size - size // 4),size // 8)
            pygame.draw.circle(win, (255, 255, 255), (self.x*size + size // 4, self.y*size + size // 4),size // 4)
            pygame.draw.circle(win, (0, 0, 0), (self.x*size + size // 4, self.y*size + size // 4), size // 8)
        if self.moving_side == 'right':
            pygame.draw.circle(win, (255, 255, 255), (self.x*size + size - size // 4, self.y*size + size // 4),size // 4)
            pygame.draw.circle(win, (0, 0, 0), (self.x*size + size - size // 4, self.y*size + size // 4),size // 8)
            pygame.draw.circle(win, (255, 255, 255),(self.x*size + size - size // 4, self.y*size + size - size // 4),size // 4)
            pygame.draw.circle(win, (0, 0, 0),(self.x*size + size - size // 4, self.y*size + size - size // 4),size // 8)

    def draw_body(self,size):
        pygame.draw.rect(win, GREEN, (self.x * size, self.y * size, size, size))

class Snake:
    def __init__(self,x,y,size):
        self.size = size
        self.x = x
        self.y = y
        self.head = snake_body(x,y,'down')
        self.array = []
        self.array.append(self.head)
        self.array.append(snake_body(x,y-1,'down'))
        self.array.append(snake_body(x, y - 2, 'down'))

    def move(self):
        for i in range(len(self.array) - 1, -1, -1):  # moves  the snake starting from tail
            if self.array[i].moving_side == 'up':
                self.array[i].y -= 1
            if self.array[i].moving_side == 'down':
                self.array[i].y += 1
            if self.array[i].moving_side == 'left':
                self.array[i].x -= 1
            if self.array[i].moving_side == 'right':
                self.array[i].x += 1

            if i > 0 and self.array[i - 1].moving_side != self.array[i].moving_side:  # If the direction of 'snake body' in front of isnt the same, changes it to
                self.array[i].moving_side = self.array[i - 1].moving_side

    def draw(self,win):
        for snake_body in reversed(self.array):
            if snake_body == self.head:
                snake_body.draw_head(self.size)
            else:
                snake_body.draw_body(self.size)

    def increase(self):
        #adds the snake body at the end of snake
        if self.array[-1].moving_side == 'up':
            self.array.append(snake_body(self.array[-1].x,self.array[-1].y+1,self.array[-1].moving_side))
        if self.array[-1].moving_side == 'down':
            self.array.append(snake_body(self.array[-1].x,self.array[-1].y-1,self.array[-1].moving_side))
        if self.array[-1].moving_side == 'left':
            self.array.append(snake_body(self.array[-1].x+1,self.array[-1].y,self.array[-1].moving_side))
        if self.array[-1].moving_side == 'right':
            self.array.append(snake_body(self.array[-1].x-1,self.array[-1].y,self.array[-1].moving_side))

    def check_collision(self,num_of_squares):
        if self.head.y>num_of_squares-1 or self.head.x > num_of_squares-1 or self.head.x<0 or self.head.y<0:
            return True
        for body in self.array:
            if self.head.x == body.x and self.head.y == body.y and self.head!=body:
                return True
        return False

    def getAngle(self,apple):
        #angle of the snake head and apple
        angle = math.atan2((self.head.x-apple.x),(self.head.y-apple.y))
        return angle

    def getSideDistance(self,numSquares):
        #maximum distance that snake can go to without loosing
        distance = [0,0,0,0,0,0,0,0] #up,down,left,right,up-right,up-left,down-right,down-left
        bodyInWay = False
        if self.head.moving_side !='down':
            for i in reversed(range(self.head.y)):
                for body in self.array:
                    if body.x == self.head.x and body.y == i:
                        bodyInWay = True
                if not bodyInWay:
                    distance[0]+=1
                else:
                    break
        bodyInWay = False
        if self.head.moving_side != 'up':
            for i in range(self.head.y+1,numSquares):
                for body in self.array:
                    if body.x == self.head.x and body.y == i:
                        bodyInWay = True
                if not bodyInWay:
                    distance[1] += 1
                else:
                    break
        bodyInWay = False
        if self.head.moving_side != 'right':
            for i in reversed(range(self.head.x)):
                for body in self.array:
                    if body.y == self.head.y and body.x == i:
                        bodyInWay = True
                if not bodyInWay:
                    distance[2]+=1
                else:
                    break
        bodyInWay = False
        if self.head.moving_side != 'left':
            for i in range(self.head.x + 1, numSquares):
                for body in self.array:
                    if body.y == self.head.y and body.x == i:
                        bodyInWay = True
                if not bodyInWay:
                    distance[3] += 1
                else:
                    break
        distance[4:]=[1,1,1,1]
        for body in self.array:
            #up - left,up-right, down - right, down - left
            if body.x == self.head.x-1 and body.y ==self.head.y-1 or self.head.x -1<0 or self.head.y-1<0:
                distance[4]=0
            if body.x == self.head.x+1 and body.y ==self.head.y-1 or self.head.x+1 >=numSquares or self.head.y-1<0:
                distance[5]=0
            if body.x == self.head.x+1 and body.y ==self.head.y+1 or self.head.x+1 >=numSquares or self.head.y+1>=numSquares:
                distance[6]=0
            if body.x == self.head.x-1 and body.y ==self.head.y+1 or self.head.x-1<0 or self.head.y+1>=numSquares:
                distance[7]=0

        #print(distance)
        return distance

    def getDistanceFromApple(self,apple):
        return math.sqrt((apple.x-self.x)**2+(apple.y-self.y)**2)



class Apple:
    def __init__(self,x,y,size,num_of_squares,snake):
        self.x = x
        self.y = y
        self.size = size
        self.num_of_squares =num_of_squares
        self.snake = snake

    def draw(self,win):
        if len(self.snake.array) < self.num_of_squares ** 2:
            pygame.draw.rect(win, RED, (self.x * self.size, self.y * self.size, self.size, self.size))

    def generate(self):
        #generates apple at random position
        if len(self.snake.array) < self.num_of_squares**2:
            self.x = random.randint(0, self.num_of_squares-1)
            self.y = random.randint(0, self.num_of_squares-1)

            # Checks if spawned apple is on the snake
            for body in self.snake.array:
                if self.x == body.x and self.y == body.y:
                    self.generate()

def draw_distance(head,distance,apple):
    #draws the distance that snake can go to
    if distance[0]>0:
        pygame.draw.rect(win,(255,255,255),(head.x*SQUARE_SIZE,(head.y-distance[0])*SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE*distance[0]))
    if distance[1]>0:
        pygame.draw.rect(win,(255,255,255),(head.x*SQUARE_SIZE,(head.y+1)*SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE*distance[1]))
    if distance[2]>0:
        pygame.draw.rect(win,(255,255,255),((head.x-distance[2])*SQUARE_SIZE,head.y*SQUARE_SIZE,SQUARE_SIZE*distance[2],SQUARE_SIZE))
    if distance[3]>0:
        pygame.draw.rect(win,(255,255,255),((head.x+1)*SQUARE_SIZE,head.y*SQUARE_SIZE,SQUARE_SIZE*distance[3],SQUARE_SIZE))
    if distance[4]>0:
        pygame.draw.rect(win,(255,255,255),((head.x-1)*SQUARE_SIZE,(head.y-1)*SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE))
    if distance[5]>0:
        pygame.draw.rect(win,(255,255,255),((head.x+1)*SQUARE_SIZE,(head.y-1)*SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE))
    if distance[6]>0:
        pygame.draw.rect(win,(255,255,255),((head.x+1)*SQUARE_SIZE,(head.y+1)*SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE))
    if distance[7]>0:
        pygame.draw.rect(win,(255,255,255),((head.x-1)*SQUARE_SIZE,(head.y+1)*SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE))

    pygame.draw.line(win,(0,0,255),(head.x*SQUARE_SIZE,head.y*SQUARE_SIZE),(apple.x*SQUARE_SIZE,apple.y*SQUARE_SIZE),5)

def draw_window(snake,draw_vision,distance,apple,gen,max_score):
    win.fill(BLACK)
    snake.draw(win)
    if draw_vision:
        draw_distance(snake.head, distance, apple)
    apple.draw(win)
    draw_score(gen, max_score)
    pygame.display.update()


def draw_score(gen,score):
    text = FONT.render("Score: "+str(score),1,(255,255,255))
    win.blit(text,(WIN_SIZE - 5 - text.get_width(),5))
    text = FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (10, 5))

def draw_training_window():
    #draws the Training screen, increases training speed because the program doesnt need
    win.fill(BLACK)
    text = FONT_TRAINING.render("Training...", 1, (255, 255, 255))
    win.blit(text, (WIN_SIZE//2-text.get_width()//2,WIN_SIZE//2-text.get_height()//2))
    pygame.display.update()

def moving_side_to_int(side):
    if side == 'up':
        return 0
    if side == 'down':
        return 1
    if side == 'left':
        return 2
    if side == 'right':
        return 3

def set_move(y, snake):
    m = -math.inf
    side = snake.head.moving_side
    if y[0]>m and snake.head.moving_side!='down':
        side = 'up'
        m = y[0]
    if y[1]>m and snake.head.moving_side!='up':
        side = 'down'
        m=y[1]
    if y[2]>m and snake.head.moving_side!='right':
        side = 'left'
        m=y[2]
    if y[3]>m and snake.head.moving_side!='left':
        side = 'right'
        m=y[3]

    snake.head.moving_side = side


def game():
    draw_vision = False

    snake = Snake(NUMBER_OF_SQUARES//2, NUMBER_OF_SQUARES//2, SQUARE_SIZE)
    apple = Apple(1, 1, SQUARE_SIZE, NUMBER_OF_SQUARES, snake)
    apple.generate()

    clock = pygame.time.Clock()
    score = 0

    pause = False

    run = True
    while run:
        clock.tick(5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    draw_vision = not draw_vision
                if event.key == pygame.K_w:
                    if snake.head.moving_side != 'down':
                        snake.head.moving_side = 'up'
                if event.key == pygame.K_s:
                    if snake.head.moving_side != 'up':
                        snake.head.moving_side = 'down'
                if event.key == pygame.K_a:
                    if snake.head.moving_side != 'right':
                        snake.head.moving_side = 'left'
                if event.key == pygame.K_d:
                    if snake.head.moving_side != 'left':
                        snake.head.moving_side = 'right'
                if event.key == pygame.K_p:
                    pause = not pause

        if not pause:
            snake.move()
            m=get_matrix(snake,apple)
            for i in range(NUMBER_OF_SQUARES):
                for j in range(NUMBER_OF_SQUARES):
                    print(m[j][i],end=' ')
                print()
            distance = snake.getSideDistance(NUMBER_OF_SQUARES)
            angle = snake.getAngle(apple)
            dist_from_apple = snake.getDistanceFromApple(apple)

            print(distance)


            if snake.check_collision(NUMBER_OF_SQUARES):
                print("You lost! Score:",score)
                #run = False


            if snake.head.x == apple.x and snake.head.y == apple.y:
                score += 1
                snake.increase()
                apple.generate()

            draw_window(snake, draw_vision, distance, apple, "Human", score)

def get_matrix(snake,apple):
    matrix = [[0 for i in range(NUMBER_OF_SQUARES)] for j in range(NUMBER_OF_SQUARES)]
    matrix[apple.x][apple.y] = 3
    for body in snake.array:
        if body.x<NUMBER_OF_SQUARES and body.x>=0 and body.y < NUMBER_OF_SQUARES and body.y>=0:
            matrix[body.x][body.y] = 1
    if snake.head.x < NUMBER_OF_SQUARES and snake.head.x >= 0 and snake.head.y < NUMBER_OF_SQUARES and snake.head.y >= 0:
        matrix[snake.head.x][snake.head.y] = 2

    return matrix

gen = 0
max_score = 0
draw_win = False
normal_speed = False
draw_vision = False
def eval_genomes(genomes, config):
    global gen,max_score,draw_win,normal_speed,draw_vision
    gen+=1

    for id, genome in genomes:
        num_of_moves = 0
        net =  neat.nn.FeedForwardNetwork.create(genome,config)
        genome.fitness = 0
        score = 0

        snake = Snake(NUMBER_OF_SQUARES//2, NUMBER_OF_SQUARES//2, SQUARE_SIZE)
        apple = Apple(0, 0, SQUARE_SIZE, NUMBER_OF_SQUARES,snake)
        apple.generate()

        clock = pygame.time.Clock()
        run = True
        while run:
            if normal_speed:
                clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_o:
                        draw_vision = not draw_vision
                    if event.key == pygame.K_p:
                        draw_win = not draw_win
                        if draw_win== False:
                            draw_training_window()
                    if event.key == pygame.K_n and draw_win:
                        normal_speed = not normal_speed
            
            snake.move()
            num_of_moves += 1
            m =  get_matrix(snake,apple)

            if snake.check_collision(NUMBER_OF_SQUARES):
                genome.fitness -= 6
                run = False

            if num_of_moves>NUMBER_OF_SQUARES_SQUARED:
                genome.fitness -= 6
                run = False

            if snake.head.x == apple.x and snake.head.y == apple.y:
                genome.fitness += 6
                score+=1
                if score> max_score:
                    max_score = score
                    pickle.dump(genome, open("ai/"+str(max_score)+"score.pkl1", "wb"))

                snake.increase()
                apple.generate()
                num_of_moves = 0

            if score == NUMBER_OF_SQUARES_SQUARED-3:
                genome.fitness+=10
                break


            distance = snake.getSideDistance(NUMBER_OF_SQUARES)

            angle = snake.getAngle(apple)
            dist_from_apple = snake.getDistanceFromApple(apple)
                                    #X,Y and moving side of the snake
            '''
            output = net.activate((snake.head.x,snake.head.y,moving_side_to_int(snake.head.moving_side),
                                   #down, up, left,right, distance
                                   distance[0],distance[1],distance[2],distance[3],
                                   #can move diagonals
                                   distance[4],distance[5],distance[6],distance[7],
                                   #angle and distance from apple
                                   angle,dist_from_apple,apple.x,apple.y))

            '''
            output = net.activate((snake.head.x,snake.head.y,moving_side_to_int(snake.head.moving_side),
                                   #down, up, left,right, distance
                                   distance[0],distance[1],distance[2],distance[3],
                                   #can move diagonals
                                   distance[4],distance[5],distance[6],distance[7],
                                   #angle and distance from apple
                                   angle,dist_from_apple,apple.x,apple.y,
                                   m[0][0],m[1][0],m[2][0],m[3][0],m[4][0],#m[5][0],m[6][0],m[7][0],m[8][0],m[9][0],
                                   m[0][1],m[1][1],m[2][1],m[3][1],m[4][1],#m[5][1],m[6][1],m[7][1],m[8][1],m[9][1],
                                   m[0][2],m[1][2],m[2][2],m[3][2],m[4][2],#m[5][2],m[6][2],m[7][2],m[8][2],m[9][2],
                                   m[0][3],m[1][3],m[2][3],m[3][3],m[4][3],#m[5][3],m[6][3],m[7][3],m[8][3],m[9][3],
                                   m[0][4],m[1][4],m[2][4],m[3][4],m[4][4],#m[5][4],m[6][4],m[7][4],m[8][4],m[9][4],
                                   #m[0][5],m[1][5],m[2][5],m[3][5],m[4][5],m[5][5],m[6][5],m[7][5],m[8][5],m[9][5],
                                   #m[0][6],m[1][6],m[2][6],m[3][6],m[4][6],m[5][6],m[6][6],m[7][6],m[8][6],m[9][6],
                                   #m[0][7],m[1][7],m[2][7],m[3][7],m[4][7],m[5][7],m[6][7],m[7][7],m[8][7],m[9][7],
                                   #m[0][8],m[1][8],m[2][8],m[3][8],m[4][8],m[5][8],m[6][8],m[7][8],m[8][8],m[9][8],
                                   #m[0][9],m[1][9],m[2][9],m[3][9],m[4][9],m[5][9],m[6][9],m[7][9],m[8][9],m[9][9]
            ))

            set_move(output,snake)

            if draw_win:
                draw_window(snake,draw_vision,distance,apple,gen,max_score)

def run_ai(config_path):
    global normal_speed, draw_win
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    with open("winner.pkl", "rb") as f:
        genome = pickle.load(f)
    print(type(genome))
    genomes = [(1,genome)]
    normal_speed = True
    draw_win = True
    eval_genomes(genomes,config)


def run(config_path):
    config  = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes,1000)
    print(winner)
    pickle.dump(winner,open("winner.pkl","wb"))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config-file.txt")
    #run(config_path)
    game()
    #run_ai(config_path)