import pygame
from math import pi, sin, cos

from numpy import random
from collections import namedtuple
import time
from random import uniform

pygame.init()

font = pygame.font.Font('arial.ttf', 25)
Point = namedtuple('Point','x,y')

HEIGHT = 420
WIDTH = 800

PADD_H= 200
PADD_W = 20
PADD_V=1

PADD_X=WIDTH-PADD_W
PADD_Y=HEIGHT / 2

#direccion de la pelota
ball_dir_x=1
ball_dir_y=1

BALL_SIZE=20
SCORE=0
GAME_OVER=0

##
dir = True
reward = 0
SPEED=12000
##

class PongGame:
    def __init__(self):
        self.height = HEIGHT
        self.width = WIDTH
        
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Pong')
        self.ball = Ball()
        self.paddle = Paddle()
        self.clock = pygame.time.Clock()
        self.reset()
  
        self.game_over = False
        
        
    def reset(self):
        self.ball.reset()
        self.paddle.reset()
        self.score = 0
        self.game_over = False

    def step(self, action): #action

        if action == [1]:
              dir = True
        elif action ==[0]:
            dir = False
        
        self.paddle.move(dir)
        self.ball.move()
        
        score, game_over=self.update()
        self.clock.tick(SPEED)
        reward=0

        if score > 0:
            reward=10
        done=game_over
        if done:
            reward=-10

        #self.update()
        return reward,done,score

    def update(self):

        self.display.fill((0,0,0))
        
        pygame.draw.rect(self.display, (255,255,255),(self.paddle.padd_x,self.paddle.padd_y, PADD_W,PADD_H))
        
        pygame.draw.circle(self.display,(255,255,255),(self.ball.ball_x,self.ball.ball_y), radius=10)
        
        if not self.ball.fall():
            self.score = self.ball.bounce(self.paddle.collision(self.ball))
        else:
            self.game_over = True
            
            
        text = font.render(f"Score:{self.score}, X:{self.ball.ball_x}, Y:{self.ball.ball_y},P:{self.paddle.padd_y}".format(self.score,self.ball.ball_x,self.ball.ball_y,self.paddle.padd_y),True,(0,255,0))
        self.display.blit(text,[0,0])
        pygame.display.flip()

        return self.score, self.game_over
    
    def get_st(self):
        #print(self.ball.ball_x, self.ball.ball_y, self.paddle.padd_y)
        return self.ball.ball_x, self.ball.ball_y, self.paddle.padd_y

    def reset(self):
        self.ball.reset()
        self.paddle.reset()
        self.game_over = False
        

class Ball:

    def __init__(self):
        self.ball_x = random.randint(0,150)
        self.ball_y = random.randint(400)
        self.width = WIDTH
        self.height = HEIGHT
        #self.direction = uniform(5 * pi / 6, 7 * pi / 6)
        #self.ball_dir_x = int(2 * cos(self.direction))
        #self.ball_dir_y = int(2 * sin(self.direction))

        self.ball_dir_x=1
        self.ball_dir_y=1

        self.score = 0


    def move(self):
        self.ball_x += self.ball_dir_x
        self.ball_y += self.ball_dir_y

    def bounce(self, collision):
        global padd_h
        # Verificar si la pelota ha tocado la raqueta
        if collision:
            self.score += 1
            self.ball_dir_x = -self.ball_dir_x

        # Verificar si la pelota ha tocado las paredes
        if self.ball_y <= 0 or self.ball_y >= self.height:
            self.ball_dir_y = -self.ball_dir_y

        if self.ball_x <= 0:
            self.ball_dir_x=-self.ball_dir_x
        return self.score
    
    def fall(self):
        if self.ball_x >= self.w:
            return True
        else:
            return False
        
    def reset(self):
        self.w = self.width
        self.h = self.height
        self.ball_x = random.randint(0,self.w/4) #0
        self.ball_y = random.randint(0,self.h) #300
        self.score = 0

class Paddle:

    def __init__(self):
        self.padd_h=PADD_H
        self.padd_w=PADD_W
        self.width= WIDTH
        self.height= HEIGHT
        global dir

        #coordenadas de la raqueta
        self.padd_x=self.width - self.padd_w
        self.padd_y=self.height - self.padd_h        
        self.pad_speed = 1

    def move(self,dir):

        if(self.padd_y <= 0):
            self.padd_y = 0
        if(self.padd_y >= self.height - self.padd_h):
            self.padd_y = self.height - self.padd_h
            
        if dir:
            self.padd_y -= self.pad_speed
        else:
            self.padd_y += self.pad_speed
 
        
    def collision(self, ball):

        if ball.ball_y + 10 > self.padd_y and ball.ball_y < self.padd_y+self.padd_h:
            if ball.ball_x > self.padd_x - self.padd_w//2 and ball.ball_x < self.padd_x+self.padd_w//2:
                return True
        return False

    def reset(self):
        self.padd_x = self.width-30
        self.padd_y = random.randint(0,self.width)


if __name__ == '__main__':
    pong = PongGame()
    while True:
        done = pong.step()
        
        if done == True:
            print("Game Over")
            break
    
    pygame.quit()