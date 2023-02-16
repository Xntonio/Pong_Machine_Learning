import pygame
from math import pi, sin, cos

from numpy import random
from collections import namedtuple
import time
from random import uniform

pygame.init()

font = pygame.font.Font('arial.ttf', 22)
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
GAME_OVER=0

##
DIRECCION = True
reward = 0
SPEED=400
##

class PongGame:
    def __init__(self):
        self.height = HEIGHT
        self.width = WIDTH
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Pong')
        self.direccion=DIRECCION
        self.ball = Ball()
        self.paddle = Paddle()

        #self.direccion=self.dir

        self.time = time.time()
        self.clock = pygame.time.Clock()
        self.max_score = 0
        self.sim = 0
        self.reset()
        self.game_over = False
        

    def step(self, action): #action
        #action from agent
        if action == 1:
            self.direccion=True  
            dir = True
        elif action ==0:
            self.direccion=False
            dir = False
        
        #move ball and paddle

        self.paddle.move(action)
        self.ball.move()
        
        #check if gameover
        reward = 0
        game_over = False
        
        if self.ball.fall():
            game_over = True
            reward = -10
            self.sim += 1
            return reward, game_over, self.score
        
        if self.paddle.collision(self.ball):
            self.score += 1
            self.max_score = self.score if self.score >= self.max_score else self.max_score
            reward = 10
        
        #update ui and clock    
        self.update()
        self.clock.tick(SPEED)

        #self.update()
        return reward,game_over, self.score

    def update(self):
        self.display.fill((0,0,0))
        
        pygame.draw.rect(self.display, (255,255,255),(self.paddle.padd_x,self.paddle.padd_y, PADD_W,PADD_H))
        
        pygame.draw.circle(self.display,(255,255,255),(self.ball.ball_x,self.ball.ball_y), radius=10)
        self.time_final = int(time.time()-self.time)
        text = font.render(" Score: " + str(self.score)+ "    |   ScoreMax: " +str(self.max_score)+ "   Simulation: " +str(self.sim)+ "   Time: " +str(self.time_final) , True, (255,255,255))
        self.display.blit(text,[0,0])
        pygame.display.flip()
    
    def get_st(self):
        #print("ball x", self.ball.ball_x)
        #print("ball y", self.ball.ball_y)
        #print("ball p", self.paddle.padd_y)
        return self.ball.ball_x, self.ball.ball_y, (self.paddle.padd_y+(self.paddle.padd_h/2)), self.direccion

    def reset(self):
        self.ball.reset()
        self.paddle.reset()
        self.score = 0
        self.game_over = False
        

class Ball:

    def __init__(self):
        self.ball_x = random.randint(0,150)
        self.ball_x=1
        self.ball_y = random.randint(400)
        self.width = WIDTH
        self.height = HEIGHT
        self.vel = 10
        #self.direction = uniform(5 * pi / 6, 7 * pi / 6)
        #self.ball_dir_x = int(2 * cos(self.direction))
        #self.ball_dir_y = int(2 * sin(self.direction))
        self.ball_dir_x=1
        self.ball_dir_y=1


    def move(self):
        self.ball_x += self.ball_dir_x*self.vel
        self.ball_y += self.ball_dir_y*self.vel
        
        # Verificar si la pelota ha tocado las paredes
        if self.ball_y <= 0 or self.ball_y >= self.height:
            self.ball_dir_y = -self.ball_dir_y

        if self.ball_x <= 0:
            self.ball_dir_x=-self.ball_dir_x
        
        self.headB=Point(self.ball_dir_x,self.ball_dir_y)

    
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

class Paddle:

    def __init__(self):
        self.padd_h=PADD_H
        self.padd_w=PADD_W
        self.width= WIDTH
        self.height= HEIGHT
        global direccion

        #coordenadas de la raqueta
        self.padd_x=self.width - self.padd_w
        self.padd_y=self.height - self.padd_h        
        self.pad_speed = 10

    def move(self,dir):

        if(self.padd_y <= 0):
            self.padd_y = 0
            #print("padd_y",self.padd_y)
        if(self.padd_y >= self.height - self.padd_h):
            self.padd_y = self.height - self.padd_h
            #print("padd_y",self.padd_y)

        #print(direccion,"direccion")

        self.direccion=dir
        
        if dir:
            self.padd_y -= self.pad_speed
        else:
            self.padd_y += self.pad_speed

        self.headP=Point(self.padd_x,self.padd_y)


 
        
    def collision(self, ball):

        if ball.ball_y + 10 > self.padd_y and ball.ball_y < self.padd_y+self.padd_h:
            if ball.ball_x > self.padd_x - self.padd_w//2 and ball.ball_x < self.padd_x+self.padd_w//2:
                ball.ball_x = ball.ball_x-2
                ball.ball_dir_x = -ball.ball_dir_x
                return True
        return False

    def reset(self):
        self.padd_y = self.width-30
        #self.padd_y = random.randint(0,self.width)


if __name__ == '__main__':
    pong = PongGame()
    while True:
        game_over = pong.step()
        
        if game_over == True:
            print("Game Over")
            break
    
    pygame.quit()