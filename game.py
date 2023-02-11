import pygame
from numpy import random
from collections import namedtuple
import time
#from agente import Agent 
pygame.init()
font = pygame.font.Font('arial.ttf', 25)
Point = namedtuple('Point','x,y')

BLOCK_SIZE = 110
SPEED = 1200
width = 800
height = 420
padd_w = 20
dir = True
reward = 0


class PongGame:

    def __init__(self):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Pong')
        self.clock = pygame.time.Clock()
        self.ball = Ball()
        self.paddle = Paddle()
        self.reset()
        global  dir
        dir = bool(random.rand(0,1))
        self.bound_m =0 
        self.game_over = False
        
        
    def reset(self):
        self.ball.reset()
        self.paddle.reset()
        self.score = 0
        self.game_over = False

    def step(self,action): #action
        global dir
        global reward
        if action == [1]:
              dir = True
        elif action ==[0]:
             dir = False
                            
        self.paddle.move(dir)
        self.ball.move()
        score, game_over = self.update()
       
        
        if score > self.bound_m:
            reward = 10
            self.bound_m = score
        self.clock.tick(SPEED)
        #state = self.get_state()
        #reward = self.get_reward()
        done = game_over
        if done:
            reward = -10
        
        return reward, done, score

    def update(self):
        global padd_w
        
        self.display.fill((0,0,0))
        
        for pt in self.paddle.shape:
            pygame.draw.rect(self.display, (0,150,255),pygame.Rect(self.paddle.x,self.paddle.y, padd_w,BLOCK_SIZE))
        
        pygame.draw.circle(self.display,(0,100,255),(self.ball.x,self.ball.y),12)
        
        if not self.ball.fall():
            self.score = self.ball.bounce(self.paddle.collision(self.ball))
        else:
            self.game_over = True
            
            
        text = font.render(f"Score:{self.score}, X:{self.ball.x}, Y:{self.ball.y},P:{self.paddle.y}".format(self.score,self.ball.x,self.ball.y,self.paddle.y),True,(0,255,0))
        self.display.blit(text,[0,0])
        pygame.display.flip()
        #time.sleep(500)
        
        return self.score, self.game_over
    
    def get_st(self):
        return self.ball.x, self.ball.y, self.paddle.y

    def reset(self):
        self.ball.reset()
        self.paddle.reset()
        self.game_over = False
        

class Ball:

    def __init__(self):
        self.x = random.rand(0,150)
        self.y = random.rand(400)
        self.w = width
        self.h = height
        self.speed_x = 18
        self.speed_y = 23
        self.score = 0
        #self.pos_ball = Point 

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y

    def bounce(self, collision):
        if self.x <= 0:
            self.speed_x = -self.speed_x
        if self.y <=0 or self.y >= self.h:
            self.speed_y = -self.speed_y
        if collision:
            self.speed_x = -self.speed_x
            self.score += 1
        return self.score
    
    def fall(self):
        if self.x >= self.w+BLOCK_SIZE:
            return True
        else:
            return False
        
    def reset(self):
        self.w = width
        self.h = height
        self.x = random.randint(0,self.w/4) #0
        self.y = random.randint(0,self.h) #300
        self.score = 0

class Paddle:

    def __init__(self):
        self.w = width
        self.h = height
        self.x = width-BLOCK_SIZE
        self.y = 350
        global dir
        
        self.shape = [Point(self.x,self.y)]
        self.speed = 20

    def move(self,dir):
        if(self.y <= 0):
            self.y = 0+BLOCK_SIZE
        if(self.y+BLOCK_SIZE >= self.h):
            self.y = self.h-BLOCK_SIZE
            
        if dir:
            self.y -= self.speed
        else:
            self.y += self.speed
 
        
    def collision(self, ball):
        if self.y <= ball.y <= self.y + BLOCK_SIZE  and self.x - padd_w <= ball.x <= self.x:
            return True
        return False
    
    def reset(self):
        self.w = width
        self.h = height
        self.x = width-30
        self.y = random.randint(0,self.w)


if __name__ == '__main__':
    pong = PongGame()
    
    while True:
        done = pong.step()
        
        if done == True:
            print("Game Over")
            break
    
    pygame.quit()