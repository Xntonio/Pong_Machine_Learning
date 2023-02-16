import torch
import random
import numpy as np
from collections import deque
from game import PongGame, Ball, Paddle
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate

        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(6, 256, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        #game = PongGame()


    def get_state(self, game):
        ball_x, ball_y, paddle_y, direccion = game.get_st()
        #print("dir",dir)
        #print(ball_x,ball_y)
        dir_u=direccion == True
        dir_d=direccion == False
        #print("dir",dir_u) 
        #dir_d=game.dir ==
        #print(direccion)
        paddlg=Paddle()
        #print("hola",paddlg.collision(game.ball))
        #print("ccc",game.ball.ball_x,game.ball.ball_y)
        
        state = [
                (dir_u and paddlg.collision(game.ball)), #colision
                (dir_d and paddlg.collision(game.ball)), #colision
                dir_u, #direccion
                dir_d, #direccion
                ball_y<paddle_y, #la bola esta abajo
                ball_y>paddle_y, #la bola esta abajo
                ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
        #for state, action, reward, nexrt_state, game_over in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, game_over)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)
        #print(" - -short memory - - ")
        #print("state ", state)
        #print("action", action)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        #movimientos random/ exploracion y explotacion
        #self.epsilon=max(0.01,0.9-0.05*self.n_games)

        self.epsilon = 80 - self.n_games
        final_move = 0

        if random.uniform(0, 200) < self.epsilon:
            final_move = random.randint(0, 1)
            #print("aleatorio:", final_move)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) #.detach().numpy()
            #print("prediction",prediction)
            action = torch.argmax(prediction).item()
            max_value,max_index=torch.max(prediction,dim=0)
            if max_index == 0:
                final_move = 1  # acción 1
                print("true")
            else:
                final_move = 0  # acción 2

            print("prediction", prediction)
            #print("action", action)
            #print("action", action)
            #final_move = action
            #print("random:", final_move)

        return final_move
  

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = PongGame()
    while True:
        # get old state
        state_old = agent.get_state(game)
        #print("state viejo",state_old)
        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, game_over, score = game.step(final_move)
        #reward = score*10
        #print("REWARD:",reward,game_over,score)
        
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            # print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()