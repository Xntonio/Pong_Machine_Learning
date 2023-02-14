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
        self.gamma = 0.2 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(3, 64, 1)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        game = PongGame()


    def get_state(self, game):
        ball_x, ball_y, paddle_y = game.get_st()
        state = [ball_x, ball_y, paddle_y]
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

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        #self.epsilon = 100 - self.n_games
        #movimientos random/ exploracion y explotacion
        self.epsion=max(0.01,0.9-0.05*self.n_games)

        final_move = [0]

        if random.uniform(0, 1) < self.epsilon:
            move = 0
            final_move[move] = random.randint(0, 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0).detach().numpy()
            action = np.argmax(prediction)
            final_move[0] = action
        return final_move
        
        #if random.randint(0, 100) < self.epsilon:
        #    move = 0
        #    final_move[move] = random.randint(0, 1)
        #    #print("ENTTRE 1")
        #else:
        #    state0 = torch.tensor(state, dtype=torch.float)
        #    prediction = self.model(state0)
        #    if prediction[0] < prediction[1]:
        #        fm=1
        #        print("Prediction",prediction)
        #    else:
        #        fm=1
        #        print("Prediction",prediction)

        #    final_move[0] = fm
           

        # if final_move[0] == 0:
        #     print("UP")
        # if final_move[0] == 1:
        #     print("DOWN")            
        #return final_move


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

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, game_over, score = game.step(final_move)
        #reward = score*10
        print("REWARD:",reward,game_over,score)
        
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