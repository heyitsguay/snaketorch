import random

import numpy as np
import pygame
import torch

from collections import deque

from ai_game import AISnakeGame, BLOCK_SIZE, Direction, Point
from helper import plot
from model import LinearQNet, QTrainer

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness control
        self.d_epsilon = 0 # user-specified randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 512, 3)
        self.lr = LR
        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)
        pass
    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        coll_l = game.is_collision(point_l)
        coll_r = game.is_collision(point_r)
        coll_u = game.is_collision(point_u)
        coll_d = game.is_collision(point_d)
        
        # dx0, dx1, dy0, dy1 = game.relative_bbox()
        
        state = [
            # Danger straight
            (dir_r and coll_r) or
            (dir_l and coll_l) or
            (dir_u and coll_u) or
            (dir_d and coll_d),
            
            # Danger right
            (dir_u and coll_r) or
            (dir_d and coll_l) or
            (dir_l and coll_u) or
            (dir_r and coll_d),
            
            # Danger left
            (dir_d and coll_r) or
            (dir_u and coll_l) or
            (dir_r and coll_u) or
            (dir_l and coll_d),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y]
            
            # Time the bbox has been unchanged
            #game.same_bbox_count / 20]
        
        return np.array(state, dtype=float)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        pass
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        return
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        return
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        if self.model.pretrained:
            self.epsilon = np.clip(self.d_epsilon + max(0, 0.002 - self.n_games / 8000), 0, 1)
        else: 
            self.epsilon = np.clip(self.d_epsilon + max(0, 0.4 - self.n_games / 3000), 0, 1)
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_t = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_t)
            move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move
    
def train(w=640, h=480):
    plot_scores = []
    plot_mean_scores = []
    plot_rolling_mean_scores = []
    total_score = 0
    record = 0
    
    agent = Agent()
    game = AISnakeGame(agent, w, h)
    
    
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # train long memory (experience replay)
            game.reset()
            agent.n_games += 1
            if agent.n_games % 3000 == 2999:
                agent.lr *- 0.5
                agent.trainer.update_lr(agent.lr)
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print(f' Game {agent.n_games}: Score (Record) {score} {record}')
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_rolling_mean_scores.append(rolling_mean(plot_scores, 50))
            plot(plot_scores, plot_mean_scores, plot_rolling_mean_scores)
    return


def rolling_mean(x, n):
    if len(x) <= n:
        return sum(x) / len(x)
    else:
        return sum(x[-n:]) / n


if __name__ == '__main__':
    train()
