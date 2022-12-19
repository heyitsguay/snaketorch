import colorsys
import random

import numpy as np
import pygame

from collections import namedtuple
from enum import Enum

# reset
# reward
# play(action) -> direction
# game_iteration
# is_collision change


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
CLOCKWISE = [
    Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
N_STATES = len(CLOCKWISE)
    
Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLUE3 = (0, 50, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
COLOR_SCALE = 60
SPEEDS = [20, 60, 1000]


class AISnakeGame:
    
    def __init__(self, agent, w=640, h=480):
        pygame.init()
        self.w = w
        self.h = h
        self.agent = agent
        self.display = pygame.display.set_mode((self.w, self.h))
        self.font = pygame.font.Font('arial.ttf', 25)
        pygame.display.set_caption('SnakeTorch')
        self.clock = pygame.time.Clock()
        self.t_start = 0
        self.idx = None
        self.speed = 20
        self.speed_idx = 0
        self.last_bbox = None
        self.last_reward = None
        self.same_bbox_count = 0
        self.show_text = True
        self.reset()
        
    def relative_bbox(self):
        xh = self.snake[0].x
        yh = self.snake[0].y
        x0, x1, y0, y1 = self.last_bbox
        dx0 = int(xh - x0)
        dx1 = int(xh - x1)
        dy0 = int(yh - y0)
        dy1 = int(yh - y1)
        return dx0, dx1, dy0, dy1
        
    def _snake_bbox(self):
        xs = [p.x + BLOCK_SIZE // 2 for p in self.snake[1:]]
        ys = [p.y + BLOCK_SIZE // 2 for p in self.snake[1:]]
        x0 = min(xs)
        x1 = max(xs)
        y0 = min(ys)
        y1 = max(ys)
        return np.array((x0, x1, y0, y1), dtype=np.int)
        
    def play_step(self, action):
        self.frame_iteration += 1
        # Get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFTBRACKET:
                    self.speed_idx = (self.speed_idx - 1) % len(SPEEDS)
                    self.speed = SPEEDS[self.speed_idx]
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.speed_idx = (self.speed_idx + 1) % len(SPEEDS)
                    self.speed = SPEEDS[self.speed_idx]
                elif event.key == pygame.K_q:
                    pygame.quit()
                    quit()
                elif event.key == pygame.K_i:
                    self.agent.lr = min(0.1, self.agent.trainer.get_lr() * 1.05)
                    self.agent.trainer.update_lr(self.agent.lr)
                elif event.key == pygame.K_k:
                    self.agent.lr = max(1e-6, self.agent.trainer.get_lr() * 0.95)
                    self.agent.trainer.update_lr(self.agent.lr)
                elif event.key == pygame.K_m:
                    self.agent.lr = 0.001
                    self.agent.trainer.update_lr(self.agent.lr)
                elif event.key == pygame.K_l:
                    self.agent.d_epsilon = min(1, self.agent.d_epsilon + 0.05)
                elif event.key == pygame.K_j:
                    self.agent.d_epsilon = max(-1, self.agent.d_epsilon - 0.05)
                elif event.key == pygame.K_o:
                    self.agent.d_epsilon = min(1, 2 * self.agent.d_epsilon)
                elif event.key == pygame.K_u:
                    self.agent.d_epsilon = 0.5 * self.agent.d_epsilon
                elif event.key == pygame.K_n:
                    self.agent.d_epsilon = 0
                elif event.key == pygame.K_t:
                    self.show_text = not self.show_text
                elif event.key == pygame.K_r:
                    self.agent.model.load()
        
        # Move
        self.last_reward = 0
        game_over = False
        self._move(action)
        if np.array_equal(action, [0, 1, 0]):
            self.last_reward -= 0.1
        if np.array_equal(action, [0, 0, 1]):
            self.last_reward -= 0.1
        self.snake.insert(0, self.head)
        
        # new_bbox = self._snake_bbox()
        # if np.array_equal(new_bbox, self.last_bbox):
        #     self.same_bbox_count += 1
        # else:
        #     self.same_bbox_count = 0
        # self.last_bbox = new_bbox
        
        # same_bbox_penalty = min(0, len(self.snake) - self.same_bbox_count)
        # self.last_reward += same_bbox_penalty

        # Check if game over
        if self.is_collision() or self.frame_iteration > 500*len(self.snake):
            game_over = True
            self.last_reward -= 10
            return self.last_reward, game_over, self.score

        # Place new food or move
        if self.head == self.food:
            self.score += 1
            self.last_reward += 10
            self._place_food()
        else:
            self.snake.pop()

        # Update ui and clock
        self._update_ui()
        self.clock.tick(self.speed)
        
        return self.last_reward, game_over, self.score
    
    def reset(self):
        self.t_start = pygame.time.get_ticks()
        self.direction = Direction.RIGHT
        self.idx = CLOCKWISE.index(self.direction)
        x0 = 10 * round(self.w/2/10)
        y0 = 10 * round(self.h/2/10)
        self.head = Point(x0, y0)
        self.mp_x = 0
        self.mp_y = 0
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        # self.last_bbox = self._snake_bbox()
        self.last_reward = 0
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return
        
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
    
    def _move(self, action):
        # [straight, right left]
        
        if np.array_equal(action, [0, 1, 0]):
            self.idx = (self.idx + 1) % N_STATES
        elif np.array_equal(action, [0, 0, 1]):
            self.idx = (self.idx - 1) % N_STATES
        self.direction = CLOCKWISE[self.idx]
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(round(x), round(y))
        
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE 
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        n_points = len(self.snake)
        n = min(n_points, COLOR_SCALE)
        h_min = 2/3 * (1 - n / COLOR_SCALE)
        h_max = 2/3 * (1 + n / (2.5 * COLOR_SCALE))
        
        t = (pygame.time.get_ticks() - self.t_start) / 1000 * self.speed / 20
        
        for i, pt in enumerate(self.snake):
            h = min(h_max, h_min + i / COLOR_SCALE * (h_max - h_min))
            sparkle = max(0, 0.001 * t ** 2 / 300 - 0.03)
            h = h + np.sin(sparkle * t + sparkle * i / COLOR_SCALE) % 1
            
#             s = 0.9 + min(sparkle, 1.5) * np.sin(2 * np.pi * (sparkle + 0.2) * (t + sparkle * i / n_points))
#             s = np.clip(s, 0.2, 1)
            
            col_in = [int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.94, 0.96)]
            col_out = [int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.93, 0.9)]
            
            pygame.draw.rect(self.display, col_out, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, col_in, pygame.Rect(pt.x + 3, pt.y + 3, BLOCK_SIZE - 6, BLOCK_SIZE - 6))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, [0, 0])
        if self.show_text:
            text = self.font.render(f'Snake: {self.head.x} {self.head.y}', True, WHITE)
            self.display.blit(text, [0, 30])
            text = self.font.render(f'Food: {self.food.x} {self.food.y}', True, WHITE)
            self.display.blit(text, [0, 60])
            text = self.font.render(f'LR: {self.agent.lr}', True, WHITE)
            self.display.blit(text, [0, 90])
            text = self.font.render(f'Epsilon: {self.agent.epsilon}', True, WHITE)
            self.display.blit(text, [0, 120])
        
        pygame.display.flip()
        return
