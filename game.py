import colorsys
import random

import numpy as np
import pygame

from collections import namedtuple
from enum import Enum


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLUE3 = (0, 50, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        pygame.init()
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        self.pause_overlay = pygame.Surface((self.w, self.h))
        self.pause_overlay.set_alpha(128)
        self.pause_overlay.fill((0, 0, 0))
        self.font = pygame.font.Font('arial.ttf', 25)
        pygame.display.set_caption('SnakeTorch')
        self.clock = pygame.time.Clock()
        
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.paused = True
        self.play()
        
    def play(self):
        while True:
            game_over, score = self.play_step()
            if game_over == True:
                break
        print(f'Final Score: {score}')
        pygame.quit()
        return                
        
    def play_step(self):
        # Get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_d:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_w:
                    self.direction = Direction.UP
                elif event.key == pygame.K_s:
                    self.direction = Direction.DOWN
                    
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_p:
                    self.paused = not self.paused
        
        # Move
        game_over = False
        if not self.paused:
            self._move(self.direction)
            self.snake.insert(0, self.head)

            # Check if game over
            if self._is_collision():
                game_over = True
                return game_over, self.score

            # Place new food or move
            if self.head == self.food:
                self.score += 1
                self._place_food()
            else:
                self.snake.pop()

        # Update ui and clock
        self._update_ui()
        if not self.paused:
            self.clock.tick(SPEED)
        
        return game_over, self.score
    
    def _is_collision(self):
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        if self.head in self.snake[1:]:
            return True
        return False
    
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)
        
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE 
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        n_points = len(self.snake)
        n = min(n_points, 30)
        h_min = 2/3 * (1 - n / 30)
        h_max = 2/3 * (1 + n / 80)
        
        t = pygame.time.get_ticks() / 1000
        
        
        for i, pt in enumerate(self.snake):
            h = min(h_max, h_min + i / 30 * (h_max - h_min))
            sparkle = max(0, 0.001 * t ** 2 / 30 - 0.03)
            
            s = 0.9 + min(sparkle, 1.5) * np.sin(2 * np.pi * (sparkle + 0.2) * (t + sparkle * i / n_points))
            s = np.clip(s, 0.2, 1)
            
            v = 0.95 + min(sparkle, 1.5) * np.sin(1.5 * np.pi * (sparkle + 0.22) * (t + sparkle * i / n_points))
            v = np.clip(v, 0.5, 1)
            
            col_in = [int(255 * c) for c in colorsys.hsv_to_rgb(h, s, v)]
            col_out = [int(255 * c) for c in colorsys.hsv_to_rgb(h, s * 0.9, v * 0.85)]
            
            pygame.draw.rect(self.display, col_out, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, col_in, pygame.Rect(pt.x + 3, pt.y + 3, BLOCK_SIZE - 6, BLOCK_SIZE - 6))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, [0, 0])
        
        if self.paused:
            self.display.blit(self.pause_overlay, [0, 0])
        
        pygame.display.flip()
        return
