import random
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import deque


# 1. THE BRAIN: PyTorch Neural Network
class SnakeNet(nn.Module):

  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = self.relu(self.linear1(x))
    return self.linear2(x)


# 2. THE TRAINER: Backpropagation Engine
class QTrainer:

  def __init__(self, model, lr=0.0005, gamma=0.9):
    self.lr = lr
    self.gamma = gamma
    self.model = model
    self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
    self.criterion = nn.MSELoss()

  def train_step(self, state, action, reward, next_state, game_over):
    state = torch.tensor(state, dtype=torch.float)
    next_state = torch.tensor(next_state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float)

    if len(state.shape) == 1:
      state = torch.unsqueeze(state, 0)
      next_state = torch.unsqueeze(next_state, 0)
      action = torch.unsqueeze(action, 0)
      reward = torch.unsqueeze(reward, 0)
      game_over = (game_over,)

    # 1. Predicted Q-values with current state
    pred = self.model(state)

    # 2. Vectorized next state predictions (faster + no gradient tracking needed)
    with torch.no_grad():
      next_pred = self.model(next_state)

    target = pred.clone()
    for idx in range(len(game_over)):
      Q_new = reward[idx]
      if not game_over[idx]:
        Q_new = reward[idx] + self.gamma * torch.max(next_pred[idx])

      target[idx][action[idx].item()] = Q_new

    # 3. Backpropagation
    self.optimizer.zero_grad()
    loss = self.criterion(target, pred)
    loss.backward()
    self.optimizer.step()


# 3. THE ENVIRONMENT: Pygame Snake Class
class SnakeGame:

  def __init__(self, width=600, height=600):
    pygame.init()
    self.width = width
    self.height = height
    self.BLOCK_SIZE = 20

    self.BLACK = (0, 0, 0)
    self.GREEN = (0, 255, 0)
    self.RED = (255, 0, 0)
    self.WHITE = (255, 255, 255)

    self.screen = pygame.display.set_mode((self.width, self.height))
    pygame.display.set_caption("Snake AI")
    self.font = pygame.font.Font(None, 36)
    self.clock = pygame.time.Clock()

    self.reset_game()

  def reset_game(self):
    self.snake = [[100, 100], [80, 100], [60, 100]]
    self.direction = [self.BLOCK_SIZE, 0]
    self.score = 0
    self.frame_iteration = 0
    self.food = self._place_food()

    # Track the last 4 moves to catch tight circles
    self.recent_actions = deque(maxlen=4)
    self._draw()

  def _get_food_distance(self):
    head = self.snake[0]
    return math.dist(head, self.food)

  def play_step(self, action, render=True):
    self.frame_iteration += 1
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()

    # Measure distance before moving
    old_distance = self._get_food_distance()

    self._update_direction(action)
    self.recent_actions.append(action)

    head_x = self.snake[0][0] + self.direction[0]
    head_y = self.snake[0][1] + self.direction[1]
    head = [head_x, head_y]
    self.snake.insert(0, head)

    reward = 0
    game_over = False

    # 1. Crash or timeout punishment
    if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
      game_over = True
      reward = -25
      return reward, game_over, self.score

    # 2. Food reward vs. movement shaping
    if head == self.food:
      self.score += 1
      reward = 15
      self.food = self._place_food()
      self.recent_actions.clear()  # Reset loop tracker after scoring
    else:
      self.snake.pop()

      # Distance shaping: nudge it toward the apple
      new_distance = self._get_food_distance()
      if new_distance < old_distance:
        reward += 0.1
      else:
        reward -= 0.15

    # 3. Spinning punishment: 4 right turns or 4 left turns in a row
    if len(self.recent_actions) == 4:
      if list(self.recent_actions) == [1, 1, 1, 1] or list(
              self.recent_actions
      ) == 0:
        reward -= 2

    if render:
      self._draw()
      self.clock.tick()

    return reward, game_over, self.score

  def _place_food(self):
    food = [
        random.randrange(0, self.width, self.BLOCK_SIZE),
        random.randrange(0, self.height, self.BLOCK_SIZE),
    ]
    if food in self.snake:
      return self._place_food()
    return food

  def is_collision(self, pt=None):
    if pt is None:
      pt = self.snake[0]
    if (
        pt[0] < 0
        or pt[0] >= self.width
        or pt[1] < 0
        or pt[1] >= self.height
    ):
      return True
    if pt in self.snake[1:]:
      return True
    return False

  def play_stepv2(self, action):
    self.frame_iteration += 1
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()

    self._update_direction(action)

    head_x = self.snake[0][0] + self.direction[0]
    head_y = self.snake[0][1] + self.direction[1]
    head = [head_x, head_y]
    self.snake.insert(0, head)

    reward = 0
    game_over = False

    self._draw()
    self.clock.tick(360)  # Fast speed so AI trains quickly
    return reward, game_over, self.score

  def _update_direction(self, action):
    clock_wise = [
        [self.BLOCK_SIZE, 0],  # Right
        [0, self.BLOCK_SIZE],  # Down
        [-self.BLOCK_SIZE, 0],  # Left
        [0, -self.BLOCK_SIZE],  # Up
    ]
    idx = clock_wise.index(self.direction)

    if action == 0:
      new_dir = clock_wise[idx]
    elif action == 1:
      new_dir = clock_wise[(idx + 1) % 4]
    else:
      new_dir = clock_wise[(idx - 1) % 4]

    self.direction = new_dir

  def _draw(self):
    self.screen.fill(self.BLACK)
    pygame.draw.rect(
        self.screen,
        self.RED,
        pygame.Rect(
            self.food[0], self.food[1], self.BLOCK_SIZE, self.BLOCK_SIZE
        ),
    )
    for block in self.snake:
      pygame.draw.rect(
          self.screen,
          self.GREEN,
          pygame.Rect(block[0], block[1], self.BLOCK_SIZE, self.BLOCK_SIZE),
      )
    score_surface = self.font.render(f"Score: {self.score}", True, self.WHITE)
    self.screen.blit(score_surface, (self.width - 130, 15))
    pygame.display.flip()


def _count_reachable(game, start_pt):
  """Counts reachable empty grid cells from start_pt using BFS."""
  if game.is_collision(start_pt):
    return 0.0

  b = game.BLOCK_SIZE
  cols = game.width // b
  rows = game.height // b
  total_cells = cols * rows

  # Convert snake body to a set of tuples for fast O(1) lookup
  # Exclude the tail tip because it will move out of the way next frame
  obstacles = set((part[0], part[1]) for part in game.snake[:-1])

  visited = {(start_pt[0], start_pt[1])}
  queue = deque([(start_pt[0], start_pt[1])])

  while queue:
    cx, cy = queue.popleft()
    for dx, dy in [(-b, 0), (b, 0), (0, -b), (0, b)]:
      nx, ny = cx + dx, cy + dy
      if 0 <= nx < game.width and 0 <= ny < game.height:
        if (nx, ny) not in obstacles and (nx, ny) not in visited:
          visited.add((nx, ny))
          queue.append((nx, ny))

  # Return a normalized float between 0.0 and 1.0
  return len(visited) / total_cells


def get_state(game):
  head = game.snake[0]
  b = game.BLOCK_SIZE

  point_l = [head[0] - b, head[1]]
  point_r = [head[0] + b, head[1]]
  point_u = [head[0], head[1] - b]
  point_d = [head[0], head[1] + b]

  dir_l = game.direction == [-b, 0]
  dir_r = game.direction == [b, 0]
  dir_u = game.direction == [0, -b]
  dir_d = game.direction == [0, b]

  # Identify the next coordinate for Straight, Right, and Left relative to heading
  if dir_r:
    pt_straight, pt_right, pt_left = point_r, point_d, point_u
  elif dir_l:
    pt_straight, pt_right, pt_left = point_l, point_u, point_d
  elif dir_u:
    pt_straight, pt_right, pt_left = point_u, point_r, point_l
  else:  # dir_d
    pt_straight, pt_right, pt_left = point_d, point_l, point_r

  state = [
      # 1. Immediate Danger (3 inputs)
      float(game.is_collision(pt_straight)),
      float(game.is_collision(pt_right)),
      float(game.is_collision(pt_left)),
      # 2. Reachable Open Space Ratio (3 new inputs: 0.0 to 1.0)
      _count_reachable(game, pt_straight),
      _count_reachable(game, pt_right),
      _count_reachable(game, pt_left),
      # 3. Current move direction (4 inputs)
      float(dir_l),
      float(dir_r),
      float(dir_u),
      float(dir_d),
      # 4. Food location relative to head (4 inputs)
      float(game.food[0] < head[0]),  # Food left
      float(game.food[0] > head[0]),  # Food right
      float(game.food[1] < head[1]),  # Food up
      float(game.food[1] > head[1]),  # Food down
  ]
  return state
MAX_MEMORY = 100_000
BATCH_SIZE = 1000

if __name__ == "__main__":
  game = SnakeGame()
  model = SnakeNet(input_size=14, hidden_size=256, output_size=3)
  trainer = QTrainer(model, lr=0.0005, gamma=0.95)

  memory = deque(maxlen=MAX_MEMORY)
  games_played = 0
  record = 0

  while True:
    state_old = get_state(game)

    epsilon = max(0, 80 - games_played)
    if random.randint(0, 200) < epsilon:
      action = random.randint(0, 2)
    else:
      prediction = model(torch.tensor(state_old, dtype=torch.float))
      action = torch.argmax(prediction).item()

    reward, game_over, score = game.play_step(action)
    state_new = get_state(game)

    # 1. Train short memory
    trainer.train_step(state_old, action, reward, state_new, game_over)

    # 2. Save transition to long-term memory
    memory.append((state_old, action, reward, state_new, game_over))

    if game_over:
      game.reset_game()
      games_played += 1

      # 3. Train long memory
      if len(memory) > BATCH_SIZE:
        mini_sample = random.sample(memory, BATCH_SIZE)
      else:
        mini_sample = memory

      states, actions, rewards, next_states, dones = zip(*mini_sample)
      trainer.train_step(states, actions, rewards, next_states, dones)

      if score > record:
        record = score
        torch.save(model.state_dict(), "best_snake_model.pth")

      print(f"Game: {games_played} | Score: {score} | Record: {record}")