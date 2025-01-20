import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


BLACK = 1
WHITE = 2
EMPTY = 0

class OthelloNet(nn.Module):
    def __init__(self):
        super(OthelloNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))  # -1 から 1 のスコア


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OthelloNet().to(device)


class Node:
    def __init__(self, board, stone, parent=None, move=None):
        self.board = [row[:] for row in board]
        self.stone = stone
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0

    def uct(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return self.value_sum / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c=0):
        return max(self.children, key=lambda child: child.uct(c))

    def is_fully_expanded(self):
        valid_moves = [(x, y) for y in range(len(self.board)) for x in range(len(self.board[0]))
                       if can_place_x_y(self.board, self.stone, x, y)]
        return len(valid_moves) == len(self.children)


def can_place_x_y(board, stone, x, y):
    if board[y][x] != EMPTY:
        return False

    opponent = 3 - stone
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        found_opponent = False

        while 0 <= nx < len(board[0]) and 0 <= ny < len(board) and board[ny][nx] == opponent:
            nx += dx
            ny += dy
            found_opponent = True

        if found_opponent and 0 <= nx < len(board[0]) and 0 <= ny < len(board) and board[ny][nx] == stone:
            return True

    return False

def flip_stones(board, stone, x, y):
    opponent = 3 - stone
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        stones_to_flip = []

        while 0 <= nx < len(board[0]) and 0 <= ny < len(board) and board[ny][nx] == opponent:
            stones_to_flip.append((nx, ny))
            nx += dx
            ny += dy

        if stones_to_flip and 0 <= nx < len(board[0]) and 0 <= ny < len(board) and board[ny][nx] == stone:
            for fx, fy in stones_to_flip:
                board[fy][fx] = stone


class PurinAI(object):

   class PurinAI:
    def __init__(self, model):
        self.model = model
        self.device = device
    
    def face(self):
        return "🍮"

    def place(self, board, stone):
        root = Node(board, stone)
        for _ in range(500): 
            self.simulate(root)
        best_move = root.best_child(c=0).move
        return best_move

    def simulate(self, node):
        path = self.select(node)
        leaf = path[-1]
        if not leaf.is_fully_expanded():
            leaf = self.expand(leaf)
        value = self.evaluate(leaf)
        self.backpropagate(path, value)

    def select(self, node):
        path = [node]
        while node.children:
            node = node.best_child()
            path.append(node)
        return path

    def expand(self, node):
        valid_moves = [(x, y) for y in range(len(node.board)) for x in range(len(node.board[0]))
                       if can_place_x_y(node.board, node.stone, x, y)]
        for move in valid_moves:
            if all(child.move != move for child in node.children):
                new_board = [row[:] for row in node.board]
                x, y = move
                new_board[y][x] = node.stone
                flip_stones(new_board, node.stone, x, y)
                child = Node(new_board, 3 - node.stone, parent=node, move=move)
                node.children.append(child)
                return child

    def evaluate(self, node):
        board_tensor = self.board_to_tensor(node.board).to(self.device)
        with torch.no_grad():
            score = self.model(board_tensor).item()
        return score

    def backpropagate(self, path, value):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value

    def board_to_tensor(self, board):
        board_array = np.array(board, dtype=np.float32)
        board_tensor = torch.tensor(board_array).unsqueeze(0).unsqueeze(0).to(self.device)
        return board_tenso
