import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 定数定義
BLACK = 1
WHITE = 2
EMPTY = 0

# 深層学習モデルの定義
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
        return torch.tanh(self.fc2(x))  # -1から1のスコアを出力

# モデル初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OthelloNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ノードクラス
class Node:
    def __init__(self, board, stone, parent=None, move=None):
        self.board = [row[:] for row in board]
        self.stone = stone
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0

    def uct(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def is_fully_expanded(self):
        valid_moves = [(x, y) for y in range(len(self.board)) for x in range(len(self.board[0]))
                       if can_place_x_y(self.board, self.stone, x, y)]
        return len(valid_moves) == len(self.children)

    def best_child(self, c=0):
        return max(self.children, key=lambda child: child.uct(c))

# ユーティリティ関数
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

# PurinAIの定義
class PurinAI:
    def face(self):
        return "🍮"

    def place(self, board, stone):
        move = None
        if self.is_endgame(board):
            # 終盤はミニマックス法で最適な手を計算
            move = self.minimax(board, stone, depth=4)[1]
        else:
            # 中盤はMCTS + 深層学習で探索
            root = Node(board, stone)
            for _ in range(300):  # MCTSのシミュレーション回数
                self.simulate(root)
            move = root.best_child(c=0).move
        return move

    def is_endgame(self, board):
        return sum(row.count(EMPTY) for row in board) <= 10

    def simulate(self, node):
        path = self.select(node)
        leaf = path[-1]
        if not leaf.is_fully_expanded():
            leaf = self.expand(leaf)
        winner = self.rollout(leaf)
        self.backpropagate(path, winner)

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

    def rollout(self, node):
        board = [row[:] for row in node.board]
        stone = node.stone
        while can_place(board, BLACK) or can_place(board, WHITE):
            if can_place(board, stone):
                x, y = random.choice([(x, y) for y in range(len(board)) for x in range(len(board[0])) if can_place_x_y(board, stone, x, y)])
                board[y][x] = stone
                flip_stones(board, stone, x, y)
            stone = 3 - stone
        black_score = sum(row.count(BLACK) for row in board)
        white_score = sum(row.count(WHITE) for row in board)
        return BLACK if black_score > white_score else WHITE if white_score > black_score else 0

    def backpropagate(self, path, winner):
        for node in reversed(path):
            node.visits += 1
            if node.stone == winner:
                node.wins += 1

    def minimax(self, board, stone, depth):
        if depth == 0 or not can_place(board, stone):
            return self.evaluate(board, stone), None

        best_score = -float('inf') if stone == BLACK else float('inf')
        best_move = None

        for y in range(len(board)):
            for x in range(len(board[0])):
                if can_place_x_y(board, stone, x, y):
                    new_board = [row[:] for row in board]
                    new_board[y][x] = stone
                    flip_stones(new_board, stone, x, y)
                    score, _ = self.minimax(new_board, 3 - stone, depth - 1)
                    if (stone == BLACK and score > best_score) or (stone == WHITE and score < best_score):
                        best_score = score
                        best_move = (x, y)

        return best_score, best_move

    def evaluate(self, board, stone):
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        score = model(board_tensor).item()
        return score
