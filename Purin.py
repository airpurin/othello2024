import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

BLACK=1
WHITE=2

board = [
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,1,2,0,0],
        [0,0,2,1,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
]

def can_place_x_y(board, stone, x, y):
    """
    石を置けるかどうかを調べる関数。
    board: 2次元配列のオセロボード
    x, y: 石を置きたい座標 (0-indexed)
    stone: 現在のプレイヤーの石 (1: 黒, 2: 白)
    return: 置けるなら True, 置けないなら False
    """
    if board[y][x] != 0:
        return False  # 既に石がある場合は置けない

    opponent = 3 - stone  # 相手の石 (1なら2、2なら1)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        found_opponent = False

        while 0 <= nx < len(board[0]) and 0 <= ny < len(board) and board[ny][nx] == opponent:
            nx += dx
            ny += dy
            found_opponent = True

        if found_opponent and 0 <= nx < len(board[0]) and 0 <= ny < len(board) and board[ny][nx] == stone:
            return True  # 石を置ける条件を満たす

    return False

def can_place(board, stone):
    """
    石を置ける場所を調べる関数。
    board: 2次元配列のオセロボード
    stone: 現在のプレイヤーの石 (1: 黒, 2: 白)
    """
    for y in range(len(board)):
        for x in range(len(board[0])):
            if can_place_x_y(board, stone, x, y):
                return True
    return False

def random_place(board, stone):
    """
    石をランダムに置く関数。
    board: 2次元配列のオセロボード
    stone: 現在のプレイヤーの石 (1: 黒, 2: 白)
    """
    while True:
        x = random.randint(0, len(board[0]) - 1)
        y = random.randint(0, len(board) - 1)
        if can_place_x_y(board, stone, x, y):
            return x, y

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

# モデルの初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OthelloNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# AI クラス
class PurinAI:
    def __init__(self):
        self.model = model
        self.device = device

    def face(self):
        return "🍮"

    def place(self, board, stone):
        """
        モンテカルロ木探索 + 深層学習で最適な手を選択。
        """
        root = Node(board, stone)
        for _ in range(500):  # シミュレーション回数
            self.simulate(root)
        best_move = root.best_child(c=0).move
        return best_move

    def simulate(self, node):
        """
        MCTS のシミュレーションを実行。
        """
        path = self.select(node)
        leaf = path[-1]
        if not leaf.is_fully_expanded():
            leaf = self.expand(leaf)
        winner = self.rollout(leaf)
        self.backpropagate(path, winner)

    def select(self, node):
        """
        UCT に基づいて最適な子ノードを選択。
        """
        path = [node]
        while node.children:
            node = node.best_child()
            path.append(node)
        return path

    def expand(self, node):
        """
        新しい子ノードを1つ生成。
        """
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
        """
        モデルを用いて盤面のスコアを予測。
        """
        board_tensor = self.board_to_tensor(node.board).to(self.device)
        score = self.model(board_tensor).item()
        return score

    def backpropagate(self, path, winner):
        """
        MCTS の結果をバックプロパゲーション。
        """
        for node in reversed(path):
            node.visits += 1
            if node.stone == winner:
                node.wins += 1

    def board_to_tensor(self, board):
        """
        ボードをモデルの入力形式に変換。
        """
        board_array = np.array(board)
        board_tensor = torch.tensor(board_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return board_tensor

# モデルの学習ループ（任意）
def train_model(model, optimizer, criterion, data_loader):
    model.train()
    for epoch in range(10):  # 10エポックの学習
        for boards, targets in data_loader:
            boards = boards.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
