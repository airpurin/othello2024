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

class Node:
    def __init__(self, board, stone, parent=None, move=None):
        self.board = [row[:] for row in board]  # 現在の盤面
        self.stone = stone  # 現在のプレイヤーの石
        self.parent = parent  # 親ノード
        self.move = move  # このノードに到達するための手
        self.children = []  # 子ノードリスト
        self.visits = 0  # このノードが訪問された回数
        self.wins = 0  # このノードで得た勝利数

    def uct(self, c=1.41):
        """
        UCT (Upper Confidence Bound for Trees) の計算。
        c: 探索と利用のバランスを調整する定数。
        """
        if self.visits == 0:
            return float('inf')  # 未訪問のノードを優先
        return self.wins / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def is_fully_expanded(self):
        """
        子ノードがすべて生成されているかを確認。
        """
        valid_moves = [(x, y) for y in range(len(self.board)) for x in range(len(self.board[0]))
                       if can_place_x_y(self.board, self.stone, x, y)]
        return len(valid_moves) == len(self.children)

    def best_child(self, c=0):
        """
        最も評価の高い子ノードを返す。
        c=0 の場合、探索ではなく利用のみを重視。
        """
        return max(self.children, key=lambda child: child.uct(c))

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
                    
                def flip_stones(board, stone, x, y):
                    """
                    石を (x, y) に置いたとき、挟まれた相手の石をひっくり返す。
                    """
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
def best_child(self, c=1.41):
    """
    UCTに基づく最良の子ノードを選択。
    """
    choices_weights = [(child.wins / (child.visits + 1e-6)) + c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
                       for child in self.children]
    return self.children[np.argmax(choices_weights)]
