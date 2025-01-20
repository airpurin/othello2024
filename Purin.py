import math
import random

BLACK = 1
WHITE = 2

board = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 0, 0],
    [0, 0, 2, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

def can_place_x_y(board, stone, x, y):
    if board[y][x] != 0:
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

def can_place(board, stone):
    for y in range(len(board)):
        for x in range(len(board[0])):
            if can_place_x_y(board, stone, x, y):
                return True
    return False

def random_place(board, stone):
    while True:
        x = random.randint(0, len(board[0]) - 1)
        y = random.randint(0, len(board) - 1)
        if can_place_x_y(board, stone, x, y):
            return x, y

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

    def face(self):
        return "🍮"

    def place(self, board, stone):
        root = Node(board, stone)
        for _ in range(500):  # シミュレーション回数を調整可能
            self.simulate(root)
        best_move = root.best_child(c=0).move
        return best_move

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
                x, y = random_place(board, stone)
                board[y][x] = stone
                flip_stones(board, stone, x, y)
            stone = 3 - stone
        black_count = sum(row.count(BLACK) for row in board)
        white_count = sum(row.count(WHITE) for row in board)
        return BLACK if black_count > white_count else WHITE if white_count > black_count else 0

    def backpropagate(self, path, winner):
        for node in reversed(path):
            node.visits += 1
            if node.stone == winner:
                node.wins += 1
