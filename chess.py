import pygame
import sys
import os
import random
import time

# --- Monte Carlo simulation helper ---
class SimGame:
    def __init__(self, game=None):
        # デフォルト値の設定（主に game=None の場合や、上書き前のプレースホルダーとして）
        self.ROWS = 8
        self.COLS = 8
        self.board = [['' for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.turn = 'w'
        self.white_money = 0
        self.black_money = 0
        self.white_income_per_turn = 0
        self.black_income_per_turn = 0
        self.game_over = False
        self.winner = None
        self.piece_rewards = {
            'p': 1,  # ポーン
            'n': 3,  # ナイト
            'b': 3,  # ビショップ
            'r': 5,  # ルーク
            'q': 9,  # クイーン
            'k': 0   # キング（取られるとゲームオーバーなので価値は0）
        }
        self.class_change_map = {
            'p': 'n',  # ポーン→ナイト
            'n': 'b',  # ナイト→ビショップ
            'b': 'r',  # ビショップ→ルーク
            'r': 'q'   # ルーク→クイーン
        }
        self.buy_options = {
            'normal': {'cost': 10, 'pieces': [('p', 0.7), ('n', 0.2), ('b', 0.1)]},
            'rare': {'cost': 30, 'pieces': [('n', 0.4), ('b', 0.4), ('r', 0.2)]},
            'epic': {'cost': 50, 'pieces': [('b', 0.3), ('r', 0.4), ('q', 0.3)]}
        }
        
        # game オブジェクトが渡された場合、その値で属性を上書きする
        if game:
            self.ROWS = game.ROWS
            self.COLS = game.COLS
            self.board = [row[:] for row in game.board]
            self.turn = game.turn
            self.white_money = game.white_money
            self.black_money = game.black_money
            self.white_income_per_turn = game.white_income_per_turn
            self.black_income_per_turn = game.black_income_per_turn
            self.game_over = game.game_over
            self.winner = game.winner
            # 以下の属性はAIの行動決定と評価に不可欠なため、必ずコピーする
            self.buy_options = game.buy_options
            self.class_change_map = game.class_change_map
            self.piece_rewards = game.piece_rewards


    def is_path_clear(self, start, end):
        sr, sc = start
        er, ec = end
        if sr == er:
            step = 1 if ec > sc else -1
            for c in range(sc + step, ec, step):
                if self.board[sr][c] != '':
                    return False
        elif sc == ec:
            step = 1 if er > sr else -1
            for r in range(sr + step, er, step):
                if self.board[r][sc] != '':
                    return False
        elif abs(sr - er) == abs(sc - ec):
            r_step = 1 if er > sr else -1
            c_step = 1 if ec > sc else -1
            r, c = sr + r_step, sc + c_step
            while r != er:
                if self.board[r][c] != '':
                    return False
                r += r_step
                c += c_step
        return True

    def is_valid_move(self, start, end):
        sr, sc = start
        er, ec = end

        if not (0 <= er < self.ROWS and 0 <= ec < self.COLS):
            return False

        piece = self.board[sr][sc]
        if not piece or piece[0] != self.turn:
            return False

        target = self.board[er][ec]
        if target and target[0] == self.turn:
            return False

        # ポーンの移動ロジック (ChessGameクラスのロジックと一貫性を保つ)
        if piece[1] == 'p':
            direction = -1 if self.turn == 'w' else 1
            # 1マス前進
            if sc == ec and target == '' and er == sr + direction:
                return True
            # 2マス前進 (初期位置から)
            if sc == ec and target == '' and \
               ((self.turn == 'w' and sr == self.ROWS - 2 and er == sr - 2 and self.is_path_clear(start, end)) or \
                (self.turn == 'b' and sr == 1 and er == sr + 2 and self.is_path_clear(start, end))):
                return True
            # 斜め攻撃
            if abs(sc - ec) == 1 and er == sr + direction and target and target[0] != self.turn:
                return True
            return False
        # ルークの移動ロジック
        elif piece[1] == 'r':
            return (sr == er or sc == ec) and self.is_path_clear(start, end)
        # ナイトの移動ロジック
        elif piece[1] == 'n':
            dr = abs(sr - er)
            dc = abs(sc - ec)
            return (dr == 1 and dc == 2) or (dr == 2 and dc == 1)
        # ビショップの移動ロジック
        elif piece[1] == 'b':
            return abs(sr - er) == abs(sc - ec) and self.is_path_clear(start, end)
        # クイーンの移動ロジック
        elif piece[1] == 'q':
            return ((sr == er or sc == ec) or (abs(sr - er) == abs(sc - ec))) and self.is_path_clear(start, end)
        # キングの移動ロジック
        elif piece[1] == 'k':
            dr = abs(sr - er)
            dc = abs(sc - ec)
            return dr <= 1 and dc <= 1

        return False

    def end_turn(self):
        if self.turn == 'w':
            self.white_money += self.white_income_per_turn
        else:
            self.black_money += self.black_income_per_turn
        self.turn = 'b' if self.turn == 'w' else 'w'

    def move_piece(self, start, end):
        sr, sc = start
        er, ec = end
        captured_piece = self.board[er][ec]
        self.board[er][ec] = self.board[sr][sc]
        self.board[sr][sc] = ''
        if captured_piece:
            in_enemy_territory = (
                (self.turn == 'w' and er <= 2) or
                (self.turn == 'b' and er >= self.ROWS - 3)
            )
            if in_enemy_territory:
                if self.turn == 'w':
                    self.white_money += 5
                else:
                    self.black_money += 5
        moved_piece = self.board[er][ec]
        if moved_piece and moved_piece[1] == 'p' and (
            (moved_piece[0] == 'w' and er == 0) or
            (moved_piece[0] == 'b' and er == self.ROWS - 1)
        ):
            self.board[er][ec] = moved_piece[0] + 'q'

        self.check_for_king_capture()
        if not self.game_over:
            self.end_turn()

    def get_random_piece(self, option_type):
        option = self.buy_options.get(option_type)
        if not option:
            return None
        pieces_with_prob = option['pieces']
        total_prob = sum(prob for _, prob in pieces_with_prob)
        if total_prob == 0:
            return None
        rand_val = random.random()
        cumulative_prob = 0.0
        for piece_type, prob in pieces_with_prob:
            cumulative_prob += prob
            if rand_val < cumulative_prob:
                return piece_type
        return None

    def buy_piece(self, option_type, target_pos):
        cost = self.buy_options[option_type]['cost']
        if self.turn == 'w':
            current_money = self.white_money
        else:
            current_money = self.black_money
        if current_money >= cost:
            drawn_piece_type = self.get_random_piece(option_type)
            if drawn_piece_type:
                if self.turn == 'w':
                    self.white_money -= cost
                    self.board[target_pos[0]][target_pos[1]] = 'w' + drawn_piece_type
                else:
                    self.black_money -= cost
                    self.board[target_pos[0]][target_pos[1]] = 'b' + drawn_piece_type
                self.end_turn()

    def hire_employee(self):
        cost = 40
        if self.turn == 'w':
            if self.white_money >= cost:
                self.white_money -= cost
                self.white_income_per_turn += 2
                self.end_turn()
        else:
            if self.black_money >= cost:
                self.black_money -= cost
                self.black_income_per_turn += 2
                self.end_turn()

    def class_change(self, pos):
        cost = 70
        sr, sc = pos
        piece_to_change = self.board[sr][sc]
        if not piece_to_change or piece_to_change[0] != self.turn:
            return
        if self.turn == 'w':
            if self.white_money >= cost:
                original_type = piece_to_change[1]
                new_type = self.class_change_map.get(original_type)
                if new_type:
                    self.white_money -= cost
                    self.board[sr][sc] = self.turn + new_type
                    self.end_turn()
        else:
            if self.black_money >= cost:
                original_type = piece_to_change[1]
                new_type = self.class_change_map.get(original_type)
                if new_type:
                    self.black_money -= cost
                    self.board[sr][sc] = self.turn + new_type
                    self.end_turn()

    def check_for_king_capture(self):
        white_king_exists = False
        black_king_exists = False
        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = self.board[r][c]
                if piece == 'wk':
                    white_king_exists = True
                elif piece == 'bk':
                    black_king_exists = True
        if not white_king_exists:
            self.game_over = True
            self.winner = 'b'
        elif not black_king_exists:
            self.game_over = True
            self.winner = 'w'

    def get_possible_actions(self):
        actions = []
        # 駒の移動
        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = self.board[r][c]
                if piece and piece[0] == self.turn:
                    for er in range(self.ROWS):
                        for ec in range(self.COLS):
                            if self.is_valid_move((r, c), (er, ec)):
                                actions.append({'type': 'move', 'start': (r, c), 'end': (er, ec)})

        # 駒の購入
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.board[r][c] == '':
                    is_in_territory = False
                    if self.turn == 'w' and r >= self.ROWS - 3:
                        is_in_territory = True
                    elif self.turn == 'b' and r <= 2:
                        is_in_territory = True
                    if is_in_territory:
                        for option_type in self.buy_options.keys():
                            cost = self.buy_options[option_type]['cost']
                            current_money = self.white_money if self.turn == 'w' else self.black_money
                            if current_money >= cost:
                                actions.append({'type': 'buy', 'option': option_type, 'target_pos': (r, c)})

        # 従業員の雇用
        cost_hire = 40
        current_money = self.white_money if self.turn == 'w' else self.black_money
        if current_money >= cost_hire:
            actions.append({'type': 'hire'})

        # クラス変更
        cost_class_change = 70
        if current_money >= cost_class_change:
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    piece = self.board[r][c]
                    if piece and piece[0] == self.turn and piece[1] in self.class_change_map:
                        actions.append({'type': 'class_change', 'pos': (r, c)})

        return actions

    def apply_action(self, action):
        if action['type'] == 'move':
            self.move_piece(action['start'], action['end'])
        elif action['type'] == 'buy':
            self.buy_piece(action['option'], action['target_pos'])
        elif action['type'] == 'hire':
            self.hire_employee()
        elif action['type'] == 'class_change':
            self.class_change(action['pos'])

# --- ChessGame クラスの定義 ---
class ChessGame:
    def __init__(self):
        self.ROWS, self.COLS = 6, 6
        self.board = [
            ['', '', 'bk', '', '', ''],
            [''] * self.COLS,
            [''] * self.COLS,
            [''] * self.COLS,
            [''] * self.COLS,
            ['', '', '', 'wk', '', '']
        ]
        self.turn = 'w' # 'w' for white, 'b' for black
        self.game_over = False
        self.winner = None
        self.white_money = 50
        self.black_money = 50
        self.white_income_per_turn = 10
        self.black_income_per_turn = 10
        self.game_log_messages = ["Game Start!"]
        self.last_cpu_action = None

        self.piece_rewards = {
            'p': 10, 'r': 30, 'n': 30, 'b': 30, 'q': 30, 'k': 0
        }
        self.buy_options = {
            'normal': {'cost': 20, 'pieces': [('p', 0.90), ('r', 0.033), ('n', 0.033), ('b', 0.033), ('q', 0.001)]},
            'rare': {'cost': 30, 'pieces': [('p', 0.50), ('r', 0.15), ('n', 0.15), ('b', 0.15), ('q', 0.05)]},
            'epic': {'cost': 40, 'pieces': [('p', 0.10), ('r', 0.25), ('n', 0.25), ('b', 0.25), ('q', 0.15)]}
        }
        self.class_change_map = {
            'p': 'n', 'n': 'b', 'b': 'r',
        }

        # --- 犬の画像管理
        self.dog_images_dict = {}
        self.current_dog_image_key = 'default'
        self.dog_image_display_size = (500, 500)
        self._load_all_dog_images()

    def _load_all_dog_images(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dog_folder_path = os.path.join(current_dir, "dog")

        if not os.path.exists(dog_folder_path):
            print(f"警告: 'dog' フォルダが見つかりません: {dog_folder_path}")
            return

        action_image_map = {
            'default': 'default.png', 'move': 'move.png', 'buy': 'buy.png',
            'hire': 'hire.png', 'class_change': 'class_change.png',
            'win': 'win.png', 'lose': 'lose.png', 'thinking': 'thinking.png',
            'happy': 'happy.png', 'sad': 'sad.png',
        }

        for action, filename in action_image_map.items():
            path = os.path.join(dog_folder_path, filename)
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path).convert_alpha()
                    self.dog_images_dict[action] = pygame.transform.scale(img, self.dog_image_display_size)
                except pygame.error as e:
                    print(f"警告: 犬の画像 '{path}' の読み込み中にエラーが発生しました: {e}")
            else:
                print(f"警告: 犬の画像 '{path}' が見つかりません。'{action}' の画像は表示されません。")

        if 'default' not in self.dog_images_dict:
            print("エラー: 'default.png' が 'dog' フォルダに見つかりませんでした。デフォルト画像は必須です。")
            dummy_surface = pygame.Surface(self.dog_image_display_size, pygame.SRCALPHA)
            dummy_surface.fill((255, 0, 0, 128))
            pygame.draw.circle(dummy_surface, (255, 255, 255), (self.dog_image_display_size[0]//2, self.dog_image_display_size[1]//2), self.dog_image_display_size[0]//2 - 10)
            self.dog_images_dict['default'] = dummy_surface

    def set_dog_image(self, key):
        if key in self.dog_images_dict:
            self.current_dog_image_key = key
        else:
            self.current_dog_image_key = 'default'

    def get_current_dog_image(self):
        return self.dog_images_dict.get(self.current_dog_image_key)

    def reset_game(self):
        """Reset all game state to start a new game."""
        self.__init__()
        self.set_dog_image('default')

    def add_log_message(self, message):
        self.game_log_messages.append(message)
        if len(self.game_log_messages) > 2:
            self.game_log_messages.pop(0)

    def is_path_clear(self, start, end):
        sr, sc = start
        er, ec = end
        # 水平移動
        if sr == er:
            step = 1 if ec > sc else -1
            for c in range(sc + step, ec, step):
                if self.board[sr][c] != '':
                    return False
        # 垂直移動
        elif sc == ec:
            step = 1 if er > sr else -1
            for r in range(sr + step, er, step):
                if self.board[r][sc] != '':
                    return False
        # 斜め移動
        elif abs(sr - er) == abs(sc - ec):
            r_step = 1 if er > sr else -1
            c_step = 1 if ec > sc else -1
            r, c = sr + r_step, sc + c_step
            while r != er:
                if self.board[r][c] != '':
                    return False
                r += r_step
                c += c_step
        return True

    def is_valid_move(self, start, end):
        sr, sc = start
        er, ec = end

        if not (0 <= er < self.ROWS and 0 <= ec < self.COLS):
            return False

        piece = self.board[sr][sc]
        if not piece or piece[0] != self.turn:
            return False

        target = self.board[er][ec]
        if target and target[0] == self.turn:
            return False

        # ポーンの移動ロジック
        if piece[1] == 'p':
            direction = -1 if self.turn == 'w' else 1
            # 1マス前進
            if sc == ec and target == '' and er == sr + direction: return True
            # 2マス前進 (初期位置から)
            if sc == ec and target == '' and \
               ((self.turn == 'w' and sr == self.ROWS - 2 and er == sr - 2 and self.is_path_clear(start, end)) or \
                (self.turn == 'b' and sr == 1 and er == sr + 2 and self.is_path_clear(start, end))): return True
            # 斜め攻撃
            if abs(sc - ec) == 1 and er == sr + direction and target and target[0] != self.turn: return True
            return False
        # ルークの移動ロジック
        elif piece[1] == 'r':
            return (sr == er or sc == ec) and self.is_path_clear(start, end)
        # ナイトの移動ロジック
        elif piece[1] == 'n':
            dr = abs(sr - er)
            dc = abs(sc - ec)
            return (dr == 1 and dc == 2) or (dr == 2 and dc == 1)
        # ビショップの移動ロジック
        elif piece[1] == 'b':
            return abs(sr - er) == abs(sc - ec) and self.is_path_clear(start, end)
        # クイーンの移動ロジック
        elif piece[1] == 'q':
            return ((sr == er or sc == ec) or (abs(sr - er) == abs(sc - ec))) and self.is_path_clear(start, end)
        # キングの移動ロジック
        elif piece[1] == 'k':
            dr = abs(sr - er)
            dc = abs(sc - ec)
            return dr <= 1 and dc <= 1

        return False

    def end_turn(self):
        if self.turn == 'w':
            self.white_money += self.white_income_per_turn
        else:
            self.black_money += self.black_income_per_turn

        self.turn = 'b' if self.turn == 'w' else 'w'
        self.add_log_message(f"{'white' if self.turn == 'w' else 'black'} turn")
        self.set_dog_image('default')

    def move_piece(self, start, end):
        sr, sc = start
        er, ec = end

        if self.turn == 'b':
            self.last_cpu_action = {'type': 'move', 'start': start, 'end': end}

        captured_piece_full_name = self.board[er][ec]

        reward = 0
        if captured_piece_full_name:
            captured_piece_type = captured_piece_full_name[1]
            if captured_piece_type == 'k':
                reward = 10
                self.add_log_message(f"King captured!")
                self.set_dog_image('happy')
            else:
                reward = 1
                self.set_dog_image('move')
            in_enemy_territory = (
                (self.turn == 'w' and er <= 2) or
                (self.turn == 'b' and er >= self.ROWS - 3)
            )
            if in_enemy_territory:
                if self.turn == 'w':
                    self.white_money += 5
                else:
                    self.black_money += 5
                self.add_log_message("Bonus +5G for attack!")
        else:
            self.set_dog_image('move')

        self.board[er][ec] = self.board[sr][sc]
        self.board[sr][sc] = ''
        moved_piece = self.board[er][ec]
        if moved_piece and moved_piece[1] == 'p' and (
            (moved_piece[0] == 'w' and er == 0) or
            (moved_piece[0] == 'b' and er == self.ROWS - 1)
        ):
            self.board[er][ec] = moved_piece[0] + 'q'
            self.add_log_message(
                f"{'White' if moved_piece[0] == 'w' else 'Black'} pawn promoted to Queen!"
            )

        self.check_for_king_capture()
        if self.game_over:
            self.add_log_message(f"Finish! Winner is {'white' if self.winner == 'w' else 'black'}!")
            if (self.winner == 'w' and self.turn == 'w') or (self.winner == 'b' and self.turn == 'b'):
                self.set_dog_image('win')
            else:
                self.set_dog_image('lose')
            return reward

        self.end_turn()
        return reward

    def get_random_piece(self, option_type):
        if option_type not in self.buy_options: return None
        option = self.buy_options[option_type]
        pieces_with_prob = option['pieces']
        total_prob = sum(prob for _, prob in pieces_with_prob)
        if total_prob == 0: return None

        rand_val = random.random()
        cumulative_prob = 0.0
        for piece_type, prob in pieces_with_prob:
            cumulative_prob += prob
            if rand_val < cumulative_prob: return piece_type
        return None

    def buy_piece(self, option_type, target_pos):
        cost = self.buy_options[option_type]['cost']
        current_money = self.white_money if self.turn == 'w' else self.black_money

        if current_money >= cost:
            drawn_piece_type = self.get_random_piece(option_type)
            if drawn_piece_type:
                if self.turn == 'w':
                    self.white_money -= cost
                    self.board[target_pos[0]][target_pos[1]] = 'w' + drawn_piece_type
                else:
                    self.black_money -= cost
                    self.board[target_pos[0]][target_pos[1]] = 'b' + drawn_piece_type

                if self.turn == 'b':
                    self.last_cpu_action = {'type': 'buy', 'pos': target_pos}

                self.add_log_message(f"{'White' if self.turn == 'w' else 'Black'} choice {option_type}, and buy {drawn_piece_type}.")
                self.set_dog_image('buy')
                self.end_turn()
                return True
        return False

    def hire_employee(self):
        cost = 40
        current_money = self.white_money if self.turn == 'w' else self.black_money
        if current_money >= cost:
            if self.turn == 'w':
                self.white_money -= cost
                self.white_income_per_turn += 2
            else:
                self.black_money -= cost
                self.black_income_per_turn += 2
            if self.turn == 'b':
                self.last_cpu_action = {'type': 'hire'}
            self.add_log_message(f"You hire employee!")
            self.set_dog_image('buy')
            self.end_turn()
            return True
        return False

    def class_change(self, selected_pos):
        cost = 70
        sr, sc = selected_pos
        piece_to_change = self.board[sr][sc]

        if not piece_to_change or piece_to_change[0] != self.turn:
            return False

        current_money = self.white_money if self.turn == 'w' else self.black_money

        if current_money >= cost:
            original_type = piece_to_change[1]
            if original_type in self.class_change_map:
                new_type = self.class_change_map[original_type]

                if self.turn == 'w':
                    self.white_money -= cost
                else:
                    self.black_money -= cost

                self.board[sr][sc] = self.turn + new_type
                if self.turn == 'b':
                    self.last_cpu_action = {'type': 'class_change', 'pos': (sr, sc)}
                self.add_log_message(f"{'White' if self.turn == 'w' else 'Black'}'s {original_type} is changed class to {new_type}.")
                self.set_dog_image('buy')
                self.end_turn()
                return True
        return False

    def check_for_king_capture(self):
        white_king_exists = False
        black_king_exists = False

        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = self.board[r][c]
                if piece == 'wk':
                    white_king_exists = True
                elif piece == 'bk':
                    black_king_exists = True

        if not white_king_exists:
            self.game_over = True
            self.winner = 'b'
        elif not black_king_exists:
            self.game_over = True
            self.winner = 'w'
        return 0

    def get_possible_actions(self):
        actions = []
        # 駒の移動
        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = self.board[r][c]
                if piece and piece[0] == self.turn:
                    for er in range(self.ROWS):
                        for ec in range(self.COLS):
                            if self.is_valid_move((r, c), (er, ec)):
                                actions.append({'type': 'move', 'start': (r, c), 'end': (er, ec)})

        # 駒の購入
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.board[r][c] == '':
                    is_in_territory = False
                    if self.turn == 'w' and r >= self.ROWS - 3:
                        is_in_territory = True
                    elif self.turn == 'b' and r <= 2:
                        is_in_territory = True
                    if is_in_territory:
                        for option_type in self.buy_options.keys():
                            cost = self.buy_options[option_type]['cost']
                            current_money = self.white_money if self.turn == 'w' else self.black_money
                            if current_money >= cost:
                                actions.append({'type': 'buy', 'option': option_type, 'target_pos': (r, c)})

        # 従業員の雇用
        cost_hire = 40
        current_money = self.white_money if self.turn == 'w' else self.black_money
        if current_money >= cost_hire:
            actions.append({'type': 'hire'})

        # クラス変更
        cost_class_change = 70
        if current_money >= cost_class_change:
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    piece = self.board[r][c]
                    if piece and piece[0] == self.turn and piece[1] in self.class_change_map:
                        actions.append({'type': 'class_change', 'pos': (r, c)})
        return actions

    # --- AI: CPUのターン処理 ---
    def cpu_play_turn(self):
        self.set_dog_image('thinking')
        actions = self.get_possible_actions()

        if not actions:
            self.add_log_message(f"No valid actions for {'white' if self.turn == 'w' else 'black'} CPU.")
            self.end_turn()
            return

        # --- CPU: Pure Alpha-Beta search ---
        def evaluate_state(state, player):
            score = 0
            piece_count = 0
            king_position = None

            # 駒の価値と位置による評価
            for r in range(state.ROWS):
                for c in range(state.COLS):
                    piece = state.board[r][c]
                    if piece:
                        if piece[0] == player:
                            piece_count += 1
                            val = state.piece_rewards.get(piece[1], 0)
                            position_bonus = 0
                            if piece[1] == 'p':  # ポーン
                                if player == 'w':
                                    position_bonus = (state.ROWS - 1 - r) * 0.5
                                else:
                                    position_bonus = r * 0.5
                            elif piece[1] == 'k':  # キング
                                king_position = (r, c)
                                if player == 'w':
                                    position_bonus = (state.ROWS - 1 - r) * 2
                                else:
                                    position_bonus = r * 2
                            elif piece[1] in ['r', 'q']:  # ルークとクイーン
                                center_distance = abs(c - state.COLS // 2)
                                position_bonus = (state.COLS // 2 - center_distance) * 0.5
                            score += val + position_bonus
                        else:
                            val = state.piece_rewards.get(piece[1], 0)
                            score -= val

            if piece_count > 0:
                efficiency_bonus = piece_count * 0.5
                score += efficiency_bonus

            if king_position:
                r, c = king_position
                defense_score = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < state.ROWS and 0 <= nc < state.COLS:
                            piece = state.board[nr][nc]
                            if piece and piece[0] == player:
                                defense_score += 1
                score += defense_score * 0.5

            money_diff = state.white_money - state.black_money
            score += money_diff if player == 'w' else -money_diff

            if state.game_over:
                if state.winner == player:
                    score += 1000
                elif state.winner and state.winner != player:
                    score -= 1000
            return score

        def alpha_beta(state, depth, alpha, beta, maximizing, player):
            if depth == 0 or state.game_over:
                return evaluate_state(state, player)

            actions = state.get_possible_actions()
            if maximizing:
                value = float('-inf')
                for action in actions:
                    sim = SimGame(state)
                    sim.apply_action(action)
                    value = max(value, alpha_beta(sim, depth - 1, alpha, beta, False, player))
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value
            else:
                value = float('inf')
                for action in actions:
                    sim = SimGame(state)
                    sim.apply_action(action)
                    value = min(value, alpha_beta(sim, depth - 1, alpha, beta, True, player))
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                return value

        def best_action(state, depth):
            actions = state.get_possible_actions()
            if not actions: return None
            best_val = float('-inf')
            best_actions = []
            for action in actions:
                sim = SimGame(state)
                sim.apply_action(action)
                val = alpha_beta(sim, depth - 1, float('-inf'), float('inf'), False, state.turn)
                if val > best_val:
                    best_val = val
                    best_actions = [action]
                elif val == best_val:
                    best_actions.append(action)
            return random.choice(best_actions)

        chosen_action = best_action(SimGame(self), 3)
        if not chosen_action:
             self.add_log_message(f"CPU has no move.")
             self.end_turn()
             return

        self.add_log_message(f"CPU chooses: {chosen_action['type']}")

        if chosen_action['type'] == 'move':
            self.move_piece(chosen_action['start'], chosen_action['end'])
        elif chosen_action['type'] == 'buy':
            self.buy_piece(chosen_action['option'], chosen_action['target_pos'])
        elif chosen_action['type'] == 'hire':
            self.hire_employee()
        elif chosen_action['type'] == 'class_change':
            self.class_change(chosen_action['pos'])

# --- メインゲームループの関数 ---
def run_chess_game():
    pygame.init()
    pygame.mixer.init()

    music_path = "sound/ohirusugi.mp3"
    move_sound_path = "sound/kon.mp3"

    current_dir = os.path.dirname(os.path.abspath(__file__))

    if os.path.exists(os.path.join(current_dir, music_path)):
        pygame.mixer.music.load(os.path.join(current_dir, music_path))
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play(-1)
    else:
        print(f"警告: BGMファイル '{os.path.join(current_dir, music_path)}' が見つかりません。")

    move_sound = None
    if os.path.exists(os.path.join(current_dir, move_sound_path)):
        move_sound = pygame.mixer.Sound(os.path.join(current_dir, move_sound_path))
        move_sound.set_volume(0.7)
    else:
        print(f"警告: 効果音ファイル '{os.path.join(current_dir, move_sound_path)}' が見つかりません。")

    WIDTH, HEIGHT = 1000, 600
    SQUARE_SIZE = 64
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess Game")

    LIGHT = (240, 217, 181)
    DARK = (181, 136, 99)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    WHITE_TEXT = (255, 255, 255)
    BLACK_TEXT = (0, 0, 0)
    GRAY_OVERLAY = (100, 100, 100, 128)
    UI_BACKGROUND_COLOR = (50, 50, 50)
    MOVE_HIGHLIGHT = (144, 238, 144, 160)
    OPPONENT_HIGHLIGHT = (255, 165, 0, 160)
    PROB_BG_COLOR = (0, 0, 0)

    pygame.font.init()
    font = pygame.font.SysFont('meiryo', 48)
    money_font = pygame.font.SysFont('meiryo', 20)
    buy_font = pygame.font.SysFont('meiryo', 23)
    button_font = pygame.font.SysFont('meiryo', 18)
    turn_font = pygame.font.SysFont('meiryo', 23)
    log_font = pygame.font.SysFont('meiryo', 23)

    movement_descriptions = {
        'p': 'ポーン: 前に1マス(初回は2マス)。斜め前で敵を取る',
        'r': 'ルーク: 縦横に好きなだけ移動',
        'n': 'ナイト: L字に移動',
        'b': 'ビショップ: 斜めに好きなだけ移動',
        'q': 'クイーン: 縦横斜めに好きなだけ移動',
        'k': 'キング: 周囲1マス移動'
    }

    bg_path = "haikei.png"
    bg = None
    if os.path.exists(os.path.join(current_dir, bg_path)):
        bg = pygame.image.load(os.path.join(current_dir, bg_path)).convert_alpha()
        bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
    else:
        print(f"警告: 背景画像ファイル '{os.path.join(current_dir, bg_path)}' が見つかりません。")

    BOARD_OFFSET_X = 120
    BOARD_OFFSET_Y = 120

    piece_names = ['wp', 'wr', 'wn', 'wb', 'wq', 'wk', 'bp', 'br', 'bn', 'bb', 'bq', 'bk']
    pieces = {}
    for name in piece_names:
        path = os.path.join(current_dir, "chess_pieces", f"{name}.png")
        try:
            if not os.path.exists(path):
                print(f"エラー: 画像ファイル '{path}' が見つかりません。")
                sys.exit(1)
            img = pygame.image.load(path)
            pieces[name] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        except pygame.error as e:
            print(f"画像 '{path}' の読み込み中にエラーが発生しました: {e}")
            sys.exit(1)

    game = ChessGame()
    game.set_dog_image('default')

    selected = None
    selected_empty_square = None
    clock = pygame.time.Clock()
    hovered_text = None
    hover_pos = (0, 0)

    UI_AREA_START_X = BOARD_OFFSET_X + game.COLS * SQUARE_SIZE + 30
    turn_display_rect = pygame.Rect(UI_AREA_START_X, 40, WIDTH - UI_AREA_START_X - 260, 36)
    log_display_rect = pygame.Rect(UI_AREA_START_X, turn_display_rect.bottom + 5, WIDTH - UI_AREA_START_X - 20, 80)
    dog_image_display_rect = pygame.Rect(500, 110, 10, 60)
    
    MONEY_RECT_SIZE = (240, 80)
    money_bg_rect = pygame.Rect(20, HEIGHT - MONEY_RECT_SIZE[1] - 20, MONEY_RECT_SIZE[0], MONEY_RECT_SIZE[1])
    
    button_area_center_y = HEIGHT - 45
    action_button_rect = pygame.Rect(0, 0, 140, 50)
    action_button_rect.center = (UI_AREA_START_X + 100, button_area_center_y)
    
    buy_normal_button = pygame.Rect(0, 0, 120, 30)
    buy_normal_button.center = (UI_AREA_START_X + 100, button_area_center_y + 20)
    buy_rare_button = pygame.Rect(0, 0, 120, 30)
    buy_rare_button.center = (UI_AREA_START_X + 100, button_area_center_y - 20)
    buy_epic_button = pygame.Rect(0, 0, 120, 30)
    buy_epic_button.center = (UI_AREA_START_X + 100, button_area_center_y - 60)
    
    reset_button_rect = pygame.Rect(0, 0, 120, 30)
    reset_button_initial_center = (UI_AREA_START_X + 100, button_area_center_y - 100)
    reset_button_rect.center = reset_button_initial_center

    def draw_button(surface, rect, text, color, text_color, font_obj):
        pygame.draw.rect(surface, color, rect)
        text_surface = font_obj.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)

    def draw_game():
        if game.game_over:
            reset_button_rect.center = (WIDTH // 2, HEIGHT // 2 + 60)
        else:
            reset_button_rect.center = reset_button_initial_center

        screen.fill(UI_BACKGROUND_COLOR)
        if bg:
            screen.blit(bg, (0, 0))

        for row in range(game.ROWS):
            for col in range(game.COLS):
                color = LIGHT if (row + col) % 2 == 0 else DARK
                pygame.draw.rect(screen, color, (col*SQUARE_SIZE + BOARD_OFFSET_X, row*SQUARE_SIZE + BOARD_OFFSET_Y, SQUARE_SIZE, SQUARE_SIZE))
                piece = game.board[row][col]
                if piece:
                    screen.blit(pieces[piece], (col*SQUARE_SIZE + BOARD_OFFSET_X, row*SQUARE_SIZE + BOARD_OFFSET_Y))

        if hovered_text:
            tooltip_surface = button_font.render(hovered_text, True, WHITE_TEXT)
            tooltip_rect = tooltip_surface.get_rect(topleft=(hover_pos[0] + 10, hover_pos[1] - 20))
            bg_rect = tooltip_rect.inflate(4, 4)
            pygame.draw.rect(screen, (0, 0, 0), bg_rect)
            screen.blit(tooltip_surface, tooltip_rect)

        if game.turn == 'w' and game.last_cpu_action and not game.game_over:
            action = game.last_cpu_action
            positions = []
            if action['type'] == 'move':
                positions = [action['start'], action['end']]
            elif action['type'] in ('buy', 'class_change'):
                positions = [action['pos']]
            for pr, pc in positions:
                highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                highlight_surface.fill(OPPONENT_HIGHLIGHT)
                screen.blit(highlight_surface, (pc*SQUARE_SIZE + BOARD_OFFSET_X, pr*SQUARE_SIZE + BOARD_OFFSET_Y))

        if selected and not game.game_over:
            for er in range(game.ROWS):
                for ec in range(game.COLS):
                    if game.is_valid_move(selected, (er, ec)):
                        highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                        highlight_surface.fill(MOVE_HIGHLIGHT)
                        screen.blit(highlight_surface, (ec*SQUARE_SIZE + BOARD_OFFSET_X, er*SQUARE_SIZE + BOARD_OFFSET_Y))
            pygame.draw.rect(screen, GREEN, (selected[1]*SQUARE_SIZE + BOARD_OFFSET_X, selected[0]*SQUARE_SIZE + BOARD_OFFSET_Y, SQUARE_SIZE, SQUARE_SIZE), 3)

        if selected_empty_square and not game.game_over:
            pygame.draw.rect(screen, (0, 150, 255), (selected_empty_square[1]*SQUARE_SIZE + BOARD_OFFSET_X, selected_empty_square[0]*SQUARE_SIZE + BOARD_OFFSET_Y, SQUARE_SIZE, SQUARE_SIZE), 3)

        pygame.draw.rect(screen, (70, 70, 70), turn_display_rect)
        turn_text = turn_font.render(f"{'white' if game.turn == 'w' else 'black'} turn", True, WHITE_TEXT)
        turn_text_rect = turn_text.get_rect(center=turn_display_rect.center)
        screen.blit(turn_text, turn_text_rect)

        pygame.draw.rect(screen, (30, 30, 30), log_display_rect)
        for i, msg in enumerate(game.game_log_messages):
            log_text_surface = log_font.render(msg, True, WHITE_TEXT)
            screen.blit(log_text_surface, (log_display_rect.x + 10, log_display_rect.y + 10 + i * 25))

        current_dog_img = game.get_current_dog_image()
        if current_dog_img:
            dog_rect = current_dog_img.get_rect(topleft=(UI_AREA_START_X, log_display_rect.bottom + 20))
            screen.blit(current_dog_img, dog_rect.topleft)

        money_overlay = pygame.Surface((money_bg_rect.width, money_bg_rect.height), pygame.SRCALPHA)
        money_overlay.fill(GRAY_OVERLAY)
        screen.blit(money_overlay, money_bg_rect.topleft)

        white_money_text = money_font.render(f"white money: {game.white_money}G (+{game.white_income_per_turn}G)", True, WHITE_TEXT)
        black_money_text = money_font.render(f"black money: {game.black_money}G (+{game.black_income_per_turn}G)", True, BLACK_TEXT)
        white_money_text_rect = white_money_text.get_rect(center=(money_bg_rect.centerx, money_bg_rect.y + 20))
        black_money_text_rect = black_money_text.get_rect(center=(money_bg_rect.centerx, money_bg_rect.y + 60))
        screen.blit(white_money_text, white_money_text_rect)
        screen.blit(black_money_text, black_money_text_rect)

        if not game.game_over:
            draw_button(screen, reset_button_rect, "Reset", (180, 80, 80), WHITE_TEXT, button_font)

        if not game.game_over:
            if selected_empty_square:
                draw_button(screen, buy_normal_button, f"Normal ({game.buy_options['normal']['cost']}G)", (150, 150, 200), WHITE_TEXT, buy_font)
                draw_button(screen, buy_rare_button, f"Rare ({game.buy_options['rare']['cost']}G)", (200, 150, 100), WHITE_TEXT, buy_font)
                draw_button(screen, buy_epic_button, f"Epic ({game.buy_options['epic']['cost']}G)", (250, 100, 50), WHITE_TEXT, buy_font)

                def format_prob(option):
                    piece_names = {'p': 'ポーン', 'r': 'ルーク', 'n': 'ナイト', 'b': 'ビショップ', 'q': 'クイーン'}
                    return ', '.join([f"{piece_names.get(pt, pt)}:{prob*100:.1f}%" for pt, prob in game.buy_options[option]['pieces']])

                prob_y_offset = 18
                normal_prob_surface = button_font.render(format_prob('normal'), True, WHITE_TEXT)
                normal_prob_rect = normal_prob_surface.get_rect(midleft=(buy_normal_button.right + 10, buy_normal_button.centery))
                screen.blit(normal_prob_surface, normal_prob_rect)
                
                rare_prob_surface = button_font.render(format_prob('rare'), True, WHITE_TEXT)
                rare_prob_rect = rare_prob_surface.get_rect(midleft=(buy_rare_button.right + 10, buy_rare_button.centery))
                screen.blit(rare_prob_surface, rare_prob_rect)

                epic_prob_surface = button_font.render(format_prob('epic'), True, WHITE_TEXT)
                epic_prob_rect = epic_prob_surface.get_rect(midleft=(buy_epic_button.right + 10, buy_epic_button.centery))
                screen.blit(epic_prob_surface, epic_prob_rect)

            elif selected and game.board[selected[0]][selected[1]] and game.board[selected[0]][selected[1]][0] == game.turn:
                piece_at_selected = game.board[selected[0]][selected[1]]
                if piece_at_selected[1] in game.class_change_map:
                    draw_button(screen, action_button_rect, f"Class Change (70G)", (100, 150, 200), WHITE_TEXT, button_font)
                else:
                    draw_button(screen, action_button_rect, f"Hire Employee (40G)", (50, 150, 50), WHITE_TEXT, button_font)
            else:
                draw_button(screen, action_button_rect, f"Hire Employee (40G)", (50, 150, 50), WHITE_TEXT, button_font)

        if game.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            text_surface = font.render(f"{'white' if game.winner == 'w' else 'black'} win!", True, RED)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(text_surface, text_rect)
            draw_button(screen, reset_button_rect, "Reset", (180, 80, 80), WHITE_TEXT, button_font)

    running = True
    while running:
        draw_game()
        pygame.display.flip()

        if not game.game_over and game.turn == 'b':
            pygame.time.wait(500) # CPUの思考時間を少し短縮
            game.cpu_play_turn()
            selected = None
            selected_empty_square = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                if BOARD_OFFSET_X <= x < BOARD_OFFSET_X + game.COLS * SQUARE_SIZE and \
                   BOARD_OFFSET_Y <= y < BOARD_OFFSET_Y + game.ROWS * SQUARE_SIZE:
                    col = (x - BOARD_OFFSET_X) // SQUARE_SIZE
                    row = (y - BOARD_OFFSET_Y) // SQUARE_SIZE
                    piece = game.board[row][col]
                    if piece:
                        hovered_text = movement_descriptions.get(piece[1])
                        hover_pos = (x, y)
                    else:
                        hovered_text = None
                else:
                    hovered_text = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if reset_button_rect.collidepoint(x, y):
                    game.reset_game()
                    selected = None
                    selected_empty_square = None
                    continue

                if not game.game_over and game.turn == 'w':
                    if BOARD_OFFSET_X <= x < BOARD_OFFSET_X + game.COLS * SQUARE_SIZE and \
                       BOARD_OFFSET_Y <= y < BOARD_OFFSET_Y + game.ROWS * SQUARE_SIZE:
                        col = (x - BOARD_OFFSET_X) // SQUARE_SIZE
                        row = (y - BOARD_OFFSET_Y) // SQUARE_SIZE
                        if selected:
                            if game.is_valid_move(selected, (row, col)):
                                game.add_log_message(f"Moved piece from {selected} to {(row, col)}.")
                                if move_sound: move_sound.play()
                                game.move_piece(selected, (row, col))
                                game.last_cpu_action = None
                            selected = None
                            selected_empty_square = None
                        elif game.board[row][col] == '':
                            if game.turn == 'w' and row >= game.ROWS - 3:
                                selected_empty_square = (row, col)
                                selected = None
                                game.add_log_message("Select a piece to buy.")
                            else:
                                game.add_log_message("You can only place pieces here.")
                                selected_empty_square = None
                                selected = None
                        elif game.board[row][col] and game.board[row][col][0] == game.turn:
                            selected = (row, col)
                            selected_empty_square = None
                            game.add_log_message(f"Selected {game.board[row][col]} at {(row, col)}.")
                        else:
                            selected = None
                            selected_empty_square = None
                            game.add_log_message("What are you doing?")
                    else:
                        if selected_empty_square:
                            if buy_normal_button.collidepoint(x, y):
                                if game.buy_piece('normal', selected_empty_square):
                                    selected_empty_square = None; selected = None; game.last_cpu_action = None
                            elif buy_rare_button.collidepoint(x, y):
                                if game.buy_piece('rare', selected_empty_square):
                                    selected_empty_square = None; selected = None; game.last_cpu_action = None
                            elif buy_epic_button.collidepoint(x, y):
                                if game.buy_piece('epic', selected_empty_square):
                                    selected_empty_square = None; selected = None; game.last_cpu_action = None
                            else:
                                selected_empty_square = None; selected = None; game.add_log_message("You cancel to buy?")
                        else:
                            if action_button_rect.collidepoint(x, y):
                                if selected and game.board[selected[0]][selected[1]] and \
                                   game.board[selected[0]][selected[1]][0] == game.turn and \
                                   game.board[selected[0]][selected[1]][1] in game.class_change_map:
                                    if game.class_change(selected):
                                        selected = None; game.last_cpu_action = None
                                else:
                                    if game.hire_employee():
                                        game.last_cpu_action = None
                                selected = None
                                selected_empty_square = None
                            else:
                                selected = None
                                selected_empty_square = None
        clock.tick(60)

    pygame.quit()
    sys.exit()

# --- 実行 ---
if __name__ == "__main__":
    run_chess_game()