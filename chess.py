import pygame
import sys
import os
import random
import time

# --- Monte Carlo simulation helper ---
class SimGame:
    def __init__(self, game):
        self.ROWS = game.ROWS
        self.COLS = game.COLS
        self.board = [row[:] for row in game.board]
        self.turn = game.turn
        self.white_money = game.white_money
        self.black_money = game.black_money
        self.white_income_per_turn = game.white_income_per_turn
        self.black_income_per_turn = game.black_income_per_turn
        self.buy_options = game.buy_options
        self.class_change_map = game.class_change_map
        self.game_over = game.game_over
        self.winner = game.winner

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
        piece = self.board[sr][sc]

        if not (0 <= er < self.ROWS and 0 <= ec < self.COLS):
            return False

        if not piece or piece[0] != self.turn:
            return False

        target = self.board[er][ec]
        if target and target[0] == self.turn:
            return False

        if piece[1] == 'p':
            direction = -1 if self.turn == 'w' else 1
            if sc == ec and self.board[er][ec] == '' and er == sr + direction:
                return True
            if sc == ec and self.board[er][ec] == '' and (
                (self.turn == 'w' and sr == self.ROWS - 2 and er == sr - 2 and self.board[sr - 1][sc] == '') or
                (self.turn == 'b' and sr == 1 and er == sr + 2 and self.board[sr + 1][sc] == '')
            ):
                return True
            if abs(sc - ec) == 1 and er == sr + direction and target and target[0] != self.turn:
                return True
            return False
        elif piece[1] == 'r':
            if (sr == er or sc == ec) and self.is_path_clear(start, end):
                return True
            return False
        elif piece[1] == 'n':
            dr = abs(sr - er)
            dc = abs(sc - ec)
            if (dr == 1 and dc == 2) or (dr == 2 and dc == 1):
                return True
            return False
        elif piece[1] == 'b':
            if abs(sr - er) == abs(sc - ec) and self.is_path_clear(start, end):
                return True
            return False
        elif piece[1] == 'q':
            if ((sr == er or sc == ec) or (abs(sr - er) == abs(sc - ec))) and self.is_path_clear(start, end):
                return True
            return False
        elif piece[1] == 'k':
            dr = abs(sr - er)
            dc = abs(sc - ec)
            if dr <= 1 and dc <= 1:
                return True
            return False

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
        self.board[er][ec] = self.board[sr][sc]
        self.board[sr][sc] = ''
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
        cost = 20
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
        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = self.board[r][c]
                if piece and piece[0] == self.turn:
                    for er in range(self.ROWS):
                        for ec in range(self.COLS):
                            if self.is_valid_move((r, c), (er, ec)):
                                actions.append({'type': 'move', 'start': (r, c), 'end': (er, ec)})
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
        cost_hire = 20
        current_money = self.white_money if self.turn == 'w' else self.black_money
        if current_money >= cost_hire:
            actions.append({'type': 'hire'})
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

        self.piece_rewards = {
            'p': 10, 'r': 30, 'n': 30, 'b': 30, 'q': 30, 'k': 0
        }
        self.buy_options = {
            'normal': {'cost': 10, 'pieces': [('p', 0.90), ('r', 0.033), ('n', 0.033), ('b', 0.033), ('q', 0.001)]},
            'rare': {'cost': 20, 'pieces': [('p', 0.50), ('r', 0.15), ('n', 0.15), ('b', 0.15), ('q', 0.05)]},
            'epic': {'cost': 30, 'pieces': [('p', 0.10), ('r', 0.25), ('n', 0.25), ('b', 0.25), ('q', 0.15)]}
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
        # 実行中のスクリプトのディレクトリを取得
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dog_folder_path = os.path.join(current_dir, "dog")

        if not os.path.exists(dog_folder_path):
            print(f"警告: 'dog' フォルダが見つかりません: {dog_folder_path}")
            return

        # 想定される行動と画像ファイル名のマッピング (ここに画像名を追加/変更)
        # 例えば、'move.png' がない場合は 'default.png' が使われるように fallback を考慮することも可能
        action_image_map = {
            'default': 'default.png', # デフォルト画像 (必須)
            'move': 'move.png',       # 駒を動かした時
            'buy': 'buy.png',         # 駒を購入した時
            'hire': 'hire.png',       # 従業員を雇用した時
            'class_change': 'class_change.png', # クラス変更した時
            'win': 'win.png',         # 勝利時 (UIで直接描画するため使わないかもしれないが、念のため)
            'lose': 'lose.png',       # 敗北時
            'thinking': 'thinking.png', # CPU思考中
            'happy': 'happy.png',     # ポジティブなイベント
            'sad': 'sad.png',         # ネガティブなイベント
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
        
        # デフォルト画像がロードされていることを確認
        if 'default' not in self.dog_images_dict:
            print("エラー: 'default.png' が 'dog' フォルダに見つかりませんでした。デフォルト画像は必須です。")
            # デフォルト画像がない場合は、何らかのプレースホルダーを設定するか、エラー終了
            # ここでは簡単なエラー表示用のダミー画像を生成
            dummy_surface = pygame.Surface(self.dog_image_display_size, pygame.SRCALPHA)
            dummy_surface.fill((255, 0, 0, 128)) # 赤色の半透明
            pygame.draw.circle(dummy_surface, (255, 255, 255), (self.dog_image_display_size[0]//2, self.dog_image_display_size[1]//2), self.dog_image_display_size[0]//2 - 10)
            self.dog_images_dict['default'] = dummy_surface

    def set_dog_image(self, key):
        if key in self.dog_images_dict:
            self.current_dog_image_key = key
        elif 'default' in self.dog_images_dict: # キーが見つからない場合はデフォルトに戻す
            self.current_dog_image_key = 'default'
        else: # デフォルトもなければ何もしない（エラーになる可能性あり）
            self.current_dog_image_key = None # 何も表示しない

    def get_current_dog_image(self):
        return self.dog_images_dict.get(self.current_dog_image_key)

    def reset_game(self):
        """Reset all game state to start a new game."""
        ChessGame.__init__(self)
        self.set_dog_image('default')


    def add_log_message(self, message):
        self.game_log_messages.append(message)
        if len(self.game_log_messages) > 2: # 最新2行を表示 (ここは調整してください)
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
        piece = self.board[sr][sc]
        
        if not (0 <= er < self.ROWS and 0 <= ec < self.COLS):
            return False

        if not piece or piece[0] != self.turn:
            return False
        
        target = self.board[er][ec]
        if target and target[0] == self.turn:
            return False

        # ポーンの移動ロジック
        if piece[1] == 'p':
            direction = -1 if self.turn == 'w' else 1
            if sc == ec and self.board[er][ec] == '' and er == sr + direction: return True
            if sc == ec and self.board[er][ec] == '' and \
               ((self.turn == 'w' and sr == self.ROWS - 2 and er == sr - 2 and self.board[sr - 1][sc] == '') or \
                (self.turn == 'b' and sr == 1 and er == sr + 2 and self.board[sr + 1][sc] == '')): return True
            if abs(sc - ec) == 1 and er == sr + direction and target and target[0] != self.turn: return True
            return False
        # ルークの移動ロジック
        elif piece[1] == 'r':
            if (sr == er or sc == ec) and self.is_path_clear(start, end): return True
            return False
        # ナイトの移動ロジック
        elif piece[1] == 'n':
            dr = abs(sr - er)
            dc = abs(sc - ec)
            if (dr == 1 and dc == 2) or (dr == 2 and dc == 1): return True
            return False
        # ビショップの移動ロジック
        elif piece[1] == 'b':
            if abs(sr - er) == abs(sc - ec) and self.is_path_clear(start, end): return True
            return False
        # クイーンの移動ロジック
        elif piece[1] == 'q':
            if ((sr == er or sc == ec) or (abs(sr - er) == abs(sc - ec))) and self.is_path_clear(start, end): return True
            return False
        # キングの移動ロジック
        elif piece[1] == 'k':
            dr = abs(sr - er)
            dc = abs(sc - ec)
            if dr <= 1 and dc <= 1: return True
            return False
        
        return False

    def end_turn(self):
        # ターンプレイヤーの所持金を増やす
        if self.turn == 'w':
            self.white_money += self.white_income_per_turn
        else:
            self.black_money += self.black_income_per_turn
        
        # ターンを交代
        self.turn = 'b' if self.turn == 'w' else 'w'
        self.add_log_message(f"{'white' if self.turn == 'w' else 'black'} turn")
        self.set_dog_image('default') # ターン終了時はデフォルト画像に戻す

    def move_piece(self, start, end):
        sr, sc = start
        er, ec = end
        
        captured_piece_full_name = self.board[er][ec]
        
        reward = 0
        if captured_piece_full_name:
            captured_piece_type = captured_piece_full_name[1]
            if captured_piece_type == 'k':
                reward = 10 
                self.add_log_message(f"King captured!") # キング奪取のログ
                self.set_dog_image('happy') # キング取ったらハッピー
            else:
                reward = 1 
                self.set_dog_image('move') # 駒を取る移動
        else:
            self.set_dog_image('move') # 駒を取らない移動

        self.board[er][ec] = self.board[sr][sc]
        self.board[sr][sc] = ''

        self.check_for_king_capture()
        if self.game_over:
            self.add_log_message(f"Finish! Winner is {'white' if self.winner == 'w' else 'black'}!")
            # 勝利/敗北の画像を設定
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
                
                self.add_log_message(f"{'White' if self.turn == 'w' else 'Black'} choice {option_type}, and buy {drawn_piece_type}.")
                self.set_dog_image('buy') # 購入行動
                self.end_turn()
                return True
        return False

    def hire_employee(self):
        cost = 20
        current_money = self.white_money if self.turn == 'w' else self.black_money
        if current_money >= cost:
            if self.turn == 'w':
                self.white_money -= cost
                self.white_income_per_turn += 2
            else:
                self.black_money -= cost
                self.black_income_per_turn += 2
            self.add_log_message(f"You hire employee!")
            self.set_dog_image('buy') # 雇用行動
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
                self.add_log_message(f"{'White' if self.turn == 'w' else 'Black'}'s {original_type} is changed class to {new_type}.")
                self.set_dog_image('buy') # クラス変更行動
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

    # --- AI: CPUのターン処理 ---
    def cpu_play_turn(self):
        self.set_dog_image('thinking') # CPU思考中の画像
        possible_actions = []

        # 1. 駒の移動
        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = self.board[r][c]
                if piece and piece[0] == self.turn:
                    for er in range(self.ROWS):
                        for ec in range(self.COLS):
                            if self.is_valid_move((r, c), (er, ec)):
                                possible_actions.append({'type': 'move', 'start': (r, c), 'end': (er, ec)})
        
        # 2. 駒の購入
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
                                possible_actions.append({'type': 'buy', 'option': option_type, 'target_pos': (r, c)})

        # 3. 従業員を雇用
        cost_hire = 20
        current_money = self.white_money if self.turn == 'w' else self.black_money
        if current_money >= cost_hire:
            possible_actions.append({'type': 'hire'})

        # 4. クラス変更
        cost_class_change = 70
        if current_money >= cost_class_change:
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    piece = self.board[r][c]
                    if piece and piece[0] == self.turn and piece[1] in self.class_change_map:
                        possible_actions.append({'type': 'class_change', 'pos': (r, c)})

        if not possible_actions:
            self.add_log_message(f"No valid actions for {'white' if self.turn == 'w' else 'black'} CPU.")
            self.end_turn()
            return

        def can_capture_king(state, attacker):
            original_turn = state.turn
            state.turn = attacker
            target = ('w' if attacker == 'b' else 'b') + 'k'
            for r in range(state.ROWS):
                for c in range(state.COLS):
                    piece = state.board[r][c]
                    if piece and piece[0] == attacker:
                        for er in range(state.ROWS):
                            for ec in range(state.COLS):
                                if state.is_valid_move((r, c), (er, ec)) and state.board[er][ec] == target:
                                    state.turn = original_turn
                                    return True
            state.turn = original_turn
            return False

        opponent_color = 'b' if self.turn == 'w' else 'w'
        in_danger = can_capture_king(SimGame(self), opponent_color)

        if in_danger:
            safe_actions = []
            for action in possible_actions:
                sim = SimGame(self)
                sim.apply_action(action)
                if not can_capture_king(sim, sim.turn):
                    safe_actions.append(action)

            if safe_actions:
                possible_actions = safe_actions
            else:
                chosen_action = random.choice(possible_actions)
                self.add_log_message(f"CPU chooses: {chosen_action['type']}")
                if chosen_action['type'] == 'move':
                    self.move_piece(chosen_action['start'], chosen_action['end'])
                elif chosen_action['type'] == 'buy':
                    self.buy_piece(chosen_action['option'], chosen_action['target_pos'])
                elif chosen_action['type'] == 'hire':
                    self.hire_employee()
                elif chosen_action['type'] == 'class_change':
                    self.class_change(chosen_action['pos'])
                return


        # --- CPU: Monte Carlo search ---
        end_time = time.time() + 2
        wins = [0] * len(possible_actions)
        sims = [0] * len(possible_actions)
        idx = 0
        while time.time() < end_time:
            action_index = idx % len(possible_actions)
            idx += 1
            sim_game = SimGame(self)
            sim_game.apply_action(possible_actions[action_index])
            steps = 0
            while not sim_game.game_over and steps < 40:
                actions = sim_game.get_possible_actions()
                if not actions:
                    sim_game.end_turn()
                    steps += 1
                    continue
                rand_action = random.choice(actions)
                sim_game.apply_action(rand_action)
                steps += 1
            if sim_game.game_over and sim_game.winner == self.turn:
                wins[action_index] += 1
            sims[action_index] += 1

        best_rate = -1
        best_indices = []
        for i in range(len(possible_actions)):
            rate = wins[i] / sims[i] if sims[i] > 0 else 0
            if rate > best_rate:
                best_rate = rate
                best_indices = [i]
            elif rate == best_rate:
                best_indices.append(i)

        chosen_action = possible_actions[random.choice(best_indices)]

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

    # 音楽と効果音のパス
    music_path = "sound/ohirusugi.mp3"
    move_sound_path = "sound/kon.mp3"

    # BGMのロードと再生
    current_dir = os.path.dirname(os.path.abspath(__file__)) # ここでcurrent_dirを定義
    
    if os.path.exists(os.path.join(current_dir, music_path)):
        pygame.mixer.music.load(os.path.join(current_dir, music_path))
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play(-1)
    else:
        print(f"警告: BGMファイル '{os.path.join(current_dir, music_path)}' が見つかりません。BGMは再生されません。")

    # 効果音のロード
    move_sound = None
    if os.path.exists(os.path.join(current_dir, move_sound_path)):
        move_sound = pygame.mixer.Sound(os.path.join(current_dir, move_sound_path))
        move_sound.set_volume(0.7)
    else:
        print(f"警告: 効果音ファイル '{os.path.join(current_dir, move_sound_path)}' が見つかりません。移動音は再生されません。")

    # ゲーム設定
    WIDTH, HEIGHT = 1000, 600
    SQUARE_SIZE = 64
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess Game")

    # 色
    LIGHT = (240, 217, 181)
    DARK = (181, 136, 99)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    WHITE_TEXT = (255, 255, 255)
    BLACK_TEXT = (0, 0, 0)
    GRAY_OVERLAY = (100, 100, 100, 128)
    UI_BACKGROUND_COLOR = (50, 50, 50)
    MOVE_HIGHLIGHT = (144, 238, 144, 160)  # Light green with transparency

    # フォント設定
    pygame.font.init()
    font = pygame.font.SysFont('arial', 48)
    money_font = pygame.font.SysFont('arial', 20)
    buy_font = pygame.font.SysFont('arial', 23)
    button_font = pygame.font.SysFont('arial', 18)
    turn_font = pygame.font.SysFont('arial', 23)
    log_font = pygame.font.SysFont('arial', 23)

    # 背景画像の取得 (haikei.png はUIの背景として残す)
    bg_path = "haikei.png"
    bg = None
    if os.path.exists(os.path.join(current_dir, bg_path)):
        bg = pygame.image.load(os.path.join(current_dir, bg_path)).convert_alpha()
        bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
        rect_bg = bg.get_rect(topleft=(0, 0))
    else:
        print(f"警告: 背景画像ファイル '{os.path.join(current_dir, bg_path)}' が見つかりません。背景は表示されません。")
        
    # チェス盤のオフセット
    BOARD_OFFSET_X = 120
    BOARD_OFFSET_Y = 120

    # 駒画像読み込み
    piece_names = ['wp', 'wr', 'wn', 'wb', 'wq', 'wk',
                    'bp', 'br', 'bn', 'bb', 'bq', 'bk']
    pieces = {}
    for name in piece_names:
        path = os.path.join(current_dir, "chess_pieces", f"{name}.png")
        try:
            if not os.path.exists(path):
                print(f"エラー: 画像ファイル '{path}' が見つかりません。")
                print("`chess_pieces`フォルダにすべての駒の画像がダウンロードされていることを確認してください。")
                sys.exit(1)
            img = pygame.image.load(path)
            pieces[name] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        except pygame.error as e:
            print(f"画像 '{path}' の読み込み中にエラーが発生しました: {e}")
            print("画像ファイルが破損していないか確認してください。")
            sys.exit(1)

    # ゲームオブジェクトのインスタンス化
    game = ChessGame()
    game.set_dog_image('default') # 初期表示はデフォルト画像

    selected = None
    selected_empty_square = None

    clock = pygame.time.Clock()

    # UIエリアの開始X座標
    UI_AREA_START_X = BOARD_OFFSET_X + game.COLS * SQUARE_SIZE + 30
    turn_display_rect = pygame.Rect(UI_AREA_START_X, 40, WIDTH - UI_AREA_START_X - 260, 36) 

    # ゲームログ表示エリア
    log_display_rect = pygame.Rect(UI_AREA_START_X, turn_display_rect.bottom + 5, WIDTH - UI_AREA_START_X - 20, 80) 
    
    # 犬画像表示位置 
    dog_image_pos_x = WIDTH - game.dog_image_display_size[0] - 20 # 右端から20px
    dog_image_pos_y = HEIGHT - game.dog_image_display_size[1] - 20 # 下端から20px
    dog_image_display_rect = pygame.Rect(500, 110, 10, 60)

    # 所持金表示エリア
    money_bg_rect = pygame.Rect(UI_AREA_START_X + 200, HEIGHT - 90, WIDTH - UI_AREA_START_X - 210, 80)


    # ボタン配置
    button_area_center_y = HEIGHT - 45

    # Hire Employee / Class Change ボタンの位置
    action_button_rect = pygame.Rect(0, 0, 140, 50)
    action_button_rect.center = (UI_AREA_START_X + 100, button_area_center_y)

    # 購入オプションボタンの定義
    buy_normal_button = pygame.Rect(0, 0, 120, 30)
    buy_normal_button.center = (UI_AREA_START_X + 100, button_area_center_y + 20)
    
    buy_rare_button = pygame.Rect(0, 0, 120, 30)
    buy_rare_button.center = (UI_AREA_START_X + 100, button_area_center_y - 20)
    
    buy_epic_button = pygame.Rect(0, 0, 120, 30)
    buy_epic_button.center = (UI_AREA_START_X + 100, button_area_center_y - 60)

    # リセットボタンの定義
    reset_button_rect = pygame.Rect(0, 0, 120, 30)
    reset_button_initial_center = (UI_AREA_START_X + 100, button_area_center_y - 100)
    reset_button_rect.center = reset_button_initial_center

    def draw_game():
        if game.game_over:
            reset_button_rect.center = (WIDTH // 2, HEIGHT // 2 + 60)
        else:
            reset_button_rect.center = reset_button_initial_center

        screen.fill(UI_BACKGROUND_COLOR)

        # haikei.png を描画
        if bg:
            screen.blit(bg, rect_bg)
        
        # チェス盤の描画
        for row in range(game.ROWS):
            for col in range(game.COLS):
                color = LIGHT if (row + col) % 2 == 0 else DARK
                pygame.draw.rect(screen, color, (col*SQUARE_SIZE + BOARD_OFFSET_X, row*SQUARE_SIZE + BOARD_OFFSET_Y, SQUARE_SIZE, SQUARE_SIZE))
                piece = game.board[row][col]
                if piece:
                    screen.blit(pieces[piece], (col*SQUARE_SIZE + BOARD_OFFSET_X, row*SQUARE_SIZE + BOARD_OFFSET_Y))
        # 駒が選択されている場合は移動可能マスをハイライト
        if selected and not game.game_over:
            for er in range(game.ROWS):
                for ec in range(game.COLS):
                    if game.is_valid_move(selected, (er, ec)):
                        highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                        highlight_surface.fill(MOVE_HIGHLIGHT)
                        screen.blit(highlight_surface, (ec*SQUARE_SIZE + BOARD_OFFSET_X, er*SQUARE_SIZE + BOARD_OFFSET_Y))

            pygame.draw.rect(screen, GREEN, (selected[1]*SQUARE_SIZE + BOARD_OFFSET_X, selected[0]*SQUARE_SIZE + BOARD_OFFSET_Y, SQUARE_SIZE, SQUARE_SIZE), 3)
        
        # 選択中の空マス
        if selected_empty_square and not game.game_over:
            pygame.draw.rect(screen, (0, 150, 255), (selected_empty_square[1]*SQUARE_SIZE + BOARD_OFFSET_X, selected_empty_square[0]*SQUARE_SIZE + BOARD_OFFSET_Y, SQUARE_SIZE, SQUARE_SIZE), 3)

        # UI要素の描画
        pygame.draw.rect(screen, (70, 70, 70), turn_display_rect)
        turn_text = turn_font.render(f"{'white' if game.turn == 'w' else 'black'} turn", True, WHITE_TEXT)
        turn_text_rect = turn_text.get_rect(center=turn_display_rect.center)
        screen.blit(turn_text, turn_text_rect)

        pygame.draw.rect(screen, (30, 30, 30), log_display_rect)
        for i, msg in enumerate(game.game_log_messages):
            log_text_surface = log_font.render(msg, True, WHITE_TEXT)
            screen.blit(log_text_surface, (log_display_rect.x + 10, log_display_rect.y + 10 + i * 20))

        current_dog_img = game.get_current_dog_image()
        if current_dog_img:
            # 所持金表示エリアの近くに描画するため、dog_image_display_rect を調整
            # money_bg_rect の位置を基準に、その背面に配置する
            # 例えば、money_bg_rect の左上を基準に、少しずらして配置する
            # 描画位置は game.dog_image_display_size に合わせて調整してください
            
            # 例: money_bg_rect の左上を基準に少し左上に配置
            # dog_image_display_rect の定義を以下のように変更します
            # money_bg_rect がUI_AREA_START_X + 200 から始まるので、そのあたりに犬を配置
            # 犬の画像の表示位置を調整します
            # ここでは money_bg_rect の左下隅に犬画像の右下隅が来るように配置してみます
            dog_rect_for_drawing = current_dog_img.get_rect(
                bottomright=(money_bg_rect.x + money_bg_rect.width - 10, money_bg_rect.y + money_bg_rect.height - 10)
            )
            # もしくは、シンプルに dog_image_display_rect を使って所持金表示エリアの隣に配置し、
            # 描画順序で背面に持っていく
            screen.blit(current_dog_img, dog_image_display_rect.topleft) # この行のまま

        # 所持金表示の半透明オーバーレイを描画
        money_overlay = pygame.Surface((money_bg_rect.width, money_bg_rect.height), pygame.SRCALPHA)
        money_overlay.fill(GRAY_OVERLAY)
        screen.blit(money_overlay, money_bg_rect.topleft)

        # 所持金テキストを描画
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
            elif selected and game.board[selected[0]][selected[1]] and game.board[selected[0]][selected[1]][0] == game.turn:
                piece_at_selected = game.board[selected[0]][selected[1]]
                if piece_at_selected[1] in game.class_change_map:
                    draw_button(screen, action_button_rect, f"Class Change (70G)", (100, 150, 200), WHITE_TEXT, button_font)
                else:
                    draw_button(screen, action_button_rect, f"Hire Employee (20G)", (50, 150, 50), WHITE_TEXT, button_font)
            else:
                draw_button(screen, action_button_rect, f"Hire Employee (20G)", (50, 150, 50), WHITE_TEXT, button_font)

        if game.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))

            text_surface = font.render(f"{'white' if game.winner == 'w' else 'black'} win!", True, RED)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(text_surface, text_rect)
            draw_button(screen, reset_button_rect, "Reset", (180, 80, 80), WHITE_TEXT, button_font)

    def draw_button(surface, rect, text, color, text_color, font_obj):
        pygame.draw.rect(surface, color, rect)
        text_surface = font_obj.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)


    # ゲームループ
    running = True
    while running:
        draw_game()  # まず盤面を描画してユーザー操作を反映
        pygame.display.flip()

        # AI: CPUのターン処理
        if not game.game_over and game.turn == 'b':
            pygame.time.wait(1000)
            game.cpu_play_turn()
            selected = None
            selected_empty_square = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
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
                                game.move_piece(selected, (row, col))
                            selected = None
                            selected_empty_square = None
                        elif game.board[row][col] == '':
                            if game.turn == 'w' and row >= game.ROWS - 3: 
                                selected_empty_square = (row, col)
                                selected = None
                                game.add_log_message("Select a piece to buy.") # 購入準備中のメッセージ
                            else:
                                game.add_log_message("You can only place pieces within 3 squares of your own territory.")
                                selected_empty_square = None
                                selected = None
                        elif game.board[row][col] and game.board[row][col][0] == game.turn:
                            selected = (row, col)
                            selected_empty_square = None
                            game.add_log_message(f"Selected {game.board[row][col]} at {(row, col)}.") # 駒選択のログ
                        else:
                            selected = None
                            selected_empty_square = None
                            game.add_log_message("What are you doing?")

                    else: # UIエリアのクリック
                        if selected_empty_square:
                            if buy_normal_button.collidepoint(x, y):
                                if game.buy_piece('normal', selected_empty_square):
                                    selected_empty_square = None
                                    selected = None
                            elif buy_rare_button.collidepoint(x, y):
                                if game.buy_piece('rare', selected_empty_square):
                                    selected_empty_square = None
                                    selected = None
                            elif buy_epic_button.collidepoint(x, y):
                                if game.buy_piece('epic', selected_empty_square):
                                    selected_empty_square = None
                                    selected = None
                            else:
                                selected_empty_square = None
                                selected = None
                                game.add_log_message("You cancel to buy?")
                        else:
                            if action_button_rect.collidepoint(x, y):
                                if selected and game.board[selected[0]][selected[1]] and \
                                   game.board[selected[0]][selected[1]][0] == game.turn and \
                                   game.board[selected[0]][selected[1]][1] in game.class_change_map:
                                    if game.class_change(selected):
                                        selected = None
                                else:
                                    game.hire_employee()
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
