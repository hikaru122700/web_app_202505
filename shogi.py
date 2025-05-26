import streamlit as st

# 駒のユニコードマッピング
def get_icon(piece):
    ICONS = {
        'P': '歩', 'L': '香', 'N': '桂', 'S': '銀', 'G': '金', 'B': '角', 'R': '飛', 'K': '王',
        # 成り駒
        '+P': 'と', '+L': '杏', '+N': '圭', '+S': '全', '+B': '馬', '+R': '龍'
    }
    return ICONS.get(piece, '　') if piece else '　'

# 初期盤面を生成
def init_board():
    return [
        ['L','N','S','G','K','G','S','N','L'],
        [None,'R',None,None,None,None,None,'B',None],
        ['P']*9,
        [None]*9,
        [None]*9,
        [None]*9,
        ['P']*9,
        [None,'B',None,None,None,None,None,'R',None],
        ['L','N','S','G','K','G','S','N','L'],
    ]

# 移動可能判定
def is_valid_move(piece, sr, sc, r, c, board):
    if piece is None:
        return False
    dr, dc = r - sr, c - sc
    # 一歩ずつ動く駒用ディレクトリ
    gold_dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1)]
    silver_dirs = [(-1,-1),(-1,0),(-1,1),(1,-1),(1,1)]

    # 非成駒
    if piece == 'P':
        return (dr, dc) == (-1, 0)
    if piece == 'G':
        return (dr, dc) in gold_dirs
    if piece == 'S':
        return (dr, dc) in silver_dirs
    if piece == 'N':
        return (dr, dc) in [(-2,-1),(-2,1)]
    if piece == 'K':
        return abs(dr) <= 1 and abs(dc) <= 1
    # 成駒（金に同じ動き）
    if piece in ['+P','+L','+N','+S']:
        return (dr, dc) in gold_dirs

    # 飛車（直線スライド）
    def slide_check(dr, dc):
        step_r = 0 if dr == 0 else dr//abs(dr)
        step_c = 0 if dc == 0 else dc//abs(dc)
        r0, c0 = sr + step_r, sc + step_c
        while (r0, c0) != (r, c):
            if board[r0][c0] is not None:
                return False
            r0 += step_r; c0 += step_c
        return True

    if piece == 'R':
        if (dr == 0 and dc != 0) or (dc == 0 and dr != 0):
            return slide_check(dr, dc)
        return False
    # 角（斜めスライド）
    if piece == 'B':
        if abs(dr) == abs(dc) and dr != 0:
            return slide_check(dr, dc)
        return False
    # 香（縦スライド）
    if piece == 'L':
        if dc == 0 and dr != 0:
            return slide_check(dr, dc)
        return False

    # 成駒：龍（飛+王）
    if piece == '+R':
        if abs(dr) <= 1 and abs(dc) <= 1:
            return True
        return is_valid_move('R', sr, sc, r, c, board)
    # 成駒：馬（角+王）
    if piece == '+B':
        if abs(dr) <= 1 and abs(dc) <= 1:
            return True
        return is_valid_move('B', sr, sc, r, c, board)

    return False

# セッションステートの初期化
if 'board' not in st.session_state:
    st.session_state.board = init_board()
    st.session_state.selected = None
    st.session_state.hands = []  # 捕獲した駒を保持

st.title("Streamlit 将棋")

# 駒をクリックしたときの処理
def click_cell(r, c):
    board = st.session_state.board
    sel = st.session_state.selected

    # 駒を選択していない場合
    if sel is None:
        if board[r][c] is not None:
            st.session_state.selected = (r, c)
    # 駒を選択している場合 → 移動 or キャプチャ
    else:
        sr, sc = sel
        piece = board[sr][sc]
        # 同じマスならキャンセル
        if (sr, sc) == (r, c):
            st.session_state.selected = None
            return
        # 移動可否チェック
        if not is_valid_move(piece, sr, sc, r, c, board):
            st.warning("その駒はその位置に移動できません。")
            st.session_state.selected = None
            return
        # キャプチャ処理
        target = board[r][c]
        if target is not None:
            st.session_state.hands.append(target)
        # 移動実行
        board[r][c] = piece
        board[sr][sc] = None
        st.session_state.selected = None

# リセットボタン
if st.button("リセット"):
    st.session_state.board = init_board()
    st.session_state.selected = None
    st.session_state.hands = []

# 持ち駒表示
if st.session_state.hands:
    hand_str = " " .join([get_icon(p) for p in st.session_state.hands])
    st.sidebar.markdown("**持ち駒:**")
    st.sidebar.write(hand_str)

# 選択中の駒情報表示
if st.session_state.selected:
    sr, sc = st.session_state.selected
    st.markdown(f"**選択中:** ({sr+1}, {sc+1}) {get_icon(st.session_state.board[sr][sc])}")

# 盤面表示（9x9 ボタン）
for r in range(9):
    cols = st.columns(9)
    for c, col in enumerate(cols):
        label = get_icon(st.session_state.board[r][c])
        if col.button(label, key=f"{r}-{c}"):
            click_cell(r, c)
