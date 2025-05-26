import streamlit as st
# streamlit run shogi.py


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

# セッションステートの初期化
if 'board' not in st.session_state:
    st.session_state.board = init_board()
    st.session_state.selected = None

st.title("Streamlit 将棋")

# 駒をクリックしたときの処理
def click_cell(r, c):
    board = st.session_state.board
    sel = st.session_state.selected

    # 駒を選択していない場合
    if sel is None:
        if board[r][c] is not None:
            st.session_state.selected = (r, c)
    # 既に駒を選択している場合 → 移動
    else:
        sr, sc = sel
        # 移動先が同じならキャンセル
        if (sr, sc) != (r, c):
            board[r][c] = board[sr][sc]
            board[sr][sc] = None
        st.session_state.selected = None

# リセットボタン
if st.button("リセット"):
    st.session_state.board = init_board()
    st.session_state.selected = None

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
