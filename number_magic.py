import streamlit as st
import random
import base64


st.title("数当てゲーム (Guess the Number)")
# 背景画像をローカルファイルから設定する関数
def add_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# 実行時に背景を設定（画像パスは適宜変更）
add_bg_from_local("/data/Gemini_Generated_Image_sylbb3sylbb3sylb.png")

# ゲームの初期化
def init_game():
    # st.session_state.secret = random.randint(1, 100)
    st.session_state.secret = 1
    st.session_state.count = 0
    st.session_state.finished = False
    st.session_state.message = "数字を当ててください."
    # 入力フォームの値をリセット（キーが存在しない場合はNumberInputが初期化時に設定）
    st.session_state.input_guess = 1

# 初回またはリセット後の初期化
if 'secret' not in st.session_state:
    init_game()

# 入力ウィジェット
guess = st.number_input(
    "あなたの予想:", min_value=1, max_value=100, step=1,
    key='input_guess'
)

# 予想ボタン（ゲーム終了後は無効化）
if st.button("予想する", disabled=st.session_state.finished):
    st.session_state.count += 1
    if guess < st.session_state.secret:
        st.session_state.message = f"{st.session_state.count}回目の占いの結果もっと大きいです。"
    elif guess > st.session_state.secret:
        st.session_state.message = f"{st.session_state.count}回目の占いの結果もっと小さいです。"
    else:
        st.session_state.message = f"🎉 正解！ {st.session_state.count}回目で当たりました！"
        st.session_state.finished = True

# ヒント・結果表示
st.write(st.session_state.message)

# リセットボタン（コールバックで再初期化）
st.button("リセット", on_click=init_game)