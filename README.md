# web_app_202505

# 開発手順
分からないことは@hikaru122700に聞いてください

初回のみ、自分のパソコンにクローンします。

どの階層で作業をするのか慎重に決めてください。（僕はこの作業で親レポジトリが破壊されました。）

```bash
git clone https://github.com/hikaru122700/web_app_202505.git
cd .\web_app_202505
```

クローンされた環境で開発を行います。実行するにはクローンされた環境を信頼することが必要です。

---

開発が一段落したら、以下の手順でpushします。

```bash
git add .
git commit -m "メッセージ"
```

”メッセージ”の部分はどのような変更をしたのかわかるように記述してください

```bash
git push origin main
```

---

最新のmainブランチの内容を取得したいとき

以下のエラーメッセージが出たとき

```bash
! [rejected]        main -> main (fetch first)
Updates were rejected because the remote contains work that you do not have locally
```

もしくは最新のmainブランチの内容を取得したいときは以下を実行

```bash
git pull origin main
```


以下のファイルはstreamlitで構成されています。

- X.py
- number_magic.py

実行時には以下のコマンドを入力してください。

```bash
streamlit run [filename.py]
```

以下のファイルはpygameで構成されています。

- chess.py

実行時には以下のコマンドを入力してください。

```bash
python chess.py
```

# chessルール

ポーンが相手陣地の最奥列に到達すると自動的にクイーンへ昇格します。

任意の駒を移動させる・駒を買い出現させるの中から行動を選択してください。

持ち金は左下に表示されます。

