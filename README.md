# sound_recognition

tensorflowでcnnを使って音声認識をするものをつくりました。

入力がtrueかfalseか判定するだけの二値分類器です。


### 使用したバージョン

`python 3.5.2`

`tensorflow 1.2.1`


## 使い方

### データセット

データセットが格納されているディレクトリを`wavdataset.py`に教えてあげます。

具体的には、`wavdataset.py`の`DatasetManager.true_dir_list`には正解データが入っているディレクトリのパスを、
`false_dir_list`には不正解データが入っているディレクトリのパスを、それぞれ追加します。

`wavdataset.py`はデータセットを管理するモジュールです。

`wavdataset.py`は、指定されたディレクトリからwav形式の音声ファイルを読み込んでフーリエ変換し、
変換後のデータを集め、これをデータセットとして、他のモジュールに提供します。



### 学習

データセットを使って学習させたいときはpythonインタプリタで`training.py`を実行します。

`training.py`はデータセットを取得し、これを教師データとしてcnnに与えて特徴を学習させます。

学習が終わると、得られたパラメータを保存します。

学習パラメータは`sound_recognition_model.なんたら`みたいなファイルとかに保存されます。

また、`training.py`では学習後の挙動の確認用として、各入力データに対する結果をいくつか表示するようになっています。



### マイクを使った動作確認

realtime.pyは、マイクから取得した音声をcnnに与えて、実際に正しく判定できるかどうか確認するためのモジュールです。
