import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.title('機械学習')

uploaded_file=st.file_uploader("ファイルをアップロード")
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)

    #st.write(df.columns)#dfが読み取れているかの確認

    st.divider()
    st.title('データの準備')
    features=st.multiselect('特徴量の選択',
                            df.columns.tolist(),#どの選択肢
                            df.columns.tolist(),)#デフォルトの選択
    
    target=st.selectbox('ターゲットの選択',df.columns.tolist())

    #テストデータをわける
    test_size=st.slider('テストデータのサイズ',0.0,1.0,0.5)
    df_train,df_test=train_test_split(df,test_size=test_size,random_state=1)

    st.divider()
    st.title('モデリング')

    model_option=st.selectbox('モデルを選択',['決定木','ランダムフォレスト','ブースティング決定木'])

    if model_option=='決定木':
        from sklearn.tree import DecisionTreeClassifier
        model=DecisionTreeClassifier(max_depth=4)

    elif model_option=='ランダムフォレスト':
        from sklearn.ensemble import RandomForestClassifier
        model=RandomForestClassifier(n_estimators=200,max_depth=4,random_state=1)

    elif model_option=='ブースティング決定木':
         from sklearn.ensemble import GradientBoostingClassifier
         model=GradientBoostingClassifier(n_estimators=200,max_depth=4,random_state=1)

    #st.write(model)#確かめ

    button=st.button('学習開始')
    if button:
        model.fit(df_train[features],df_train[target])

        st.divider()
        st.title('評価')

        pred_train=model.predict(df_train[features])#学習データの予測
        pred_test=model.predict(df_test[features])#テストデータの予測

        st.write('F1スコア（1であるほどよい）')
        from sklearn.metrics import f1_score
        st.write('学習スコア',f1_score(df_train[target],pred_train))
        st.write('テストスコア',f1_score(df_test[target],pred_test))