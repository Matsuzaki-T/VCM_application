#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rfc

#機械学習プログラム起動
pd.options.display.max_columns = None
data = pd.read_csv("./dataset/VCM_train(r28).csv") #dataのロード
parameter = data[["BW","BMI","age","Ccr"]]#patameterの抽出
loading = data[["loading"]]#試験室推奨loadingの抽出
maintain = data[["maintain"]]#試験室推奨の維持量の抽出
clf1 = rfc(max_depth =8, n_estimators=160)#ベストなhyperparameter。事前にgridsearch済み
clf2 = rfc(max_depth =8, n_estimators=40)#ベストなhyperparameter。事前にgridsearch済み
clf1.fit(parameter,maintain)
clf2.fit(parameter,loading)

st.title("バンコマイシン初期投与設計AI")
st.subheader("入力フォーム")
age = st.number_input("年齢", 19, 100, 40)
Ccr = st.number_input("CrCL", 0, 999, 50)
BW = st.number_input("体重", 0, 999, 50)
BMI = st.number_input("BMI", 0, 999, 22)
st.subheader("推奨投与レジメン")

start = st.button("解析開始")

if start == True:
    input = np.array([BW,BMI,age,Ccr])
    input2 = input.reshape(1,4)
    input3 = pd.DataFrame(input2)
    clf_maintenance=clf1.predict(input3)
    clf_loading=clf2.predict(input3)
    output_maintenance = str(clf_maintenance[0])
    output_loading = str(clf_loading[0])+"mg"
    st.write("維持量:",output_maintenance," ローディング量：",output_loading)
else:
    st.write("維持量:   ローディング量：   ")

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Reference: Matsuzaki T, Kato Y, Mizoguchi H, and Yamada K. (2022). J. Pharmacol. Sci. 148(4), 358-363.<br>
   <a href=https://www.sciencedirect.com/science/article/pii/S1347861322000184?via%3Dihub>https://doi.org/10.1016/j.jphs.2022.02.005</a></p>
<p><br>Copyright © Departments of Neuropsychopharmacology and Hospital Pharmacy,<br>
           Nagoya University Graduate School of Medicine, Nagoya, Aichi, 466-8560, Japan</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

