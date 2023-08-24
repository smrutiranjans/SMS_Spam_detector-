import streamlit as st
import pickle
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords 
import nltk
import csv
ps=PorterStemmer()
def filter_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    ft=[]
    for i in text:
        if i.isalnum():
            ft.append(i)
    text=ft[:]#or we can use ft.copy()
    ft.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            ft.append(i)
    text=ft.copy()
    ft.clear()
    for i in text:
        ft.append(ps.stem(i))
    return " ".join(ft)
        
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title('Spam Detector')
sms=st.text_input("Enter the message")
#preprocess
new_sms=filter_text(sms)
if st.button('predict'):
    #vectorize
    vector_inp=tfidf.transform([new_sms])
    result=model.predict(vector_inp)[0]
    if result==1:
         st.header('spam')
    else:
        st.header('not spam')
if st.button('report spam'):
    with open('sms_spam.csv','a') as f:
        csv_writer=csv.writer(f)
        csv_writer.writerow(['spam',sms])