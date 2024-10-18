from flask import Flask, request, jsonify, render_template
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import nltk

nltk.data.path.append('./nltk_data')

ps=PorterStemmer()

def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)

  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text=y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text=y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return ' '.join(y)

app=Flask(__name__)

@app.route('/', methods=['POST','GET'])
def home():
    if request.method=="POST":
        content=request.form.get('content')

        model=pickle.load(open('model.pkl','rb'))
        tfidf=pickle.load(open('vectorizer.pkl','rb'))

        
        transformed_text=transform_text(content)

        vector_input=tfidf.transform([transformed_text])

        result=model.predict(vector_input)[0]

        print(result)
        if result==0:
            print('Not Spam')
            return render_template('index.html',prediction_text="Not Spam")
        else:
            print("spam")
            return render_template('index.html',prediction_text='Spam')
      

    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)