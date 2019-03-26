import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from flask import Flask
from flask import request
from werkzeug.contrib.fixers import ProxyFix
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

app.wsgi_app = ProxyFix(app.wsgi_app)

def predict(text):
    df = pd.read_csv('./fake_or_real_news.csv')

    y = df.label
    df = df.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    user_input = [text]
    # Test Input
    count_vectorizer_input = CountVectorizer(stop_words='english')
    count_train = count_vectorizer_input.fit_transform(X_train)
    count_input = count_vectorizer_input.transform(user_input)

    # load the model from disk
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(count_input)
    return result

# print(predict("XDD"))


@app.route("/")
def hello():
    return "Welcome to our API"

@app.route("/api")
def test():
    text = request.args.get('text')
    result = predict(text)
    return str(result)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
 



