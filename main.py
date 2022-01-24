from fastapi import FastAPI 
import uvicorn
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

#downloading english stopwords from nltk
#nltk.download('stopwords')

#initializing PorterStemmer
ps = PorterStemmer()

#loading the count vectorizer
tfidf_vectorizer = open("tfidf_vectorizer.pkl", 'rb')
vectorizer = pickle.load(tfidf_vectorizer)

#loading the model
svc_model = open("svc_model.pkl", 'rb')
model = pickle.load(svc_model)

#creating the sent_processor function for removing stopwords and stemming the words left out.
def sent_processor(text):
    """
    sent(text)
    Docstring:
        Returns stemmed words in the parsed text after removal of all stopwords included in 
        nltk.stopwords.words('english')

    Parameter:
        input: strin {text}
    """
    #scrap away all non alphabetic character from the text
    sent = re.sub('[^a-zA-Z]', ' ', text)
    sent = sent.lower()
    sent = sent.split()
    
    #iterate through sent and 
    sent_stopword_removed = []
    for word in sent:
        if word not in stopwords.words('english'):
            word_stem = ps.stem(word)
            sent_stopword_removed.append(word_stem)
        else:
            pass

    new_sent = ' '.join(sent_stopword_removed)
    return new_sent

app = FastAPI()

@app.get("/")
def root():
    return {"Cyber Bullying model API ready for production"}

@app.get("/predict/{text}")
def predict(text):
    text_processed = sent_processor(text)
    text_processed = vectorizer.transform([text_processed])
    prediction = model.predict(text_processed)
    if prediction[0]=='0':
        result = "Unflagged"
    else:
        result = "Flagged"

    return {"Text": text, "Result": result}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
