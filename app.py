import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



from flask import Flask, render_template, request
import google.generativeai as genai

# Replace with your actual API key
API_KEY = "AIzaSyDYTWqn6W1T6tLPSkIc9XQHa4zh-iFoOZk"

app = Flask(__name__, static_url_path='/static')


# Configure Google GenerativeAI
genai.configure(api_key=API_KEY)

generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["hi"]
  },  
  {
    "role": "model",
    "parts": ["Hi! 👋 How can I help you today?"]
  },
])

@app.route("/")
def home():
  return render_template("index.html")

@app.route("/check_email", methods=["POST"])
def check_email():
  email_id = request.form["email_id"]
  title = request.form["title"]
  body = request.form["body"]

  prompt = f"I received a mail from {email_id} the title says \"{title}\" and the body of the mail is as :\"{body}\" can you please check for this email ID or its domain on the internet and based on the content of the mails and whatever the links are attached to it and keywords used can you give a rough estimate that how many percentage chacnce is it that its a spam email and why? response should be of type \'There's a x percent chacnce of this being a spam email because....\' in under 50 words. Dont cosider typos and grammatical errors as one of top ways to judge the legitimacy, you can ignore some. And @vit.ac.in is not a spam domain(but you can mention what email is about and is it important or urgent) and @vitstudent.ac.in is also trusted domain and not spam. if and only if the emailID is exactly of type \"firstname.secondname2024@vitstudent.ac.in\" would conatin a century( in 4 digits) , mention this informaton that \"mail is probably from a student\" dont mention these prompt informations only mention these when those conditions are met."

  convo.send_message(prompt)
  response = convo.last.text

  return render_template("result.html", email_id=email_id, title=title, response=response)

if __name__ == "__main__":
  app.run(debug=True)

