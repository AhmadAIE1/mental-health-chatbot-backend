
import os
from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from textblob import TextBlob
from langdetect import detect
from googletrans import Translator
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
translator = Translator()
memory = ConversationBufferMemory(input_key="input", memory_key="chat_history")

prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    You are a mental health support assistant for a psychology clinic. Respond empathetically and professionally. 
    If a situation requires urgent help, provide this hotline: 1-800-273-8255.
    Conversation so far: {input}
    Response:
    """
)

documents = ["Cognitive Behavioral Therapy helps identify and change negative thought patterns.",
             "Mindfulness exercises can reduce stress and improve focus."]
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=vectorstore.as_retriever(), memory=memory, prompt=prompt
)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get("message", "")
    response = retrieval_chain.run(input=user_input)
    return jsonify({"reply": response})

if __name__ == '__main__':
    app.run(debug=True)
