from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import io
import whisper

app = Flask(__name__)
cors = CORS(app)

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("documents.pdf")
data = loader.load()
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(api_key="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"),
)

retriever = vectorstore.as_retriever()



# RAG prompt
template = """Answer the question tersely and directly based only on the following context:
{context}

Question: {question}
"""
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(template)

# LLM
from langchain_community.llms import Together
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.0,
    max_tokens=2000,
    together_api_key="a53ee257c14a94a840a7abdc8e7ffa547b6995d73df153b21b6242a86720ab1d",
    top_k=1,
)

# RAG chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/convertToText', methods=['POST'])
@cross_origin()
def speechToText():
  if 'file' not in request.files:
    return "No file found", 400
  file = request.files['file']
  try:
    inMemoryFile = io.BytesIO()
    inMemoryFile.name = file.filename
    file.save(inMemoryFile)
    text = whisper.convertAudioToText(inMemoryFile)
    assert text, "no text"
    output = chain.invoke(text)
    assert output, f"no output for input: {text}"
    output = output.strip()
    if output.lower().startswith("answer: "):
      output = output[8:]
    return jsonify(
      text=output
    )
  except Exception as e:
    print(e)
    return "Unable to transcribe", 500
  
@app.route('/convertWithGivenText', methods=['POST'])
@cross_origin()
def speechToGivenText():
  text = "When is my flight to SJC?"
  try:
    output = chain.invoke(text)
    assert output, f"no output for input: {text}"
    output = output.strip()
    if output.lower().startswith("answer: "):
      output = output[8:]
    return jsonify(
      text=output
    )
  except Exception as e:
    print(e)
    return "Unable to transcribe", 500