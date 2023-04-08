"""Ask a question to the notion database."""
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import argparse

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

db = FAISS.load_local("faiss_index", OpenAIEmbeddings())

chain = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), retriever=db.as_retriever())
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
