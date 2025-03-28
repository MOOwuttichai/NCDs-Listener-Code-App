import pandas as pd
import os
import sys

from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic #v 0.1.15
#v0.2
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.indexes import VectorstoreIndexCreator

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.llm import LLMChain
import random
df = pd.read_csv('data_for_dash_01.csv',encoding='utf-8-sig')
lines = df['comments']
L =len(df)
with open('comment.txt', 'w',encoding='utf-8-sig') as file:
    i = 1
    for line in lines:
        file.write(f'{i}.'+ line + '\n')
        i += 1
if L > 60 :
  with open('comment.txt', 'r') as f:
    lines = f.readlines()

  # สุ่มความคิดเห็น 60 ความคิดเห็น
  random_comments = random.sample(lines, 60)

  # บันทึกผลลัพธ์เป็นไฟล์ใหม่
  with open('comment.txt.txt', 'w') as f:
    f.writelines(random_comments)
##---------------------------ให้บอททำการสรุป ------------------------------##
#API KEY for use
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCCgjV6w7I6W0p4elChq0NLyShEau3YGEE'
#Path DATA
# path_comment = 'comment.txt'
#LOAD DATA
raw_documents = TextLoader('comment.txt',encoding='utf-8-sig')
loader = raw_documents.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(loader)
#TRAFROM TEXT DATA TO VECTOR
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = Chroma.from_documents(documents, embedding=gemini_embeddings, persist_directory="./langchain/chroma_db")

# Load from disk
vectorstore_disk = Chroma(
                        persist_directory="./langchain/chroma_db",
                        embedding_function=gemini_embeddings   # Embedding model
                   )

retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})
#SELECT CHAT MODEL
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash",
                 temperature=0.7, top_p=0.85)
#PROMPT TEMPLATE
llm_prompt_template = """
Answer the question based only on the following context:
{context}

Question: {question}

Respond in the same language as the question.
"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# Combine data from documents to readable string format.
def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)
query = "คุณสรุปคอมเม้นให้ฉันได้มั้ย ว่ามีโรคอะไรอาการของโรคเป็นยังไงบ้าง? เเละข้อมูลที่น่าสนใจอื่น? ไม่ต้องแสดงว่าเป็นคอมเมนต์ที่เท่าไหร่ สรุปให้อยู่ในบรรทัดสวยงาม"
result = rag_chain.invoke(query)
with open('bot_summarize_comment.txt', 'w',encoding='utf-8-sig') as file:
    file.write(result)
collection_name = db._collection.name
try:
    db._client.delete_collection(name = collection_name)
    print(f"Collection '{collection_name}' deleted successfully.")
except Exception as e:
    print(f"Error deleting collection: {e}")
##---------------------------อ่านเพื่อนำไปสู่ HTML ------------------------------##
# with open('bot_summarize_comment.txt', 'r',encoding='utf-8-sig') as file:
#     look = file.read()
#     print(look)

# question = 
