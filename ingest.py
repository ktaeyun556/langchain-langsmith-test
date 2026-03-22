# ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. 환경 변수 로드 (API 키 세팅)
load_dotenv()

print("1. PDF 문서를 불러오는 중...")
# 2. PDF 로드 (프로젝트 폴더에 sample.pdf 파일이 있어야 합니다)
loader = PyPDFLoader("sample.pdf")
docs = loader.load()

print(f"2. 문서를 쪼개는 중... (총 {len(docs)} 페이지)")
# 3. 텍스트 분할 (Chunking): 문서를 1000자 단위로 자르고, 문맥이 끊기지 않게 200자씩 겹치게 설정합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(f"3. 쪼개진 텍스트를 벡터 DB에 저장하는 중... (총 {len(splits)} 조각)")
# 4. 임베딩 및 벡터 DB 저장 (구글 임베딩 모델 사용)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 로컬 폴더(./chroma_db)에 벡터 데이터를 파일 형태로 영구 저장합니다.
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("성공! 벡터 DB 저장이 완료되었습니다. 프로젝트 폴더에 chroma_db 폴더가 생겼는지 확인해 보세요.")