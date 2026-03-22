import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. .env 파일 로드 (이 코드가 실행되면서 환경 변수가 세팅됩니다)
load_dotenv()

# 2. 모델 불러오기 (환경 변수에 있는 키를 자동으로 가져다 씁니다)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. 모델에게 질문 던지기
response = llm.invoke("안녕? 넌 누구야? 한 줄로 대답해줘.")

print("LLM의 답변:", response.content)
print("성공! 이제 LangSmith 대시보드에서 로그를 확인해 보세요.")