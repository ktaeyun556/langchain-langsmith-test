# qa_bot.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 환경 변수 로드
load_dotenv()

# 2. 모델 및 임베딩 세팅 (이전 단계와 동일하게 최신 모델 사용)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 3. 저장해둔 Chroma DB 불러오기
print("로컬 벡터 DB를 불러오는 중입니다...")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 4. 검색기(Retriever) 생성: 질문과 가장 유사한 문서 조각을 3개만 가져오도록 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. 프롬프트 템플릿 작성 (LLM에게 내리는 프롬프트 엔지니어링)
system_prompt = (
    "당신은 주어진 문맥(Context)을 바탕으로 질문에 답하는 전문적인 AI 어시스턴트입니다. "
    "답을 모른다면 모른다고 명확히 말하고, 문맥에 없는 내용을 지어내지 마세요. "
    "최대한 논리적이고 이해하기 쉽게 답변해 주세요.\n\n"
    "문맥(Context): {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 6. RAG 체인(Chain) 조립
# - 문서들을 텍스트로 결합해 프롬프트에 넣는 체인 생성
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# - 검색기(Retriever)와 결합하여 최종 RAG 파이프라인 완성
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("\n🤖 PDF 기반 Q&A 챗봇이 준비되었습니다! (종료하려면 'quit' 또는 'exit' 입력)")
print("-" * 50)

# 7. 대화형 루프 시작
while True:
    query = input("\n👤 질문을 입력하세요: ")
    if query.lower() in ['quit', 'exit']:
        print("챗봇을 종료합니다. 수고하셨습니다!")
        break

    print("🤖 답변을 생성하는 중...")
    
    # 체인 실행! (이 과정이 전부 LangSmith에 기록됩니다)
    response = rag_chain.invoke({"input": query})
    
    print(f"\n💡 답변:\n{response['answer']}")
    
    # 어떤 문서 조각을 참고했는지 출처 확인
    print("\n[🔍 참고한 문서 출처]")
    for i, doc in enumerate(response['context']):
        page = doc.metadata.get('page', '알수없음')
        # 내용이 너무 길면 잘라서 보여줌
        print(f"- {i+1}번 조각 (페이지 {page}): {doc.page_content[:50].replace(chr(10), ' ')}...")