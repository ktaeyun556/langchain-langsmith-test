# LangChain LangSmith 연동 간단 테스트

PDF 문서를 업로드하면 해당 내용을 기반으로 질문에 답변하는 **RAG(Retrieval-Augmented Generation) Q&A 챗봇**입니다.  
Google Gemini 모델과 ChromaDB 벡터 저장소를 활용하며, LangSmith로 실행 로그를 추적합니다.

---

## 기술 스택

- **LangChain** - LLM 애플리케이션 프레임워크
- **LangSmith** - LLM 실행 로그 추적 및 모니터링
- **Google Gemini** - LLM 및 임베딩 모델 (`gemini-2.5-flash`, `gemini-embedding-001`)
- **ChromaDB** - 로컬 벡터 데이터베이스
- **Python 3.11+**

---

## 초기 환경 설정

### 1. 저장소 클론

```bash
git clone https://github.com/ktaeyun556/langchain-langsmith-test.git
cd langchain-langsmith-test
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 패키지 설치

```bash
pip install langchain langchain-community langchain-google-genai langchain-chroma langsmith pypdf chromadb python-dotenv
```

### 4. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 입력합니다.

```
GOOGLE_API_KEY=your_google_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_project_name_here
```

- **Google API Key**: [Google AI Studio](https://aistudio.google.com/)에서 발급
- **LangSmith API Key**: [LangSmith](https://smith.langchain.com/)에서 발급

---

## 실행 방법

### Step 1. 연동 테스트

LLM 연결과 LangSmith 로그 추적이 정상 동작하는지 확인합니다.

```bash
python test_setup.py
```

### Step 2. PDF 문서 임베딩 (벡터 DB 생성)

분석할 PDF 파일을 프로젝트 루트에 `sample.pdf` 이름으로 저장한 후 실행합니다.  
실행 결과로 `chroma_db/` 폴더가 생성됩니다.

```bash
python ingest.py
```

### Step 3. Q&A 챗봇 실행

PDF 내용을 기반으로 질문할 수 있는 대화형 챗봇을 실행합니다.  
종료하려면 `quit` 또는 `exit`를 입력합니다.

```bash
python qa_bot.py
```
