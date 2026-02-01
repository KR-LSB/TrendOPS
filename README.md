# 📈 TrendOps: AI 기반 실시간 트렌드 분석 에이전트

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker)
![LLM](https://img.shields.io/badge/LLM-Ollama-orange)
![RAG](https://img.shields.io/badge/RAG-ChromaDB-green)

**TrendOps**는 구글 트렌드를 실시간으로 모니터링하고, 뉴스 및 유튜브 여론을 수집하여 **RAG(검색 증강 생성)** 기반으로 심층 분석하는 **완전 자동화된 AI 에이전트 파이프라인**입니다.

단순한 크롤링을 넘어, 수집된 데이터를 벡터 DB에 저장하고 과거 맥락과 비교 분석하여 환각(Hallucination)을 최소화한 일일 리포트를 자동 발행합니다.

---

## 🏗️ System Architecture



이 프로젝트는 **ETL(Extract, Transform, Load)** 파이프라인과 **LLM Ops**가 결합된 구조를 가집니다.

1.  **Trigger**: 구글 트렌드 급상승 키워드(Top 10) 자동 감지
2.  **Collector (Hybrid)**:
    * **News**: RSS 피드를 통한 실시간 기사 수집
    * **YouTube**: `Playwright`를 활용한 영상 댓글 및 여론 수집
3.  **Storage & Memory**:
    * **Redis**: 중복 데이터 실시간 제거 (Deduplication)
    * **ChromaDB**: 텍스트 임베딩 저장 및 Hybrid Search (BM25 + Vector) 지원
4.  **Analyst (Brain)**:
    * **LLM (Ollama)**: 로컬 LLM(Exaone 3.5 등)을 활용한 심층 분석
    * **Self-Correction**: JSON 파싱 오류 및 품질 저하 시 자동 재시도 로직
    * **Guardrail**: 혐오 표현, 편향성, 개인정보 침해 자동 검열
5.  **Publisher**: Markdown 형식의 일일 트렌드 리포트 자동 생성

---

## 🛠 Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 3.10 |
| **AI / LLM** | Ollama, LangChain, Sentence-Transformers |
| **Database** | ChromaDB (Vector), Redis (Cache/Dedup) |
| **Collection** | Playwright (Headless Browser), BeautifulSoup4, Feedparser |
| **Infra** | Docker, Docker Compose |
| **Scheduling** | APScheduler |

---

## ✨ Key Features (핵심 기술)

### 1. Hybrid RAG (검색 증강 생성)
단순 LLM의 지식 한계를 극복하기 위해 **BM25(키워드 검색)**와 **Vector Search(의미 검색)**를 결합했습니다. 이를 통해 최신 뉴스 트렌드를 정확하게 반영하며, 과거 유사 사례와 비교 분석이 가능합니다.

### 2. Self-Correction & Robustness
LLM이 생성한 결과가 JSON 형식을 위배하거나 내용이 부실할 경우, 시스템이 이를 감지하고 **자동으로 프롬프트를 수정하여 재요청**합니다. 이를 통해 파이프라인의 중단 없는 운영을 보장합니다.

### 3. AI Guardrail System
분석 결과가 대중에게 공개되기 전, **2단계 안전 장치**를 거칩니다.
* **1단계**: 정규식 기반의 민감 정보 필터링
* **2단계**: LLM 기반의 문맥적 편향성 및 혐오 표현 검증

### 4. Dockerized 24/7 Automation
복잡한 의존성(Playwright 브라우저, Redis 등)을 **Docker Container**로 패키징했습니다. `docker-compose up` 명령어 하나로 로컬 및 서버 환경 어디서든 즉시 배포 및 24시간 운영이 가능합니다.

---

## 🚀 How to Run

### Prerequisites
* [Docker](https://www.docker.com/) & Docker Compose
* [Ollama](https://ollama.ai/) (Host 머신에 설치 및 실행 필요)

### 1. Installation
```bash
# Repository Clone
git clone [https://github.com/KR-LSB/trendops.git](https://github.com/KR-LSB/trendops.git)
cd trendops
2. Configuration (Optional)
docker-compose.yml에서 Ollama 호스트 주소 등을 환경에 맞게 수정할 수 있습니다.

3. Execution
백그라운드 모드로 서비스를 실행합니다. (초기 실행 시 Docker 이미지를 빌드합니다.)

Bash
docker-compose up -d --build
4. Check Logs
파이프라인이 정상적으로 작동하는지 로그를 통해 확인합니다.

Bash
docker-compose logs -f trendops
📂 Project Structure
Bash
trendops/
├── data/                 # ChromaDB 및 생성된 리포트 저장소 (Volume)
├── scripts/              # 실행 스크립트
│   ├── run_scheduler.py  # 스케줄러 진입점
│   └── real_e2e_pipeline.py # 핵심 파이프라인 로직
├── src/trendops/
│   ├── trigger/          # 트렌드 감지 모듈
│   ├── collector/        # 데이터 수집 (RSS, YouTube)
│   ├── service/          # 공통 서비스 (Redis, Embeddings)
│   ├── store/            # DB 관리 (ChromaDB)
│   ├── search/           # 검색 엔진 (Hybrid Search)
│   ├── analyst/          # LLM 분석 및 Guardrail
│   └── publisher/        # 리포트 생성기
├── docker-compose.yml    # 컨테이너 오케스트레이션 설정
├── Dockerfile            # 이미지 빌드 명세서
└── requirements.txt      # 의존성 목록
🧪 Performance & Results
수집 속도: Asyncio 비동기 처리를 도입하여 동기 방식 대비 3배 이상 속도 개선

중복 제거율: Redis 기반의 Deduplication 적용으로 중복 기사 약 40% 필터링

안정성: Docker 환경에서 7일 이상 무중단 테스트 완료

📝 License
This project is licensed under the MIT License.