# 1. Playwright 공식 이미지 사용
FROM mcr.microsoft.com/playwright/python:v1.41.0-jammy

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 👇 [핵심 수정] playwright-stealth와 'tzdata'(시간대 정보)를 강제로 설치합니다.
RUN pip install playwright-stealth tzdata

# 4. 소스 코드 복사
COPY . .

# 5. Playwright 브라우저 설치
RUN playwright install chromium

# 6. 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
# 시스템 시간대 설정 (이제 tzdata가 있어서 작동합니다)
ENV TZ=Asia/Seoul

# 7. 실행
CMD ["python", "scripts/run_scheduler.py"]