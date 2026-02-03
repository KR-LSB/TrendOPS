# 1. Playwright 공식 이미지 사용
FROM mcr.microsoft.com/playwright/python:v1.41.0-jammy

# 2. 작업 디렉토리 설정
WORKDIR /app

# 👇 [수정됨] 입력 대기 없이 강제로 설치하는 옵션 추가
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    echo "Asia/Seoul" > /etc/timezone && \
    apt-get clean

# 3. 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install playwright-stealth

# 4. 소스 코드 복사
COPY . .

# 5. Playwright 브라우저 설치
RUN playwright install chromium

# 6. 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV TZ=Asia/Seoul

# 7. 실행
CMD ["python", "scripts/run_scheduler.py"]