# test_report.py
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(str(Path(__file__).parent))

try:
    from trendops.publisher.report_service import ReportService
except ImportError:
    # scripts 폴더 안에 있을 경우를 대비해 상위 경로 추가
    sys.path.append(str(Path(__file__).parent.parent))
    from trendops.publisher.report_service import ReportService

def run_test():
    print("🚀 [테스트] 일일 리포트 강제 생성 시작...")
    
    # 리포트 서비스 초기화
    svc = ReportService()
    
    # 오늘 날짜 로그가 있는지 확인
    log_file = svc._get_log_file()
    if not log_file.exists():
        print(f"⚠️ 경고: 오늘 날짜의 로그 파일이 없습니다.")
        print(f"   경로: {log_file}")
        print("   먼저 'python scripts/real_e2e_pipeline.py'를 실행하여 데이터를 수집해주세요.")
        return

    # 리포트 생성 시도
    report_path = svc.generate_daily_report()
    
    if report_path:
        print(f"\n✅ 리포트 생성 성공!")
        print(f"📄 파일 위치: {report_path}")
        print("-" * 50)
        
        # 파일 내용 미리보기 (앞부분만)
        with open(report_path, "r", encoding="utf-8") as f:
            print(f.read()[:500] + "\n...")
            
    else:
        print("\n❌ 리포트 생성 실패 (에러 로그를 확인하세요)")

if __name__ == "__main__":
    run_test()