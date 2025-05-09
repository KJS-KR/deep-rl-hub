from datetime import datetime
import logging
import os

CURRENT_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")


def setup_logging(log_dir="./logs", log_file=f"{CURRENT_TIME}.log", level=logging.INFO):
    """로깅을 설정하는 함수

    Args:
        log_dir (str): 로그 파일을 저장할 디렉토리
        log_file (str): 로그 파일명
        level (int): 로깅 레벨
        - logging.DEBUG: 모든 로그 메시지
        - logging.INFO: 정보 메시지
        - logging.WARNING: 경고 메시지
        - logging.ERROR: 오류 메시지
        - logging.CRITICAL: 치명적인 오류 메시지
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=level,
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )

    logging.info(f"로깅이 설정되었습니다. 로그 파일: {log_file_path}")


if __name__ == "__main__":
    setup_logging()
    logging.info("이것은 정보 메시지입니다.")
    logging.debug("이것은 디버그 메시지입니다.")
    logging.warning("이것은 경고 메시지입니다.")
    logging.error("이것은 오류 메시지입니다.")
    logging.critical("이것은 치명적인 오류 메시지입니다.")
    logging.info("로깅 테스트 완료.")
