import io
import json
import os
import time
from datetime import datetime, timedelta, timezone
from google.api_core.exceptions import GoogleAPIError
import google.genai as genai
from google.genai import types
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import requests

# 환경 변수 로드
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")


# ==========================================
# 1. 구글 드라이브 최신 식단표 이미지 다운로드
# ==========================================
def download_latest_menu_image():
    print("1. 구글 드라이브에서 최신 식단표 이미지 다운로드 중...")

    # 구글 서비스 계정 인증
    service_account_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    credentials = Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    drive_service = build("drive", "v3", credentials=credentials)

    # 이미지 파일 검색 (jpg, jpeg, png)
    query = "mimeType contains 'image/' and trashed = false"
    results = (
        drive_service.files()
        .list(
            q=query,
            orderBy="createdTime desc",
            pageSize=1,
            fields="files(id, name)",
        )
        .execute()
    )

    files = results.get("files", [])

    if not files:
        raise FileNotFoundError(
            "구글 드라이브에서 식단표 이미지 파일을 찾을 수 없습니다."
        )

    latest_file = files[0]
    file_id = latest_file["id"]
    print(f"   ➔ 찾은 파일: {latest_file['name']}")

    # 파일 다운로드 (메모리로 바로 다운로드)
    request = drive_service.files().get_media(fileId=file_id)
    image_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(image_stream, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    return image_stream.getvalue()


# ==========================================
# 2. Gemini AI 저녁 식단 분석 및 추천
# ==========================================
def get_evening_menu_recommendation(image_bytes):
    print("2. Gemini AI 저녁 식단 분석 및 추천 중...")

    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst)
    tomorrow_kst = now_kst + timedelta(days=1)

    tomorrow_day_num = tomorrow_kst.day
    tomorrow_str = tomorrow_kst.strftime("%m월 %d일")
    tomorrow_day_kr = ["월", "화", "수", "목", "금", "토", "일"][
        tomorrow_kst.weekday()
    ]

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
너는 어린이집 식단표 분석 전문가이자 유아식 추천 AI야.
이미지 속 식단표에서 **{tomorrow_day_num}일({tomorrow_day_kr})** 칸의 식단을 찾아서 분석해줘.

⚠️ [주말 및 식단 없음 예외 처리]
- 만약 식단표에 **{tomorrow_day_num}일({tomorrow_day_kr})** 칸이 없거나(예: 주말/휴일), 내용이 비어있다면 고민하거나 혼잣말을 하지 말고 아래 양식으로만 답변해:
  📍 **내일({tomorrow_str} {tomorrow_day_kr})은 어린이집 휴원일/주말입니다.**
  오늘 저녁은 아이가 좋아하는 맛있는 특식을 준비해 보세요! 😋

---

만약 **{tomorrow_day_num}일({tomorrow_day_kr})** 식단이 존재한다면, 아래 원칙에 맞춰 추천 메시지를 작성해줘.

🎯 [저녁 식단 추천 원칙]
1. 메뉴 중복 철저 방지: 아침/점심/간식과 중복되지 않는 유아식 메뉴 선별.
2. 초간단 조리 (15~20분 컷): 요리 초보 부모도 빠르게 준비 가능한 간단 식단.
3. 시판/밀키트 적극 활용: 시판 냉동식품, 밀키트, 반가공품 활용 조리 팁 제공.
4. 소화 및 수면 고려: 밤에 자극적인 메뉴 제외.

⚠️ [중요 출력 규칙]
- 너의 내부 생각 과정, 추론, 혼잣말(Wait, Let's look at..., 영어 설명 등)은 절대로 출력하지 마.
- 오직 아래 [출력 양식] 그대로 최종 결과물만 출력해.

[출력 양식]
📍 **내일({tomorrow_str} {tomorrow_day_kr}) 어린이집 식단**
• 아침 간식: (오전 간식 메뉴)
• 점심: (점심 메뉴)
• 오후 간식: (오후 간식 메뉴)

💡 **추천 저녁: [메뉴 이름]**
• **이유:** (간단히 1~2줄)
• **재료:** (시판/밀키트 활용)
• **조리 팁:** (간단히 2단계로 요약)
• **어른용 팁:** (추가 양념 팁)

🎬 **참고 검색어:**
• 유튜브: (추천 메뉴 + 유아식 레시피)
• 블로그: (추천 메뉴 + 유아식 레시피)
"""

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Gemini API 호출 시도 ({attempt}/{max_retries})...")
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes, mime_type="image/jpeg"
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=3000,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0
                    ),  # 👈 생각(Thinking) 과정이 출력에 포함되지 않도록 차단
                ),
            )
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API 호출 중 오류 발생: {e}")
            if attempt < max_retries:
                sleep_time = attempt * 5
                print(f"{sleep_time}초 후 다시 시도합니다...")
                time.sleep(sleep_time)
            else:
                raise e


# ==========================================
# 3. 텔레그램 채널 메시지 전송
# ==========================================
def send_telegram_message(message):
    print("3. 텔레그램 채널로 메시지 전송 중...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("✅ 텔레그램 채널 전송 성공!")
    else:
        print(
            f"⚠️ 마크다운 전송 실패({response.status_code}), 일반 텍스트로 재시도..."
        )
        payload.pop("parse_mode")
        res_fallback = requests.post(url, json=payload)
        if res_fallback.status_code == 200:
            print("✅ 텔레그램 일반 텍스트 전송 성공!")
        else:
            raise Exception(f"텔레그램 전송 최종 실패: {res_fallback.text}")


# ==========================================
# 메인 실행 흐름
# ==========================================
if __name__ == "__main__":
    # 1. 구글 드라이브 식단표 이미지 다운로드
    image_bytes = download_latest_menu_image()

    # 2. Gemini AI 추천 메시지 생성
    recommendation_msg = get_evening_menu_recommendation(image_bytes)

    # 3. 텔레그램 채널로 전송
    send_telegram_message(recommendation_msg)

# 자동 실행 테스트용 커밋
