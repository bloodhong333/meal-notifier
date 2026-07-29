import io
import json
import os
import requests
from datetime import datetime, timedelta, timezone
from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==========================================
# 1. 환경 변수 설정
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY")

# 내 Refresh Token 및 아내 Refresh Token 설정
KAKAO_REFRESH_TOKEN_ME = os.environ.get("KAKAO_REFRESH_TOKEN")  # 본인 토큰
KAKAO_REFRESH_TOKEN_WIFE = os.environ.get(
    "KAKAO_REFRESH_TOKEN_WIFE"
)  # 아내 토큰


# ==========================================
# 2. Google Drive에서 식단표 이미지 다운로드
# ==========================================
def get_latest_menu_image():
    print("1. 구글 드라이브에서 최신 식단표 이미지 다운로드 중...")

    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise Exception(
            "GOOGLE_SERVICE_ACCOUNT_JSON 환경변수가 설정되지 않았습니다."
        )

    service_account_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_info(
        service_account_info, scopes=scopes
    )

    service = build("drive", "v3", credentials=creds)

    query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and mimeType contains 'image/' and trashed = false"
    results = (
        service.files()
        .list(
            q=query,
            orderBy="createdTime desc",
            pageSize=1,
            fields="files(id, name)",
        )
        .execute()
    )

    items = results.get("files", [])
    if not items:
        raise Exception(
            "구글 드라이브 폴더에서 식단표 이미지를 찾을 수 없습니다."
        )

    file_id = items[0]["id"]
    file_name = items[0]["name"]
    print(f"   ➔ 찾은 파일: {file_name}")

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    return fh.getvalue()


# ==========================================
# 3. Gemini AI 식단 분석 및 추천
# ==========================================
def get_evening_menu_recommendation(image_bytes):
    print("2. Gemini AI 저녁 식단 분석 및 추천 중...")

    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst)
    tomorrow_kst = now_kst + timedelta(days=1)

    today_str = now_kst.strftime("%Y년 %m월 %d일")
    tomorrow_day_num = tomorrow_kst.day
    tomorrow_str = tomorrow_kst.strftime("%m월 %d일")
    tomorrow_day_kr = ["월", "화", "수", "목", "금", "토", "일"][
        tomorrow_kst.weekday()
    ]

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
식단표 이미지에서 **{tomorrow_day_num}일({tomorrow_day_kr})** 칸의 [오전 간식], [점심], [오후 간식] 메뉴를 정확히 읽고, 아래 [저녁 식단 추천 원칙]에 맞춰 저녁 추천 메시지를 작성해줘.

🎯 [저녁 식단 추천 원칙]
1. 메뉴 중복 철저 방지 (아이들 선호 메뉴 중심):
   - 아침/점심/간식에 포함된 주재료(단백질/주식), 국 종류, 메인 반찬과 중복되지 않아야 함.
   - 단, 아이들이 호불호 없이 좋아한다고 널리 알려진 성공 보장 유아식 메뉴를 우선 선별할 것.
2. 초간단 조리 (15~20분 컷):
   - 요리 초보 부모도 15~20분 이내로 빠르게 준비할 수 있는 간단 식단일 것.
3. 시판 제품 및 밀키트 적극 활용:
   - 시판 냉동식품, 밀키트, 반가공품(예: 시판 떡갈비, 냉동 볶음밥, 사골육수, 훈제오리, 시판 소스 등)을 적극 활용한 조리 팁을 제시할 것.
4. 소화 및 수면 고려:
   - 밤에 자극적이거나 부담을 주어 수면에 방해되는 메뉴는 제외할 것.

⚠️ [작성 규칙 - 매우 중요]
- 전체 메시지를 2개의 카카오톡 메시지로 나누어 보낼 예정임.
- 두 메시지 사이에 정확히 [MSG_SPLIT] 라는 구분자를 넣어줄 것.
- 1번째 메시지: 어린이집 식단 요약 (200자 내외)
- 2번째 메시지: 추천 저녁 메뉴, 재료, 상세 조리 팁, 레시피 검색어 (600~750자 내외)
- 답변을 중간에 끊지 말고 끝까지 완성해서 출력해 줘.

[출력 양식]
📍 내일({tomorrow_str} {tomorrow_day_kr}) 어린이집 식단
• 아침 간식: (오전 간식 메뉴 전체)
• 점심: (점심 메뉴 전체)
• 오후 간식: (오후 간식 메뉴 전체)

💡 추천 저녁: [메뉴 이름]
• 이유: (아침/점심/간식 중복 제외 이유 및 아이 선호도)
• 재료: (시판/밀키트/냉동식품 활용 재료 포함)
• 조리 팁:
 1. (15~20분 컷 간단 조리 순서)
 2. (간단 조리 순서)

🎬 참고 검색어:
• 유튜브: (추천 메뉴 + 유아식 레시피 검색어)
• 블로그: (추천 메뉴 + 유아식 레시피 검색어)
"""

    # 공식 지원 모델인 gemini-2.0-flash 사용
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=2000, temperature=0.2
        ),
    )

    return response.text


# ==========================================
# 4. 카카오톡 메시지 전송 (2명 전송 로직)
# ==========================================
def get_kakao_access_token(refresh_token):
    """지정한 Refresh Token으로 새로운 Access Token을 발급받습니다."""
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": KAKAO_REST_API_KEY,
        "refresh_token": refresh_token,
    }
    response = requests.post(url, data=data)
    result = response.json()
    if "access_token" in result:
        return result["access_token"]
    else:
        raise Exception(f"카카오 토큰 갱신 실패: {result}")


def send_kakao_message(text_message, refresh_token, user_label="사용자"):
    """특정 대상에게 메시지를 전송합니다."""
    if len(text_message) > 950:
        text_message = (
            text_message[:950]
            + "\n\n(※ 내용이 길어 일부 생략되었습니다.)"
        )

    try:
        access_token = get_kakao_access_token(refresh_token)
        url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        template_object = {
            "object_type": "text",
            "text": text_message,
            "link": {
                "web_url": "https://www.google.com",
                "mobile_web_url": "https://www.google.com",
            },
        }

        payload = {
            "template_object": json.dumps(template_object, ensure_ascii=False)
        }

        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            print(f"   ✅ [{user_label}] 카카오톡 메시지 전송 성공!")
        else:
            print(
                f"   ❌ [{user_label}] 카카오톡 전송 실패"
                f" ({response.status_code}): {response.text}"
            )
    except Exception as e:
        print(f"   ❌ [{user_label}] 전송 중 오류 발생: {e}")


def broadcast_messages(message_list):
    """본인과 아내에게 메시지를 모두 전송합니다."""
    targets = [("본인", KAKAO_REFRESH_TOKEN_ME), ("아내", KAKAO_REFRESH_TOKEN_WIFE)]

    for label, token in targets:
        if not token:
            print(f"\n⚠️ [{label}]의 REFRESH_TOKEN 환경 변수가 설정되어 있지 않아 건너땁니다.")
            continue

        print(f"\n📩 [{label}]에게 메세지 전송 시작...")
        for msg in message_list:
            clean_msg = msg.strip()
            if clean_msg:
                send_kakao_message(clean_msg[:900], token, user_label=label)


# ==========================================
# 5. 메인 실행
# ==========================================
if __name__ == "__main__":
    try:
        img_data = get_latest_menu_image()
        recommendation = get_evening_menu_recommendation(img_data)

        print("\n--- [AI 추천 결과] ---")
        print(recommendation)
        print("----------------------\n")

        # [MSG_SPLIT] 구분자로 메시지 분할
        if "[MSG_SPLIT]" in recommendation:
            messages = recommendation.split("[MSG_SPLIT]")
        else:
            messages = [recommendation]

        # 본인 및 아내에게 메세지 전송
        print("3. 카카오톡 메시지 전송 중...")
        broadcast_messages(messages)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        exit(1)
