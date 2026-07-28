import os
import json
import requests
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# 최신 구글 AI SDK 사용
from google import genai
from google.genai import types

# ==========================================
# 1. 환경 변수 설정
# ==========================================
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY")
KAKAO_REFRESH_TOKEN = os.environ.get("KAKAO_REFRESH_TOKEN")

# ==========================================
# 2. 구글 드라이브에서 최신 식단표 이미지 다운로드
# ==========================================
def download_latest_menu_image():
    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    service = build('drive', 'v3', credentials=creds)

    query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and mimeType contains 'image/' and trashed = false"
    results = service.files().list(
        q=query, orderBy="modifiedTime desc", pageSize=1, fields="files(id, name)"
    ).execute()
    items = results.get('files', [])

    if not items:
        raise FileNotFoundError("구글 드라이브 폴더에서 식단표 이미지를 찾을 수 없습니다.")

    file_id = items[0]['id']
    request = service.files().get_media(fileId=file_id)
    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    file_stream.seek(0)
    return file_stream.read()

# ==========================================
# 3. Gemini AI 식단 분석 및 저녁 메뉴 추천
# ==========================================
def get_evening_menu_recommendation(image_bytes):
    # 최신 SDK 방식으로 클라이언트 생성
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = """
너는 쌍둥이를 키우는 요리 초보 부모를 돕는 친절하고 실용적인 식단 관리 AI 도우미야.
전달받은 어린이집 식단표 이미지에서 [내일일자 점심 메뉴]와 [오후 간식]을 분석해줘.

다음 고려사항을 엄격히 적용해서 내일 저녁으로 준비할 추천 식단을 작성해줘:

[저녁 식단 추천 6대 원칙]
1. 점심/간식 중복 절대 제외: 어린이집 점심 및 간식에 나온 주재료, 국 종류, 메인 요리와 중복되지 않을 것.
2. 조리 편의성 극대화: 요리 초보자이고 쌍둥이 육아로 시간이 부족하므로 15~20분 내로 쉽게 조리할 수 있는 메뉴일 것.
3. 보통의 재료 활용: 대형마트나 집 앞 슈퍼에서 쉽게 구하는 일반적인 식재료 사용할 것.
4. 아이들 선호도 최고 메뉴: 유아/어린이가 호불호 없이 잘 먹기로 검증된 메뉴.
5. 냉동/반가공품 적극 활용 OK: 돈가스, 냉동 떡갈비, 시판 소스, 만두 등 냉동제품/밀키트 활용 레시피 적극 환영.
6. 실제 참고 URL 첨부: 추천한 메뉴를 쉽게 따라 할 수 있는 유튜브 영상 링크나 네이버 블로그 링크를 첨부할 것.
7. 어른 연계 가능성(선택): 아이용으로 먼저 조리 후 고춧가루/청양고추/스리라차 등 어른용 고명을 추가해 함께 먹을 수 있다면 팁으로 첨부.

[출력 형식]
📍 내일 어린이집 점심/간식
• 점심: (식단표 메뉴)
• 간식: (식단표 간식)

💡 추천 저녁 메뉴: [메뉴 이름]
• 이유: 점심/간식과 겹치지 않고 아이들이 좋아하는 성공 보장 메뉴!
• 핵심 재료: (시판/냉동 재료 포함 간단한 재료)
• 초간단 조리 팁 (15분 컷):
  1. ...
  2. ...
🎬 추천 레시피 참고:
• 📺 유튜브: (유튜브 링크)
• 📝 블로그: (블로그 링크)

👨‍👩‍👧‍👦 어른을 위한 한 끗 팁: (어른용 양념 추가 팁)
"""

    # 이미지 파트 포맷 변환
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg",
    )

    # 최신 SDK 방식으로 모델 호출 (gemini-2.0-flash)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, image_part],
    )
    return response.text

# ==========================================
# 4. 카카오톡 나에게 메시지 보내기
# ==========================================
def send_kakao_message(text):
    token_url = "https://kauth.kakao.com/oauth/token"
    token_data = {
        "grant_type": "refresh_token",
        "client_id": KAKAO_REST_API_KEY,
        "refresh_token": KAKAO_REFRESH_TOKEN
    }
    token_res = requests.post(token_url, data=token_data).json()
    access_token = token_res.get("access_token")

    send_url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {
        "template_object": json.dumps({
            "object_type": "text",
            "text": text,
            "link": {
                "web_url": "https://aistudio.google.com",
                "mobile_web_url": "https://aistudio.google.com"
            },
            "button_title": "자세히 보기"
        })
    }
    res = requests.post(send_url, headers=headers, data=payload)
    return res.json()

if __name__ == "__main__":
    print("1. 식단표 이미지 다운로드 중...")
    img_data = download_latest_menu_image()
    
    print("2. Gemini AI 저녁 식단 분석 및 추천 중...")
    recommendation = get_evening_menu_recommendation(img_data)
    
    print("3. 카카오톡 메시지 발송 중...")
    send_kakao_message(recommendation)
    print("완료되었습니다!")
