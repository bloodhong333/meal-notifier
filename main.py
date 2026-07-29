import os
import io
import json
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
KAKAO_REFRESH_TOKEN = os.environ.get("KAKAO_REFRESH_TOKEN")

# ==========================================
# 2. Google Drive에서 식단표 이미지 다운로드
# ==========================================
def get_latest_menu_image():
    print("1. 구글 드라이브에서 최신 식단표 이미지 다운로드 중...")
    
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise Exception("GOOGLE_SERVICE_ACCOUNT_JSON 환경변수가 설정되지 않았습니다.")

    service_account_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    
    service = build('drive', 'v3', credentials=creds)

    query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and mimeType contains 'image/' and trashed = false"
    results = service.files().list(
        q=query,
        orderBy="createdTime desc",
        pageSize=1,
        fields="files(id, name)"
    ).execute()
    
    items = results.get('files', [])
    if not items:
        raise Exception("구글 드라이브 폴더에서 식단표 이미지를 찾을 수 없습니다.")

    file_id = items[0]['id']
    file_name = items[0]['name']
    print(f"   ➔ 찾은 파일: {file_name}")

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
        
    return fh.getvalue()

# ==========================================
# 3. Gemini AI 식단 분석 및 추천 (Gemini 2.5 Flash 사용)
# ==========================================
def get_evening_menu_recommendation(image_bytes):
    print("2. Gemini AI 저녁 식단 분석 및 추천 중...")
    
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst)
    tomorrow_kst = now_kst + timedelta(days=1)
    
    today_str = now_kst.strftime("%Y년 %m월 %d일")
    tomorrow_day_num = tomorrow_kst.day
    tomorrow_str = tomorrow_kst.strftime("%m월 %d일")
    tomorrow_day_kr = ["월", "화", "수", "목", "금", "토", "일"][tomorrow_kst.weekday()]

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
너는 식단표 이미지에서 정확한 메뉴를 읽어내고, 쌍둥이를 키우는 초보 부모를 돕는 친절한 식단 추천 AI 도우미야.

전달받은 식단표 이미지에서 **{tomorrow_day_num}일({tomorrow_day_kr})** 칸에 있는 정확한 [점심] 및 [오후 간식] 메뉴를 읽어줘.

⚠️ [카카오톡 전송 글자 수 제한 규칙 - 매우 중요]
전체 답변을 공백 포함 **600자 이내**로 작성해라. 불필요한 인사말이나 서론은 삭제하고 핵심만 작성해라.

[출력 형식]
📍 내일({tomorrow_str} {tomorrow_day_kr}) 어린이집 식단
• 점심: (해당 날짜 칸의 점심 메뉴)
• 간식: (해당 날짜 칸의 간식 메뉴)

💡 추천 저녁: [메뉴 이름]
• 이유: 점심/간식과 겹치지 않고 아이들이 좋아하는 성공 보장 메뉴!
• 핵심 재료: (시판/간편 재료 포함 간단 재료)
• 초간단 조리 팁 (15분 컷):
  1. ...
  2. ...

🎬 추천 레시피 참고:
• 📺 유튜브: (관련 유튜브 레시피 검색어)
• 📝 블로그: (관련 블로그 레시피 검색어)
"""

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=700,
            temperature=0.2
        )
    )
    
    return response.text

# ==========================================
# 4. 카카오톡 메시지 전송
# ==========================================
def get_kakao_access_token():
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": KAKAO_REST_API_KEY,
        "refresh_token": KAKAO_REFRESH_TOKEN
    }
    response = requests.post(url, data=data)
    result = response.json()
    if "access_token" in result:
        return result["access_token"]
    else:
        raise Exception(f"카카오 토큰 갱신 실패: {result}")

def send_kakao_message(text_message):
    print("3. 카카오톡 메시지 전송 중...")
    
    # 카카오톡 1,000자 제한 방지 안전장치
    if len(text_message) > 950:
        text_message = text_message[:950] + "\n\n(※ 내용이 길어 일부 생략되었습니다.)"
        
    access_token = get_kakao_access_token()
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    template_object = {
        "object_type": "text",
        "text": text_message,
        "link": {
            "web_url": "https://www.google.com",
            "mobile_web_url": "https://www.google.com"
        }
    }
    
    payload = {
        "template_object": json.dumps(template_object, ensure_ascii=False)
    }
    
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        print("✅ 카카오톡 메시지 전송 성공!")
    else:
        print(f"❌ 카카오톡 전송 실패 ({response.status_code}): {response.text}")

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
        
        send_kakao_message(recommendation)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        exit(1)
