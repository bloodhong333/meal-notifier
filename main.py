import os
import io
import json
import time
import re
import base64
import requests
from datetime import datetime, timedelta, timezone
from groq import Groq
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==========================================
# 1. 환경 변수 설정
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
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
# 3. Groq AI (Qwen Vision) 식단 분석 및 추천
# ==========================================
def get_evening_menu_recommendation(image_bytes):
    print("2. Groq AI(Qwen Vision) 저녁 식단 분석 및 추천 중...")
    
    # 한국 시간(KST) 기준 내일 날짜 구하기
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst)
    tomorrow_kst = now_kst + timedelta(days=1)
    
    today_str = now_kst.strftime("%Y년 %m월 %d일")
    tomorrow_day_num = tomorrow_kst.day
    tomorrow_str = tomorrow_kst.strftime("%m월 %d일")
    tomorrow_day_kr = ["월", "화", "수", "목", "금", "토", "일"][tomorrow_kst.weekday()]

    client = Groq(api_key=GROQ_API_KEY)
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    prompt = f"""
너는 식단표 이미지에서 정확한 정보를 추출하는 OCR 전문가이자 요리 도우미야.

[가장 중요한 임무]
이미지 형태의 월간 식단표 표에서 숫자 **'{tomorrow_day_num}'** 또는 **'{tomorrow_day_num}({tomorrow_day_kr})'** 표시가 된 날짜 칸을 매우 신중하게 찾아라.
해당 칸에 적힌 실제 텍스트만 정확하게 읽어서 [점심]과 [오후 간식] 항목을 기재해라. 절대로 다른 날짜의 메뉴를 적거나 존재하지 않는 메뉴를 지어내지 마라!

[분석 기준 날짜]
• 오늘: {today_str}
• 분석할 내일 날짜: **{tomorrow_day_num}일 ({tomorrow_day_kr}요일)**

[작성 가이드]
1. 식단표에서 {tomorrow_day_num}일({tomorrow_day_kr}) 칸의 [점심] 메뉴와 [간식] 메뉴를 있는 그대로 옮겨 적는다.
2. 해당 점심/간식과 주재료(돼지고기, 닭고기, 두부, 국수 등)가 겹치지 않는 간단하고 선호도 높은 초간단 추천 저녁 메뉴를 작성한다.
3. 생각 과정이나 사족은 적지 말고, 지정된 [출력 형식]만 답변해라.

[출력 형식]
📍 내일({tomorrow_str} {tomorrow_day_kr}) 어린이집 식단
• 점심: (해당 날짜 칸의 실제 점심 메뉴)
• 간식: (해당 날짜 칸의 실제 간식 메뉴)

💡 추천 저녁 메뉴: [메뉴 이름]
• 이유: 점심/간식과 겹치지 않고 아이들이 좋아하는 성공 보장 메뉴!
• 핵심 재료: (간단한 식재료)
• 초간단 조리 팁 (15분 컷):
  1. ...
  2. ...

🎬 추천 레시피 참고:
• 📺 유튜브: (유튜브 레시피 검색어)
• 📝 블로그: (네이버 블로그 레시피 검색어)

👨‍👩‍👧‍👦 어른을 위한 한 끗 팁: (어른용 고명/양념 추가 팁)
"""

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw_content = completion.choices[0].message.content
            
            # <think>...</think> 태그 제거
            clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            return clean_content

        except Exception as e:
            if attempt < 2:
                print(f"⚠️ API 요청 중 오류 발생 ({e}). 10초 후 재시도합니다... ({attempt + 1}/3)")
                time.sleep(10)
            else:
                raise e

# ==========================================
# 4. 카카오톡 메시지 보내기
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
