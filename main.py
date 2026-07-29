import os
import io
import json
import time
import base64
import requests
from groq import Groq
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==========================================
# 1. 환경 변수 설정 (등록하신 Secrets 이름과 동일)
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
    
    # 서비스 계정 JSON 문자열 로드
    service_account_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    
    service = build('drive', 'v3', credentials=creds)

    # 폴더 내 이미지 파일 검색 (최신순 1개)
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

    # 이미지 다운로드
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
        
    return fh.getvalue()

# ==========================================
# 3. Groq AI (Llama 3.2 Vision) 식단 분석 및 추천
# ==========================================
def get_evening_menu_recommendation(image_bytes):
    print("2. Groq AI(Llama 3.2 Vision) 저녁 식단 분석 및 추천 중...")
    
    client = Groq(api_key=GROQ_API_KEY)
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

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

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model="llama-3.2-90b-vision-instruct",
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
                temperature=0.2,
                max_tokens=1024,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                print(f"⚠️ API 요청 중 오류 발생. 10초 후 재시도합니다... ({attempt + 1}/3)")
                time.sleep(10)
            else:
                raise e

# ==========================================
# 4. 카카오톡 액세스 토큰 갱신 및 메시지 발송
# ==========================================
def get_kakao_access_token():
    """리프레시 토큰으로 새로운 액세스 토큰을 발급받습니다."""
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
        "Authorization": f"Bearer {access_token}"
    }
    
    payload = {
        "template_object": f"""{{
            "object_type": "text",
            "text": {json.dumps(text_message, ensure_ascii=False)},
            "link": {{
                "web_url": "https://www.google.com",
                "mobile_web_url": "https://www.google.com"
            }}
        }}"""
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
