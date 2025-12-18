# 🤟 실시간 수어 번역 웹 애플리케이션

병원 커뮤니케이션을 위한 한국 의료 수어 실시간 인식 및 문장 생성 시스템

---

## 📋 프로젝트 소개

이 프로젝트는 웹캠을 통해 실시간으로 수어를 인식하고, 인식된 단어들을 자연스러운 문장으로 변환하여 음성으로 출력하는 웹 애플리케이션입니다. MediaPipe 기반 관절 인식과 TensorFlow 딥러닝 모델을 활용하며, GPT를 통해 자연스러운 문장을 생성합니다.

### 주요 기능

- 🎥 실시간 웹캠 기반 수어 인식
- 🦴 MediaPipe를 활용한 관절 키포인트 추출
- 🧠 TensorFlow/Keras 딥러닝 모델 기반 단어 분류
- 💬 GPT 기반 자연어 문장 생성
- 🔊 TTS(Text-to-Speech) 음성 출력
- 🖥️ WebSocket 기반 실시간 통신

---

## 🗂️ 프로젝트 구조

```
web_sign/
├── .venv/                    # 가상환경 (직접 생성 필요)
├── requirements.txt          # 패키지 의존성
├── app.py                    # FastAPI 서버 & WebSocket 통신
├── web_sign_translator.py    # 수어 인식 모델 & 문장 생성
├── models/
│   ├── final_best_model.h5   # 학습된 수어 인식 모델
│   └── label_map_new.json    # 단어 라벨 매핑
├── static/                   # 정적 파일 (프리셋 영상)
│   ├── help.mp4
│   ├── since.mp4
│   └── where.mp4
└── templates/
    └── index.html            # 웹 UI
```

> ⚠️ **주의**: 파일 구조를 정확히 맞춰야 정상 동작합니다.

---

## 🚀 설치 및 실행

### 1. 프로젝트 클론

```bash
git clone <repository-url>
cd web_sign
```

### 2. 가상환경 생성 (Python 3.9.13)

**Windows PowerShell**
```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
python3.9 -m venv .venv
source .venv/bin/activate
python --version  # 3.9.x 확인
```

### 3. 패키지 설치

```bash
python -m pip install -r requirements.txt
```

### 4. OpenAI API 키 설정

**Windows PowerShell**
```powershell
# 임시 설정
$env:OPENAI_API_KEY="your-api-key"

# 영구 설정
setx OPENAI_API_KEY "your-api-key"
```

**Linux / macOS**
```bash
# 임시 설정
export OPENAI_API_KEY="your-api-key"

# 영구 설정
echo 'export OPENAI_API_KEY="your-api-key"' >> ~/.bashrc
source ~/.bashrc
```

### 5. 서버 실행

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --log-level debug --ws-max-size 16777216 --ws-ping-interval 20 --ws-ping-timeout 20
```

### 6. 웹 브라우저 접속

브라우저에서 `http://127.0.0.1:8000` 접속 후 **카메라 권한을 허용**해주세요.

---

## 📖 사용 방법

1. 웹페이지 접속 후 카메라 권한 허용
2. 관절이 표시된 카메라 화면과 실시간 단어 예측률 확인
3. 수어 동작 수행 → 임계값 초과 시 단어가 버퍼에 자동 등록
4. 원하는 단어들이 모두 등록되면 **번역 버튼** 클릭
5. 생성된 문장 확인 및 TTS 음성 출력

---

## 📚 인식 가능 단어 목록

| 단어 | 레퍼런스 |
|:---:|:---|
| 머리 | [YouTube](https://youtu.be/jwQ1QJKnT34?si=1Mr5yfQlrrTVyCyu) |
| 배(탈) | [국립국어원 수어사전](https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=9135) |
| 목 | [국립국어원 수어사전](https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=5468) |
| 아프다 | [YouTube](https://youtube.com/shorts/6iodFJaC3FI?si=9CdDWrHJ0Xq4DeIZ) |
| 기침 | [YouTube](https://youtube.com/shorts/qAtVxONcydk?si=bjHXUC88HuuX_qi9) |
| 어지럽다 | [YouTube](https://youtu.be/JnW6csc-BHc?si=mn0EsInkutWw1a8p) |
| 열 | [국립국어원 수어사전](https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=7828) |
| 어제 | [YouTube](https://youtube.com/shorts/fpunDqLUTqM?si=K1WMVEwbc4iUW9Uf) |
| 오늘 | [YouTube](https://youtube.com/shorts/EmDW_-JZnbY?si=4TBY8p9sub_esc_v) |
| 계속 | [YouTube](https://youtube.com/shorts/KUzI9DZW0Vg?si=FuVi64XKCJWj1ISr) |
| 많이 | [YouTube](https://youtube.com/shorts/bSKkVohSc6M?si=1GTO3GOqsdwJiZVT) |
| 조금 | [YouTube](https://youtube.com/shorts/W_LjzBAgMnM?si=ud0ndsNqBr6qfRAo) |

---

## 🔧 개선 사항 및 트러블슈팅

### 단어별 인식률 차이

일부 단어는 잘 인식되고, 일부는 인식이 어려울 수 있습니다.

**해결 방법**: `web_sign_translator.py`에서 단어별 임계값(threshold) 조정

### 문장 생성 품질 문제

예: `('어제', '오늘', '머리', '아프다')` → "어제 오늘 머리가 아프세요?" (의도치 않은 질문형)

**해결 방법**: `web_sign_translator.py`의 GPT 프롬프트에 강조 사항 추가

### 모델 정확도 향상

더 양질의 데이터와 최적화된 학습 모델 구축을 통해 개선 가능

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|:---:|:---|
| Backend | FastAPI, WebSocket, Uvicorn |
| AI/ML | TensorFlow, Keras, MediaPipe |
| NLP | OpenAI GPT API |
| Frontend | HTML, JavaScript |
| Language | Python 3.9.13 |

---

## 📝 피드백

오류 발생 시 jangjw001@gmail.com/개인톡으로 연락해주세요.

---

## 📄 License

This project is for educational purposes.
