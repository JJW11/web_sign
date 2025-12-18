# web_sign_translator.py
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from pathlib import Path
import time
import base64
import threading
import json

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from gtts import gTTS
from playsound import playsound

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class WebSignTranslator:
    """
    웹용:
      - process_frame(bgr): Mediapipe로 랜드마크 그리고 모델 추론 -> (예측/버퍼/가공프레임 dataURL) 반환
      - translate_and_tts(): 버퍼 단어를 GPT 문장으로 만들고 TTS 재생
    """

    # ======== 튜닝 파라미터(요청 반영: 반복횟수↑, threshold↓) ========
    THRESHOLD_MAP = {
        "default": 0.55,   # 기존 0.6 보다 살짝 낮춤
        "neck": 0.50,
        "hurt": 0.30,
        "yesterday": 0.40,
    }

    SKIP_FRAME = 2          # 2프레임마다 1번 추론(부하↓)
    SMOOTH_WINDOW = 5       # 확률 평균(떨림↓)
    CONFIRM_STREAK = 4      # ✅ 인식 성공에 필요한 반복횟수(기존보다 늘림)
    COOLDOWN_SEC = 0.8      # 같은 단어 연속 확정 방지

    # ✅ “동작 변화 거의 없으면 무시” (오동작/잡음 방지)
    MOTION_EPS = 0.002      # 작을수록 민감(너무 크면 잘 안 잡힘)

    # 시퀀스 길이(모델 입력)
    SEQ_LEN = 30

    ACTION_KOR = {
        "head": "머리", "stomach": "배", "neck": "목", "hurt": "아프다",
        "cough": "기침", "dizzy": "어지럽다", "fever": "열",
        "yesterday": "어제", "today": "오늘", "continue": "계속",
        "many": "많이", "little": "조금",
    }

    def __init__(self, model_path: Path, label_map_path: Path):
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Label map not found: {self.label_map_path}")

        # 모델 로드
        self.model = load_model(str(self.model_path), compile=False)

        # 라벨 로드
        label_map = json.loads(self.label_map_path.read_text(encoding="utf-8"))
        self.label_map = label_map
        self.id_to_label = {v: k for k, v in label_map.items()}

        # Mediapipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 상태
        self.sequence = []
        self.buffer_words = []
        self.final_sentence = ""

        self.frame_counter = 0
        self.prob_hist = []
        self.last_candidate = None
        self.streak = 0
        self.last_confirmed_time = 0.0

        self.last_keypoints = None  # motion 체크용

        # OpenAI
        self.client = None
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if OpenAI is not None:
            # 키는 환경변수 OPENAI_API_KEY 사용 권장
            # (없으면 문장 생성은 join으로 대체)
            try:
                self.client = OpenAI()
            except Exception:
                self.client = None

    # ---------- 공통 유틸 ----------
    @staticmethod
    def _bgr_to_dataurl(bgr: np.ndarray, quality: int = 75) -> str:
        ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
        return "data:image/jpeg;base64," + b64

    @staticmethod
    def extract_keypoints(results) -> np.ndarray:
        pose = np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(33 * 4, dtype=np.float32)
        lh = np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
        rh = np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
        return np.concatenate([pose, lh, rh]).astype(np.float32)

    @staticmethod
    def normalize_data(sequence: list) -> np.ndarray:
        seq = np.array(sequence, dtype=np.float32)
        if len(seq) == 0:
            return seq
        if seq.ndim == 1:
            seq = seq[np.newaxis, :]

        left_shoulder = 11 * 4
        right_shoulder = 12 * 4

        ls_x, ls_y, ls_z = seq[:, left_shoulder], seq[:, left_shoulder + 1], seq[:, left_shoulder + 2]
        rs_x, rs_y, rs_z = seq[:, right_shoulder], seq[:, right_shoulder + 1], seq[:, right_shoulder + 2]

        neck_x = (ls_x + rs_x) / 2
        neck_y = (ls_y + rs_y) / 2
        neck_z = (ls_z + rs_z) / 2

        pose_len = 33 * 4
        seq[:, 0:pose_len:4] -= neck_x[:, None]
        seq[:, 1:pose_len:4] -= neck_y[:, None]
        seq[:, 2:pose_len:4] -= neck_z[:, None]

        seq[:, pose_len::3] -= neck_x[:, None]
        seq[:, pose_len + 1::3] -= neck_y[:, None]
        seq[:, pose_len + 2::3] -= neck_z[:, None]

        return seq

    def _motion_ok(self, keypoints: np.ndarray) -> bool:
        """
        동작 변화가 거의 없으면 무시(오동작 방지)
        """
        if self.last_keypoints is None:
            self.last_keypoints = keypoints
            return True
        diff = np.mean(np.abs(keypoints - self.last_keypoints))
        self.last_keypoints = keypoints
        return diff >= self.MOTION_EPS

    # ---------- GPT / TTS ----------
    def speak_async(self, text: str):
        def _speak():
            try:
                filename = f"voice_{int(time.time())}.mp3"
                tts = gTTS(text=text, lang="ko")
                tts.save(filename)
                playsound(filename)
                try:
                    os.remove(filename)
                except Exception:
                    pass
            except Exception as e:
                print("TTS error:", e)

        th = threading.Thread(target=_speak, daemon=True)
        th.start()

    def generate_sentence(self, words: list[str]) -> str:
        if not words:
            return ""

        # 키가 없거나 OpenAI 초기화 실패면 그냥 join
        if self.client is None:
            return " ".join(words)

        prompt = (
            "너는 수어 단어열을 한국어로 '번역'하는 엔진이다.\n"
            "아래 단어들을 자연스러운 한국어 한 문장으로만 변환해라.\n\n"
            f"단어들: {', '.join(words)}\n\n"
            "규칙:\n"
            "- 통역사처럼 설명하지 마라. (예: '환자분은 ~라고 말합니다' 금지)\n"
            "- 존댓말 강제하지 마라. 단어에 맞는 자연스러운 말투로.\n"
            "- 추가 정보/추측/상황 설명 금지\n"
            "- 결과는 문장 1개만 출력\n"
            "- 문장이 질문형은 안된다.\n"
            "- 의사가 아니라 환자가 하는 말을 대변하는거야. 절대 질문이 아니야.\n"
        )
        # 최신 SDK에서 responses API가 있을 수 있어서 우선 시도 -> 실패하면 chat.completions fallback
        try:
            if hasattr(self.client, "responses"):
                resp = self.client.responses.create(
                    model=self.openai_model,
                    input=prompt,
                    temperature=0.5,
                    max_output_tokens=120,
                )
                text = getattr(resp, "output_text", None)
                if text:
                    return text.strip()
        except Exception:
            pass

        try:
            resp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "너는 수어 통역 도우미야."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=120,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return " ".join(words)

    # ---------- 외부에서 호출 ----------
    def reset(self):
        self.sequence = []
        self.buffer_words = []
        self.final_sentence = ""
        self.prob_hist = []
        self.last_candidate = None
        self.streak = 0
        self.last_confirmed_time = 0.0
        self.last_keypoints = None

    def translate_and_tts(self) -> str:
        if not self.buffer_words:
            return ""
        sent = self.generate_sentence(self.buffer_words)
        self.final_sentence = sent
        self.speak_async(sent)
        self.buffer_words = []  # 번역 후 버퍼 비우기
        return sent

    def close(self):
        try:
            self.holistic.close()
        except Exception:
            pass

    def process_frame(self, bgr: np.ndarray) -> dict:
        """
        반환:
          - pred, conf
          - buffer (단어버퍼)
          - frame (관절표시된 결과 이미지 dataURL)
          - status
        """
        self.frame_counter += 1

        # mediapipe는 RGB 입력
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        rgb.flags.writeable = True

        out = bgr.copy()

        # 랜드마크 draw
        self.mp_drawing.draw_landmarks(out, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        self.mp_drawing.draw_landmarks(out, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(out, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        # 키포인트
        keypoints = self.extract_keypoints(results)

        # 동작 변화 없으면 시퀀스에 누적/추론을 억제
        if not self._motion_ok(keypoints):
            return {
                "pred": "",
                "conf": None,
                "buffer": self.buffer_words,
                "final_sentence": self.final_sentence,
                "frame": self._bgr_to_dataurl(out),
                "status": "정지(동작 변화 거의 없음) - 무시 중",
            }

        # 시퀀스 누적
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-self.SEQ_LEN:]

        pred_kor = ""
        conf = None
        status = "인식 중..."

        # SKIP_FRAME 반영
        if len(self.sequence) == self.SEQ_LEN and (self.frame_counter % self.SKIP_FRAME == 0):
            seq = self.normalize_data(self.sequence)
            probs = self.model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]

            # 스무딩
            self.prob_hist.append(probs)
            self.prob_hist = self.prob_hist[-self.SMOOTH_WINDOW:]
            avg_probs = np.mean(self.prob_hist, axis=0)
            # =========================
            # [HURT BIAS] 아프다 가중치
            # =========================
            hurt_id = self.label_map.get("hurt", None)  # label_map: {"hurt": id, ...} 형태여야 함
            if hurt_id is not None:
                avg_probs[hurt_id] *= 1.25   # 1.10~1.50 사이 튜닝 (너무 크면 오검출 늘어남)
                avg_probs = avg_probs / (avg_probs.sum() + 1e-9)
            # =========================
            idx = int(np.argmax(avg_probs))
            conf = float(avg_probs[idx])

            eng = self.id_to_label.get(idx, str(idx))
            pred_kor = self.ACTION_KOR.get(eng, eng)

            thr = self.THRESHOLD_MAP.get(eng, self.THRESHOLD_MAP["default"])
            status = f"현재 예측: {pred_kor} ({conf*100:.0f}%) / thr={thr:.2f} / streak={self.streak}/{self.CONFIRM_STREAK}"

            # 반복횟수(streak)로 확정
            now = time.time()
            if conf >= thr:
                if self.last_candidate == idx:
                    self.streak += 1
                else:
                    self.last_candidate = idx
                    self.streak = 1

                if self.streak >= self.CONFIRM_STREAK and (now - self.last_confirmed_time) >= self.COOLDOWN_SEC:
                    if (not self.buffer_words) or (self.buffer_words[-1] != pred_kor):
                        self.buffer_words.append(pred_kor)
                    self.last_confirmed_time = now
                    self.streak = 0
                    self.prob_hist = []  # 확정 후 흔들림 초기화
                    status = f"✅ 단어 확정: {pred_kor}"
            else:
                # 신뢰도 낮으면 streak 천천히 리셋
                self.streak = max(0, self.streak - 1)

        return {
            "pred": pred_kor,
            "conf": conf,
            "buffer": self.buffer_words,
            "final_sentence": self.final_sentence,
            "frame": self._bgr_to_dataurl(out),
            "status": status,
        }
