# app.py
from pathlib import Path
import base64, json, asyncio
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from web_sign_translator import WebSignTranslator
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "final_best_model.h5"
LABEL_PATH = MODEL_DIR / "label_map_new.json"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/")
def index():
    return HTMLResponse(INDEX_HTML)

def decode_dataurl_to_bgr(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise ValueError("Invalid dataURL (no comma)")
    b64 = data_url.split(",", 1)[1]
    jpg = base64.b64decode(b64)
    arr = np.frombuffer(jpg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed (codec/format/size issue)")
    return bgr

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    translator = None
    latest_frame_msg = None
    processing = False

    try:
        # ✅ 무거운 생성(모델 로드 등)이면 이벤트루프 막지 않도록 스레드로
        translator = await asyncio.to_thread(
            WebSignTranslator,
            model_path=MODEL_PATH,
            label_map_path=LABEL_PATH,
        )

        await ws.send_text(json.dumps({"ok": True, "cmd": "ready"}, ensure_ascii=False))

        while True:
            msg = await ws.receive_text()

            # 프레임은 "최신 것만" 유지
            if msg.startswith("data:image"):
                latest_frame_msg = msg

                # 이미 처리 중이면 일단 최신 프레임만 갱신하고 돌아감(드롭)
                if processing:
                    continue

                # 처리 시작: 처리하는 동안에도 latest_frame_msg가 갱신될 수 있음
                processing = True
                try:
                    while latest_frame_msg is not None:
                        cur = latest_frame_msg
                        latest_frame_msg = None

                        try:
                            bgr = decode_dataurl_to_bgr(cur)
                        except Exception as e:
                            await ws.send_text(json.dumps({"error": f"bad_frame: {e}"}, ensure_ascii=False))
                            continue

                        result = await asyncio.to_thread(translator.process_frame, bgr)
                        await ws.send_text(json.dumps(result, ensure_ascii=False))
                finally:
                    processing = False

                continue

            # 명령(JSON)
            try:
                obj = json.loads(msg)
            except Exception:
                continue

            cmd = obj.get("cmd")
            if cmd == "reset":
                translator.reset()
                await ws.send_text(json.dumps({"ok": True, "cmd": "reset"}, ensure_ascii=False))

            elif cmd == "translate":
                sent = await asyncio.to_thread(translator.translate_and_tts)
                await ws.send_text(json.dumps({"ok": True, "cmd": "translate", "final_sentence": sent}, ensure_ascii=False))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        # 연결 살아있을 때만 에러 전송 시도
        try:
            await ws.send_text(json.dumps({"error": str(e)}, ensure_ascii=False))
        except Exception:
            pass
    finally:
        if translator is not None:
            try:
                await asyncio.to_thread(translator.close)
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass
