import io
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models.transformer import TransformerSpeechEnhancement
from utils.audio_processing import AudioProcessor


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "web" / "static"
OUTPUT_DIR = BASE_DIR / "web" / "outputs"
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "latest_checkpoint.pt"

app = FastAPI(title="Speech Enhancement Web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg"}
NOISE_GATE_THRESHOLD_DB = -35.0
NOISE_GATE_REDUCTION_DB = 18.0
NOISE_GATE_FRAME_MS = 20


def apply_noise_gate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.size == 0:
        return audio

    frame_len = int(sample_rate * (NOISE_GATE_FRAME_MS / 1000.0))
    if frame_len < 1:
        return audio

    hop = max(1, frame_len // 2)
    rms = []
    centers = []
    for start in range(0, max(1, len(audio) - frame_len + 1), hop):
        frame = audio[start:start + frame_len]
        rms.append(np.sqrt(np.mean(frame ** 2) + 1e-8))
        centers.append(start + frame_len // 2)

    rms = np.asarray(rms)
    centers = np.asarray(centers)
    rms_db = 20.0 * np.log10(rms + 1e-8)
    gain = np.where(rms_db < NOISE_GATE_THRESHOLD_DB,
                    10 ** (-NOISE_GATE_REDUCTION_DB / 20.0),
                    1.0)

    if centers[0] != 0:
        centers = np.insert(centers, 0, 0)
        gain = np.insert(gain, 0, gain[0])
    if centers[-1] < len(audio) - 1:
        centers = np.append(centers, len(audio) - 1)
        gain = np.append(gain, gain[-1])

    envelope = np.interp(np.arange(len(audio)), centers, gain)
    return audio * envelope


class EnhancerService:
    def __init__(self):
        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_config = self.config.get("model", {})
        self.model = TransformerSpeechEnhancement(
            n_fft=model_config.get("n_fft", 1024),
            d_model=model_config.get("d_model", 512),
            nhead=model_config.get("nhead", 8),
            num_encoder_layers=model_config.get("num_encoder_layers", 6),
            dim_feedforward=model_config.get("dim_feedforward", 2048),
            dropout=model_config.get("dropout", 0.1),
            use_cooperative=model_config.get("use_cooperative", True),
        ).to(self.device)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.audio_processor = AudioProcessor(
            sample_rate=model_config.get("sample_rate", 16000),
            n_fft=model_config.get("n_fft", 1024),
            hop_length=model_config.get("hop_length", 256),
            win_length=model_config.get("win_length", 1024),
        )
        self.audio_processor.to(self.device)

    def enhance(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        else:
            waveform = waveform.T

        audio_tensor = torch.from_numpy(waveform).float()
        target_sr = self.audio_processor.sample_rate
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
            sr = target_sr

        audio_tensor = audio_tensor.to(self.device)

        with torch.no_grad():
            noisy_stft = self.audio_processor.compute_stft(audio_tensor)
            noisy_mag = torch.abs(noisy_stft)
            noisy_phase = torch.angle(noisy_stft)
            noisy_mag_input = noisy_mag.transpose(1, 2)
            pred_mask = self.model(noisy_mag_input)
            enhanced_mag = noisy_mag_input * pred_mask
            enhanced_mag = enhanced_mag.transpose(1, 2)
            enhanced_stft = enhanced_mag * torch.exp(1j * noisy_phase)
            enhanced_audio = self.audio_processor.compute_istft(enhanced_stft)

        enhanced_audio = enhanced_audio.squeeze(0).detach().cpu().numpy()
        enhanced_audio = np.asarray(enhanced_audio, dtype=np.float32)
        enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced_audio = apply_noise_gate(enhanced_audio, self.audio_processor.sample_rate)
        return enhanced_audio


service = EnhancerService()


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = STATIC_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.post("/api/enhance")
async def enhance_audio(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})

    try:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported format '{ext}'. Use WAV, FLAC, or OGG."},
            )

        data = await file.read()
        audio_buf = io.BytesIO(data)

        waveform, sr = sf.read(audio_buf)
        enhanced = service.enhance(waveform, sr)

        if enhanced.size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Audio decoding produced empty output."},
            )

        if enhanced.ndim > 1:
            enhanced = np.mean(enhanced, axis=0)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_id = uuid.uuid4().hex
        out_path = OUTPUT_DIR / f"enhanced_{out_id}.wav"
        sf.write(out_path, enhanced, service.audio_processor.sample_rate, subtype="PCM_16")

        if out_path.stat().st_size == 0:
            out_path.unlink(missing_ok=True)
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to write enhanced audio file."},
            )

        return {
            "filename": out_path.name,
            "url": f"/outputs/{out_path.name}",
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
