import os
import re
import json
import tempfile
import subprocess
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import assemblyai as aai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")


class ProcessRequest(BaseModel):
    youtube_url: str
    api_key: str


@app.post("/process")
async def process_video(req: ProcessRequest):
    aai.settings.api_key = req.api_key

    video_id = extract_video_id(req.youtube_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    async def stream():
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.%(ext)s")

            yield f"data: {json.dumps({'status': 'downloading', 'msg': 'Downloading audio from YouTube...'})}\n\n"

            try:
                proc = await asyncio.create_subprocess_exec(
                    "yt-dlp",
                    "-f", "bestaudio",
                    "--no-playlist",
                    "-o", audio_path,
                    req.youtube_url,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )

                async for line in proc.stdout:
                    text = line.decode().strip()
                    if text:
                        yield f"data: {json.dumps({'status': 'downloading', 'msg': text})}\n\n"

                await proc.wait()
                if proc.returncode != 0:
                    yield f"data: {json.dumps({'status': 'error', 'msg': 'yt-dlp failed. Is it installed? Run: brew install yt-dlp'})}\n\n"
                    return

            except FileNotFoundError:
                yield f"data: {json.dumps({'status': 'error', 'msg': 'yt-dlp not found. Run: brew install yt-dlp'})}\n\n"
                return

            files = os.listdir(tmpdir)
            if not files:
                yield f"data: {json.dumps({'status': 'error', 'msg': 'Download failed — no file found'})}\n\n"
                return

            actual_path = os.path.join(tmpdir, files[0])
            size_mb = os.path.getsize(actual_path) / 1024 / 1024
            yield f"data: {json.dumps({'status': 'transcribing', 'msg': f'Audio downloaded ({size_mb:.1f} MB). Sending to AssemblyAI...'})}\n\n"

            try:
                config = aai.TranscriptionConfig(
                    speaker_labels=True,
                    speech_models=["universal"]
                )
                transcriber = aai.Transcriber()

                yield f"data: {json.dumps({'status': 'transcribing', 'msg': 'Uploading to AssemblyAI...'})}\n\n"
                transcript = transcriber.transcribe(actual_path, config=config)

                if transcript.status == aai.TranscriptStatus.error:
                    yield f"data: {json.dumps({'status': 'error', 'msg': transcript.error})}\n\n"
                    return

            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'msg': str(e)})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'transcribing', 'msg': 'Processing speaker diarization...'})}\n\n"

            segments = []
            for utt in transcript.utterances:
                segments.append({
                    "speaker": utt.speaker,
                    "start": utt.start / 1000.0,
                    "end": utt.end / 1000.0,
                    "text": utt.text,
                })

            speakers = sorted(list(set(s["speaker"] for s in segments)))

            yield f"data: {json.dumps({'status': 'done', 'video_id': video_id, 'segments': segments, 'speakers': speakers})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def extract_video_id(url):
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None
