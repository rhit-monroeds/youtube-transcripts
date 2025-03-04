Download, transcribe, and analyze youtube audio for videos which don't have subtitles or transcripts available

#### Workflow
1. Download audio file: `python .\download_and_transcribe\download.py https://www.youtube.com/watch?v=XXXXXXXXXXX`
2. Transcribe audio file to JSON using local OpenAI whisper model: `python .\download_and_transcribe\transcribe.py XXXXXXXXXXX.webm`
3. Analyze transcripts using model of your choice (Deepseek V3 default): `python .\analyze\analyze_transcripts.py`

#### Prerequisites
- Python 3.x
- ffmpeg

#### Notes
- Download ffmpeg and place in the base layer of this project
- For analyzing, set the environment variable `OPENROUTER_API_KEY` to your own Open Router key
  - Powershell: `$env:OPENROUTER_API_KEY="YOUR_API_KEY"`
  - Bash: `export OPENROUTER_API_KEY=YOUR_API_KEY`
- `requirements_transcribe.txt` is optimized to work on a lambdalabs cloud instance, to improve transcription speed