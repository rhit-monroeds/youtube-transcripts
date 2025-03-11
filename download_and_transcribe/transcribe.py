import sys
import os
import json
import whisper
import threading
import time
import subprocess
from datetime import datetime
from tqdm import tqdm

def check_ffmpeg():
    ffmpeg_paths = [
        '',
        os.path.abspath('ffmpeg/bin'),
        os.path.abspath('ffmpeg-master-latest-win64-gpl/bin'),
        os.path.join(os.getcwd(), 'ffmpeg-linux')
    ]
    
    ffmpeg_found = False
    for path in ffmpeg_paths:
        if path:
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            print(f"Found ffmpeg in: {path or 'System PATH'}")
            ffmpeg_found = True
            break
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    
    if not ffmpeg_found:
        if os.name == 'posix':
            print("ffmpeg not found. Please install ffmpeg with: sudo apt-get install ffmpeg")
        else:
            print("ffmpeg not found. Please install ffmpeg and make sure it's in your PATH, ")
            print("or download it from https://ffmpeg.org/download.html and extract to ./ffmpeg directory")
        return False
    
    return True

def load_metadata(audio_path):
    video_id = os.path.splitext(os.path.basename(audio_path))[0]
    metadata_path = f"{video_id}_metadata.json"
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata: {str(e)}")
    
    return {
        "video_id": video_id,
        "title": "Unknown",
        "uploader": "Unknown",
        "duration": 0,
        "download_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "audio_file": audio_path
    }

def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return None
    
    metadata = load_metadata(audio_path)
    video_id = metadata["video_id"]
    
    try:
        if not check_ffmpeg():
            return None
        
        print(f"Loading Whisper model...")
        model = whisper.load_model("base")
        
        pbar = tqdm(desc="Transcribing", bar_format='{desc}: {bar}| {elapsed}')
        stop_progress = threading.Event()
        
        def update_progress():
            while not stop_progress.is_set():
                pbar.update(1)
                time.sleep(0.1)
        
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()
        
        try:
            print(f"Starting transcription for: {metadata.get('title', audio_path)}")
            result = model.transcribe(audio_path)
        finally:
            stop_progress.set()
            progress_thread.join()
            pbar.close()
        
        transcript = []
        for segment in result['segments']:
            transcript.append({
                'timestamp': str(segment['start']),
                'text': segment['text'].strip()
            })
        
        return {
            "metadata": metadata,
            "transcript": transcript
        }
                
    except Exception as e:
        print(f"Error in transcribe_audio: {str(e)}")
        return None

def save_transcript(result, audio_path):
    if not result:
        return False
    
    video_id = result["metadata"]["video_id"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcript_{video_id}_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return filename

def main():
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <audio_file_path>")
        return

    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    print(f"Processing audio file: {audio_path}")
    result = transcribe_audio(audio_path)
    
    if result:
        filename = save_transcript(result, audio_path)
        print(f"Transcript saved to {filename}")
        print(f"\nINSTRUCTIONS:")
        print(f"Transfer '{filename}' back to your local computer")
    else:
        print("Failed to transcribe audio")

if __name__ == "__main__":
    main()