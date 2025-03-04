import yt_dlp
import sys
import re
import os
import json
from datetime import datetime

def extract_video_id(url):
    video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', url)
    if video_id_match:
        return video_id_match.group(1)
    return None

def download_audio(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': '%(id)s.%(ext)s',
            'quiet': False
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                print("Downloading audio...")
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                
                ext = info['ext']
                audio_path = f"{video_id}.{ext}"
                
                print(f"\nDownloaded audio file: {audio_path}")
                if not os.path.exists(audio_path):
                    for file in os.listdir():
                        if file.startswith(video_id):
                            audio_path = file
                            print(f"Found audio file as: {audio_path}")
                            break
                
                if not os.path.exists(audio_path):
                    raise Exception(f"Audio file not found. Files in directory: {os.listdir()}")
                
                metadata = {
                    "video_id": video_id,
                    "title": info.get('title', 'Unknown'),
                    "uploader": info.get('uploader', 'Unknown'),
                    "duration": info.get('duration', 0),
                    "download_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "audio_file": audio_path
                }
                
                metadata_path = f"{video_id}_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                print(f"Metadata saved to: {metadata_path}")
                print(f"\nINSTRUCTIONS:")
                print(f"1. Transfer both '{audio_path}' and '{metadata_path}' to your faster computer")
                print(f"2. Run 'python transcribe.py {audio_path}' on the faster computer")
                print(f"3. Transfer the resulting transcript file back to your local machine")
                
                return audio_path, metadata_path
                    
            except Exception as e:
                print(f"Error extracting video info: {str(e)}")
                return None, None
                
    except Exception as e:
        print(f"Error in download_audio: {str(e)}")
        return None, None

def main():
    if len(sys.argv) != 2:
        print("Usage: python download.py <youtube_url>")
        return

    url = sys.argv[1]
    video_id = extract_video_id(url)
    
    if not video_id:
        print("Invalid YouTube URL")
        return

    print(f"Processing video ID: {video_id}")
    audio_path, metadata_path = download_audio(url)
    
    if audio_path and metadata_path:
        print(f"Download complete. Files ready for transfer to faster computer:")
        print(f"- Audio: {audio_path}")
        print(f"- Metadata: {metadata_path}")
    else:
        print("Failed to download audio")

if __name__ == "__main__":
    main()