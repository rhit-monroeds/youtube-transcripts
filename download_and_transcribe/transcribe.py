import sys
import os
import json
import whisperx
import threading
import torch
import time
import subprocess
import platform
import ctypes.util
from datetime import datetime
from tqdm import tqdm

def check_cuda_libraries():    
    if not torch.cuda.is_available():
        print("CUDA not available. Your PyTorch installation doesn't detect a CUDA-capable GPU.")
        return False
    
    missing_libraries = []
    
    libraries_to_check = [
        ("libcudnn", "cuDNN"),
        ("libcudart", "CUDA Runtime"),
        ("libcublas", "CUDA BLAS"),
        ("libcufft", "CUDA FFT"),
        ("libcurand", "CUDA Random Numbers"),
        ("libcusolver", "CUDA Solver"),
        ("libcusparse", "CUDA Sparse Matrix")
    ]
    
    for lib_name, lib_desc in libraries_to_check:
        if not ctypes.util.find_library(lib_name):
            missing_libraries.append((lib_name, lib_desc))
    
    if missing_libraries:
        print("\nWARNING: Missing CUDA libraries detected:")
        for lib_name, lib_desc in missing_libraries:
            print(f"  - {lib_name} ({lib_desc})")
        
        print("\nInstallation instructions:")
        system = platform.system()
        
        if system == "Linux":
            print("""
For Ubuntu/Debian:
  1. Install CUDA Toolkit:
     sudo apt update
     sudo apt install nvidia-cuda-toolkit
     
  2. Install cuDNN:
     sudo apt install libcudnn8
     
  (Or download specific version from NVIDIA website with developer account)
  
For other Linux distributions, use the appropriate package manager or
visit: https://developer.nvidia.com/cuda-downloads
""")
        elif system == "Windows":
            print("""
For Windows:
  1. Download and install CUDA Toolkit from:
     https://developer.nvidia.com/cuda-downloads
     
  2. Download and install cuDNN from:
     https://developer.nvidia.com/cudnn
     (Requires NVIDIA Developer account)
     
  3. Add CUDA and cuDNN bin directories to your PATH environment variable
""")
        elif system == "Darwin":
            print("""
For macOS:
  Note: Recent macOS versions have limited CUDA support.
  Consider using CPU mode instead, or using a Linux environment.
""")
        
        return False
    
    return True
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
        
        try:
            if torch.cuda.is_available():
                cuda_libraries_available = check_cuda_libraries()
                
                if cuda_libraries_available:
                    device = "cuda"
                    
                    gpu_name = torch.cuda.get_device_name(0).lower()
                    if "quadro rtx" in gpu_name and ("6000" in gpu_name or "8000" in gpu_name):
                        compute_type = "float32"
                        print(f"Detected Quadro RTX 6000/8000. Using {compute_type} precision for compatibility...")
                    else:
                        compute_type = "float16"
                    
                    print(f"Attempting to load WhisperX model on {device} using {compute_type} precision...")
                    model = whisperx.load_model("base", device=device, compute_type=compute_type)
                else:
                    print("Missing required CUDA libraries. See installation instructions above.")
                    raise RuntimeError("Missing CUDA libraries")
            else:
                raise RuntimeError("CUDA not available")
        except Exception as e:
            print(f"CUDA error: {str(e)}")
            print(f"Falling back to CPU processing (this will be slower)...")
            device = "cpu"
            compute_type = "float32"
            model = whisperx.load_model("base", device=device, compute_type=compute_type)
        
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
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, language="en")
            
            try:
                align_device = device
                model_a, metadata = whisperx.load_align_model(language_code="en", device=align_device)
                result = whisperx.align(result["segments"], model_a, metadata, audio, device=align_device)
            except Exception as align_err:
                print(f"Warning: Alignment failed: {str(align_err)}")
                print("Continuing with non-aligned transcription...")
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

def save_transcript(result, audio_path=None):
    if not result:
        return False
    
    # Get video_id from metadata or extract from audio_path as fallback
    if "video_id" in result["metadata"]:
        video_id = result["metadata"]["video_id"]
    elif audio_path:
        video_id = os.path.splitext(os.path.basename(audio_path))[0]
    else:
        video_id = f"unknown_{int(time.time())}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcript_{video_id}_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return filename

def main():
    force_cpu = False
    audio_path = None
    
    for arg in sys.argv[1:]:
        if arg == "--cpu":
            force_cpu = True
        elif not arg.startswith("--"):
            audio_path = arg
    
    if not audio_path:
        print("Usage: python transcribe.py <audio_file_path> [--cpu]")
        print("Options:")
        print("  --cpu    Force CPU processing (use if CUDA libraries are missing)")
        return

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    if force_cpu:
        original_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        print("Forcing CPU mode as requested.")
    
    print(f"Processing audio file: {audio_path}")
    result = transcribe_audio(audio_path)
    
    if force_cpu:
        torch.cuda.is_available = original_cuda_available
    
    if result:
        filename = save_transcript(result, audio_path)
        print(f"Transcript saved to {filename}")
        print(f"\nINSTRUCTIONS:")
        print(f"Transfer '{filename}' back to your local computer")
    else:
        print("Failed to transcribe audio")

if __name__ == "__main__":
    main()
