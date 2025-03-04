import os
import json
import requests
from glob import glob
import time
import sys
from datetime import datetime

class TranscriptAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.model = "deepseek/deepseek-chat"

    def load_transcript(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_transcript_files(self, directory):
        json_files = glob(os.path.join(directory, "*.json"))
        
        transcript_files = []
        for file_path in json_files:
            try:
                if 'transcript' in os.path.basename(file_path).lower():
                    transcript_files.append(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if 'transcript' in data and isinstance(data['transcript'], list):
                                transcript_files.append(file_path)
                        except:
                            pass
            except:
                continue
        
        return transcript_files

    def extract_full_text(self, transcript_data):
        if 'transcript' not in transcript_data:
            return ""
            
        text_segments = [segment['text'] for segment in transcript_data['transcript']]
        return " ".join(text_segments)

    def get_video_info(self, transcript_data):
        metadata = transcript_data.get('metadata', {})
        return {
            "video_id": metadata.get('video_id', 'unknown'),
            "title": metadata.get('title', 'unknown'),
            "uploader": metadata.get('uploader', 'unknown'),
            "duration": metadata.get('duration', 0),
            "duration_formatted": self.format_duration(metadata.get('duration', 0))
        }

    def format_duration(self, seconds):
        if not seconds:
            return "00:00:00"
            
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def analyze_with_deepseek(self, text, prompt, max_tokens=1000):
        max_text_length = 8000
        if len(text) > max_text_length:
            text = text[:max_text_length] + "... [text truncated due to length]"
        
        full_prompt = f"{prompt}:\n\n{text}"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return f"Error: {str(e)}"

    def analyze_transcript(self, transcript_data):
        full_text = self.extract_full_text(transcript_data)
        video_info = self.get_video_info(transcript_data)
        
        if not full_text:
            return {
                "video_info": video_info,
                "error": "No transcript text found"
            }
        
        print(f"Analyzing transcript: {video_info['title']}")
        
        word_count = len(full_text.split())
        segment_count = len(transcript_data.get('transcript', []))
        
        print("  Generating summary...")
        summary = self.analyze_with_deepseek(
            full_text,
            "Provide a concise summary of the following transcript"
        )
        
        print("  Identifying key topics...")
        key_topics = self.analyze_with_deepseek(
            full_text,
            "Identify and list the main topics discussed in the following transcript"
        )
        
        print("  Analyzing sentiment...")
        sentiment = self.analyze_with_deepseek(
            full_text,
            "Analyze the overall sentiment and emotional tone of the following transcript"
        )
        
        print("  Extracting interesting quotes...")
        interesting_quotes = self.analyze_with_deepseek(
            full_text,
            "Extract 3-5 notable or interesting quotes from the following transcript, with context"
        )
        
        print("  Identifying key entities...")
        key_entities = self.analyze_with_deepseek(
            full_text,
            "Identify key people, organizations, products, or concepts mentioned in the transcript"
        )
        
        print("  Generating content classification...")
        content_classification = self.analyze_with_deepseek(
            full_text,
            "Classify the content of this transcript (e.g., educational, entertainment, news, interview, etc.) and explain why"
        )
        
        analysis_result = {
            "video_info": video_info,
            "statistics": {
                "word_count": word_count,
                "segment_count": segment_count,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "analysis": {
                "summary": summary,
                "key_topics": key_topics,
                "sentiment": sentiment,
                "interesting_quotes": interesting_quotes,
                "key_entities": key_entities,
                "content_classification": content_classification
            }
        }
        
        return analysis_result

    def run_batch_analysis(self, directory, output_file=None):
        transcript_files = self.get_transcript_files(directory)
        print(f"Found {len(transcript_files)} transcript files")
        
        if not transcript_files:
            print(f"No transcript files found in {directory}")
            return []
        
        all_results = []
        
        for file_path in transcript_files:
            print(f"\nAnalyzing: {os.path.basename(file_path)}")
            try:
                transcript_data = self.load_transcript(file_path)
                analysis = self.analyze_transcript(transcript_data)
                
                result = {
                    "file": os.path.basename(file_path),
                    "analysis": analysis
                }
                
                all_results.append(result)
                print(f"Analysis complete for {os.path.basename(file_path)}")
                
                if len(transcript_files) > 1:
                    print("Waiting before next analysis...")
                    time.sleep(5)
                
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                all_results.append({
                    "file": os.path.basename(file_path),
                    "error": str(e)
                })
        
        if output_file:
            output_path = os.path.join(os.path.dirname(__file__), output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2)
            print(f"Analysis results saved to {output_path}")
        
        return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze video transcripts using OpenRouter API with Deep Seek V3")
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--directory", default="../download_and_transcribe", help="Directory containing transcript files")
    parser.add_argument("--output", default="transcript_analysis.json", help="Output file for analysis results")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OpenRouter API key is required. Provide it with --api-key or set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.directory):
        directory = os.path.abspath(os.path.join(script_dir, args.directory))
    else:
        directory = args.directory
    
    analyzer = TranscriptAnalyzer(api_key)
    analyzer.run_batch_analysis(directory, args.output)


if __name__ == "__main__":
    main()