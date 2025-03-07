import os
import json
import aiohttp
import asyncio
from glob import glob
import time
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import warnings

class TranscriptAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.model = "google/gemini-2.0-flash-001"
        self.cache = {}

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
        
    def chunk_text(self, text, chunk_size=7000, overlap=500):
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            if end < len(text):
                sentence_end = max(
                    text.rfind(". ", start + chunk_size - 500, end),
                    text.rfind("! ", start + chunk_size - 500, end),
                    text.rfind("? ", start + chunk_size - 500, end)
                )
                
                if sentence_end != -1:
                    end = sentence_end + 2
            
            chunks.append(text[start:end])
            
            start = max(start + 1, end - overlap)
        
        return chunks

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

    async def analyze_async(self, text, prompt, max_tokens=1000, cache_key=None):
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
            
        full_prompt = f"{prompt}:\n\n{text}"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": max_tokens
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"API error (status {response.status}): {error_text}")
                        return f"Error: API returned status {response.status}"
                    
                    result = await response.json()
                    
                    if 'choices' in result and len(result['choices']) > 0:
                        choice = result['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                        elif 'text' in choice:
                            content = choice['text']
                        else:
                            print(f"Unexpected choice structure: {choice}")
                            content = "Error: Unable to extract content from API response"
                    else:
                        print(f"Unexpected API response structure: {result}")
                        content = "Error: API response missing expected 'choices' array"
                    
                    if cache_key:
                        self.cache[cache_key] = content
                    
                    return content
        except Exception as e:
            error_message = f"Error calling OpenRouter API: {str(e)}"
            print(error_message)
            return f"Error: {str(e)}"
            
    def analyze(self, text, prompt, max_tokens=1000, cache_key=None):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return self.analyze_async(text, prompt, max_tokens, cache_key)
        else:
            return asyncio.run(self.analyze_async(text, prompt, max_tokens, cache_key))

    async def analyze_stock_opinions_async(self, text, chunk_number, total_chunks):
        chunk_info = f"[Analyzing chunk {chunk_number} of {total_chunks}]"
        print(f"  {chunk_info} Extracting stock opinions and sentiment...")
        
        cache_key = f"stock_analysis_{hash(text)}"
        
        combined_prompt = (
            "Analyze the following transcript and provide TWO sections:\n\n"
            "SECTION 1 - STOCK OPINIONS:\n"
            "Focus ONLY on opinions about stocks, companies, or market sectors mentioned. "
            "Identify specific stock recommendations, predictions, or investment opinions. "
            "Include the stock ticker symbol when mentioned or when you can confidently infer it. "
            "If no stock opinions are found, state that clearly.\n\n"
            "SECTION 2 - SENTIMENT ANALYSIS:\n"
            "For each stock or company mentioned, analyze the sentiment (bullish, bearish, or neutral). "
            "Consider price targets, time horizons, and confidence levels when mentioned. "
            "If no stock opinions are present, simply state that no stock sentiment could be analyzed."
        )
        
        combined_result = await self.analyze_async(text, combined_prompt, max_tokens=2000, cache_key=cache_key)
        
        sections = combined_result.split("SECTION 2 - SENTIMENT ANALYSIS:")
        
        if len(sections) == 2:
            opinions_section = sections[0].replace("SECTION 1 - STOCK OPINIONS:", "").strip()
            sentiment_section = sections[1].strip()
        else:
            opinions_section = combined_result
            sentiment_section = "Failed to extract sentiment section."
        
        return {
            "chunk_number": chunk_number,
            "stock_opinions": opinions_section,
            "stock_sentiment": sentiment_section
        }
        
    def analyze_stock_opinions(self, text, chunk_number, total_chunks):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return self.analyze_stock_opinions_async(text, chunk_number, total_chunks)
        else:
            return asyncio.run(self.analyze_stock_opinions_async(text, chunk_number, total_chunks))

    async def analyze_transcript_async(self, transcript_data):
        full_text = self.extract_full_text(transcript_data)
        video_info = self.get_video_info(transcript_data)
        
        if not full_text:
            return {
                "video_info": video_info,
                "error": "No transcript text found"
            }
        
        print(f"Analyzing transcript for stock opinions: {video_info['title']}")
        
        word_count = len(full_text.split())
        segment_count = len(transcript_data.get('transcript', []))
        
        chunks = self.chunk_text(full_text)
        total_chunks = len(chunks)
        print(f"  Splitting transcript into {total_chunks} chunks for comprehensive analysis")
        
        tasks = []
        for i, chunk in enumerate(chunks, 1):
            task = self.analyze_stock_opinions_async(chunk, i, total_chunks)
            tasks.append(task)
        
        chunk_analyses = await asyncio.gather(*tasks)
        
        print("  Creating consolidated stock analysis...")
        all_opinions = "\n\n".join([f"CHUNK {ca['chunk_number']}:\n{ca['stock_opinions']}" for ca in chunk_analyses])
        
        cache_key = f"consolidated_{hash(all_opinions)}"
        consolidated_summary = await self.analyze_async(
            all_opinions,
            "Provide a comprehensive and organized summary of all stock opinions from the transcript. "
            "Group opinions by company/stock and highlight any conflicting views or repeated mentions across different sections. "
            "Focus only on stocks and investing information. Ignore everything else.",
            cache_key=cache_key
        )
        
        analysis_result = {
            "video_info": video_info,
            "statistics": {
                "word_count": word_count,
                "segment_count": segment_count,
                "chunk_count": total_chunks,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "chunk_analyses": chunk_analyses,
            "consolidated_stock_analysis": consolidated_summary
        }
        
        return analysis_result
        
    def analyze_transcript(self, transcript_data):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return self.analyze_transcript_async(transcript_data)
        else:
            return asyncio.run(self.analyze_transcript_async(transcript_data))

    async def run_batch_analysis_async(self, directory, output_file=None, max_concurrency=3):
        transcript_files = self.get_transcript_files(directory)
        print(f"Found {len(transcript_files)} transcript files")
        
        if not transcript_files:
            print(f"No transcript files found in {directory}")
            return []
        
        all_results = []
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_transcript(file_path):
            async with semaphore:
                print(f"\nAnalyzing: {os.path.basename(file_path)}")
                try:
                    transcript_data = self.load_transcript(file_path)
                    analysis = await self.analyze_transcript_async(transcript_data)
                    
                    result = {
                        "file": os.path.basename(file_path),
                        "analysis": analysis
                    }
                    
                    print(f"Analysis complete for {os.path.basename(file_path)}")
                    return result
                    
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
                    return {
                        "file": os.path.basename(file_path),
                        "error": str(e)
                    }
        
        tasks = [process_transcript(file_path) for file_path in transcript_files]
        results = await asyncio.gather(*tasks)
        all_results.extend(results)
        
        if output_file:
            output_path = os.path.join(os.path.dirname(__file__), output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2)
            print(f"Analysis results saved to {output_path}")
        
        return all_results
        
    def run_batch_analysis(self, directory, output_file=None):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return self.run_batch_analysis_async(directory, output_file)
        else:
            return asyncio.run(self.run_batch_analysis_async(directory, output_file))


def silence_event_loop_closed(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if str(e) != 'Event loop is closed':
                raise
    return wrapper

if sys.platform == 'win32':
    asyncio.proactor_events._ProactorBasePipeTransport.__del__ = silence_event_loop_closed(
        asyncio.proactor_events._ProactorBasePipeTransport.__del__
    )

async def async_main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze video transcripts using OpenRouter API with Deep Seek V3")
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--directory", default="..", help="Directory containing transcript files")
    parser.add_argument("--output", default="transcript_analysis.json", help="Output file for analysis results")
    parser.add_argument("--max-concurrency", type=int, default=3, help="Maximum number of concurrent transcript analyses")
    
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
    await analyzer.run_batch_analysis_async(directory, args.output, args.max_concurrency)

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nExiting gracefully due to keyboard interrupt...")

if __name__ == "__main__":
    main()
