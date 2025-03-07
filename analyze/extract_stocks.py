import os
import json
import re
from collections import defaultdict

def extract_stock_opinions():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'transcript_analysis.json')
    output_file = os.path.join(script_dir, 'stock_opinions.json')
    text_output = os.path.join(script_dir, 'stock_opinions.txt')
    
    print(f"Loading analysis from {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
    except Exception as e:
        print(f"Error loading analysis file: {e}")
        return
    
    stock_opinions = defaultdict(list)
    
    for item in analysis_data:
        if 'analysis' not in item:
            continue
        
        for chunk in item['analysis'].get('chunk_analyses', []):
            chunk_num = chunk.get('chunk_number', 'unknown')
            
            opinions_text = chunk.get('stock_opinions', '')
            sentiment_text = chunk.get('stock_sentiment', '')
            
            for section_text in [opinions_text, sentiment_text]:
                lines = section_text.split('\n')
                
                for line in lines:
                    if ('*' in line or '**' in line) and ':' in line:
                        stock_info = extract_stock_info(line)
                        
                        if stock_info:
                            name, ticker, opinion = stock_info
                            if not name:
                                continue
                                
                            key = ticker if ticker else name
                            
                            if opinion and opinion not in [o['opinion'] for o in stock_opinions[key]]:
                                stock_opinions[key].append({
                                    'name': name,
                                    'ticker': ticker,
                                    'opinion': opinion,
                                    'chunk': chunk_num
                                })
    
    results = []
    for key, opinions in stock_opinions.items():
        if not opinions:
            continue
            
        stock_info = {
            'name': opinions[0]['name'],
            'ticker': opinions[0]['ticker'],
            'opinions': []
        }
        
        for opinion_data in opinions:
            if opinion_data.get('opinion'):
                stock_info['opinions'].append({
                    'text': opinion_data['opinion'],
                    'chunk': opinion_data['chunk']
                })
        
        results.append(stock_info)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Stock opinions saved to {output_file}")
    
    with open(text_output, 'w', encoding='utf-8') as f:
        f.write("STOCK OPINIONS ANALYSIS\n")
        f.write("======================\n\n")
        
        for stock in sorted(results, key=lambda x: x['name'] if x['name'] else ''):
            name = stock['name']
            ticker = f"({stock['ticker']})" if stock['ticker'] else ""
            
            f.write(f"{name} {ticker}\n")
            f.write("-" * len(f"{name} {ticker}") + "\n")
            
            for opinion in stock['opinions']:
                f.write(f"• {opinion['text']} (Chunk {opinion['chunk']})\n")
            
            f.write("\n\n")
    
    print(f"Text report saved to {text_output}")

def extract_stock_info(line):
    stock_name = None
    ticker = None
    opinion = None
    
    cleaned_line = line.replace('*', '').strip()
    
    if '(' in cleaned_line and ')' in cleaned_line:
        parts = cleaned_line.split('(', 1)
        stock_name = parts[0].strip()
        
        ticker_part = parts[1].split(')', 1)
        ticker = ticker_part[0].strip()
        
        if ':' in ticker_part[1]:
            opinion = ticker_part[1].split(':', 1)[1].strip()
    else:
        parts = cleaned_line.split(':', 1)
        if len(parts) > 0:
            stock_name = parts[0].strip()
            if len(parts) > 1:
                opinion = parts[1].strip()
    
    return stock_name, ticker, opinion

if __name__ == "__main__":
    extract_stock_opinions()