# Converts the videos to mp3 
import os 
import subprocess
import re

# Ensure output directory exists
os.makedirs("Audios", exist_ok=True)

def extract_video_number(filename):
    filename_lower = filename.lower()
    
    # 1. Match # followed by digits (e.g. #18)
    match = re.search(r'#(\d+)', filename)
    if match:
        return match.group(1)
        
    # 2. Match lec-6, lec 6, lec6, lecture 6
    match = re.search(r'lec(?:ture)?[- \s]*(\d+)', filename_lower)
    if match:
        return match.group(1)
        
    # 3. Match tutorial #18 or tutorial 18 or tut 18
    match = re.search(r'tut(?:orial)?[- \s#]*(\d+)', filename_lower)
    if match:
        return match.group(1)
        
    # 4. Match starts with digits followed by _ or -
    match = re.match(r'^(\d+)[_-]', filename)
    if match:
        return match.group(1)
        
    # 5. Fallback: match any digits in the filename
    match = re.search(r'\b(\d+)\b', filename)
    if match:
        return match.group(1)
        
    return "unknown"

files = os.listdir("videos") 
for file in files: 
    if not file.endswith((".mp4", ".mkv", ".avi", ".mov")):
        continue
        
    # Safe parsing for tutorial number
    tutorial_number = extract_video_number(file)
        
    # Safe parsing for file name
    if " | " in file:
        file_name = file.split(" | ")[0].strip()
    else:
        file_name = os.path.splitext(file)[0].strip()
        
    output_path = f"Audios/{tutorial_number}_{file_name}.mp3"
    
    # Skip conversion if file already exists
    if os.path.exists(output_path):
        print(f"Skipping '{file}' (already converted to '{output_path}')")
        continue
        
    print(f"Converting '{file}' -> '{output_path}'...")
    subprocess.run(["ffmpeg", "-y", "-i", f"videos/{file}", output_path])