# Converts the videos to mp3 
import os 
import subprocess
import re

# Ensure output directory exists
os.makedirs("Audios", exist_ok=True)

files = os.listdir("videos") 
for file in files: 
    if not file.endswith((".mp4", ".mkv", ".avi", ".mov")):
        continue
        
    # Safe parsing for tutorial number
    num_match = re.search(r'#(\d+)', file)
    if num_match:
        tutorial_number = num_match.group(1)
    else:
        starts_with_num = re.match(r'^(\d+)[_-]', file)
        tutorial_number = starts_with_num.group(1) if starts_with_num else "unknown"
        
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