from faster_whisper import WhisperModel
import json
import os
import math
from tqdm import tqdm

model = WhisperModel("base", compute_type="int8")

os.makedirs("jsons", exist_ok=True)
audios = [f for f in os.listdir("Audios") if f.endswith(".mp3")]

n = 5  # Group size config

for audio in tqdm(audios): 
    if "_" in audio:
        parts = audio.split("_", 1)
        number = parts[0]
        title = os.path.splitext(parts[1])[0]
        print(number, title)

        segments, info = model.transcribe(
            os.path.join("Audios", audio),
            language="hi",
            task="translate"
        )

        raw_chunks = []
        for segment in segments:
            raw_chunks.append({
                "number": number,
                "title": title,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

        if not raw_chunks:
            continue

        # Group chunks in memory directly by n=5
        grouped_chunks = []
        num_chunks = len(raw_chunks)
        num_groups = math.ceil(num_chunks / n)

        for i in range(num_groups):
            st_idx = i * n
            end_idx = min((i + 1) * n, num_chunks)
            chunk_group = raw_chunks[st_idx:end_idx]

            grouped_chunks.append({
                "number": number,
                "title": title,
                "start": chunk_group[0]["start"],
                "end": chunk_group[-1]["end"],
                "text": " ".join(c["text"] for c in chunk_group)
            })

        full_text = " ".join([c["text"] for c in raw_chunks])
        chunks_with_metadata = {"chunks": grouped_chunks, "text": full_text}

        name = os.path.splitext(audio)[0]
        with open(os.path.join("jsons", f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, indent=4)

print("✅ Processing complete.")