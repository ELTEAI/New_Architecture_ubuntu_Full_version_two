#Locol
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 使用 Tests 目录下 .env 中的 OPENAI_API_KEY
_TESTS_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(__file__).resolve().parent
ENV_FILE = _TESTS_DIR / ".env"
INPUT_CSV = DATASET_DIR / "prompts.csv"
OUTPUT_DIR = DATASET_DIR / "audio"

load_dotenv(ENV_FILE)
client = OpenAI()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini-tts"

# 生成 3 种声音，避免测试集只依赖单一说话人
VOICES = ["alloy", "nova", "echo"]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_CSV}. Please create prompts.csv first.")

    with INPUT_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(
            f"[Error] 未从 {INPUT_CSV} 读取到任何数据行：文件为空、未保存，或只有表头。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[Info] 已加载 {len(rows)} 行，将写入 {OUTPUT_DIR}（{len(VOICES)} 种声线）。")

    for row in rows:
        sample_id = row["id"]
        text = row["text"]

        for voice in VOICES:
            out_path = OUTPUT_DIR / f"{sample_id}_{voice}.wav"

            if out_path.exists():
                print(f"[Skip] {out_path}")
                continue

            print(f"[TTS] {sample_id} | voice={voice} | text={text}")

            with client.audio.speech.with_streaming_response.create(
                model=MODEL,
                voice=voice,
                input=text,
                response_format="wav",
            ) as response:
                response.stream_to_file(out_path)

            print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
