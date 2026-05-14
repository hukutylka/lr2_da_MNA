import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types


TOPICS = [
    "rides_attractions",
    "staff_service",
    "queues_crowding",
    "price_food",
    "facilities_cleanliness",
    "weather_comfort",
    "location_transport",
    "general_experience",
    "other",
]


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "review_id": {"type": "integer"},
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                    },
                    "topic": {
                        "type": "string",
                        "enum": TOPICS,
                    },
                    "short_summary": {"type": "string"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                },
                "required": [
                    "review_id",
                    "sentiment",
                    "topic",
                    "short_summary",
                    "keywords",
                    "confidence",
                ],
            },
        }
    },
    "required": ["results"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Анализ отзывов Disneyland через Gemini API"
    )

    parser.add_argument("--input", required=True, help="Путь к CSV-файлу")
    parser.add_argument("--output", required=True, help="Путь к JSON-файлу")
    parser.add_argument("--limit", type=int, default=5, help="Количество отзывов")
    parser.add_argument("--batch-size", type=int, default=5, help="Размер пакета")
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        help="Модель Gemini",
    )

    return parser.parse_args()


def read_reviews(path, limit):
    df = pd.read_csv(path, encoding="cp1252")

    if limit:
        df = df.head(limit)

    return df


def make_records(df):
    records = []

    for _, row in df.iterrows():
        records.append(
            {
                "review_id": int(row["Review_ID"]),
                "rating": int(row["Rating"]),
                "year_month": str(row["Year_Month"]),
                "reviewer_location": str(row["Reviewer_Location"]),
                "branch": str(row["Branch"]),
                "review_text": str(row["Review_Text"]),
            }
        )

    return records


def analyze_batch(client, model, records):
    prompt = f"""
Ты бизнес аналитик парков сети диснейленда.

Проанализируй каждый отзыв и для каждого определи:
1. sentiment: positive, negative или neutral;
2. topic: одну основную тему из списка:
{", ".join(TOPICS)}
3. short_summary: краткое содержание на русском языке;
4. keywords: 3-6 ключевых слов на русском языке;
5. confidence: уверенность от 0 до 1.

Верни только JSON по заданной схеме.

Отзывы:
{json.dumps(records, ensure_ascii=False, indent=2)}
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    data = json.loads(response.text)
    return data["results"]


def save_results(results, output_path, input_path, model):
    output = {
        "task": "review_sentiment_and_topic_classification",
        "api_provider": "Google Gemini API",
        "input_file": input_path,
        "model": model,
        "items_count": len(results),
        "results": results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    load_dotenv()
    args = parse_args()

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    df = read_reviews(args.input, args.limit)

    all_results = []

    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start : start + args.batch_size]
        records = make_records(batch)

        print(f"Отправка отзывов {start + 1}-{min(start + args.batch_size, len(df))}")

        results = analyze_batch(client, args.model, records)
        all_results.extend(results)

    save_results(
        results=all_results,
        output_path=args.output,
        input_path=args.input,
        model=args.model,
    )

    print(f"Готово. Результат сохранен в файл: {args.output}")


if __name__ == "__main__":
    main()