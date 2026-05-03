import argparse
import json
import os
import pickle
import random
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import requests
import tiktoken


SOURCE_URLS = [
    "https://www.admissions.msstate.edu/apply/admission-process/freshman-admissions",
    "https://www.admissions.msstate.edu/tuition-scholarships-financial-aid",
    "https://www.admissions.msstate.edu/scholarships",
    "https://www.sfa.msstate.edu/cost/",
    "https://catalog.msstate.edu/undergraduate/collegesanddegreeprograms/",
    "https://www.registrar.msstate.edu/calendars/academic-calendar/2026/spring",
    "https://www.registrar.msstate.edu/calendars/academic-calendar/2026/fall",
    "https://www.housing.msstate.edu/",
    "https://www.housing.msstate.edu/housing-options/residence-halls",
    "https://www.visit.msstate.edu/",
    "https://www.transportation.msstate.edu/parking/permits",
    "https://www.transportation.msstate.edu/parking/visitor",
]

QUESTION_HINTS = {
    "admissions.msstate.edu": [
        "How do I apply to Mississippi State?",
        "What should prospective students know about Mississippi State admissions?",
        "What information does Mississippi State provide for new applicants?",
    ],
    "sfa.msstate.edu": [
        "How much does Mississippi State cost?",
        "What should I know about Mississippi State financial aid and cost of attendance?",
        "What costs should prospective Mississippi State students plan for?",
    ],
    "catalog.msstate.edu": [
        "What academic programs does Mississippi State offer?",
        "Where can I find Mississippi State colleges and degree programs?",
        "What should prospective students know about Mississippi State academics?",
    ],
    "registrar.msstate.edu": [
        "Where can I find Mississippi State academic calendar information?",
        "What academic calendar information does Mississippi State publish?",
        "How can students check important Mississippi State dates?",
    ],
    "housing.msstate.edu": [
        "What housing options does Mississippi State offer?",
        "What should new students know about Mississippi State housing?",
        "Where can I find Mississippi State residence hall information?",
    ],
    "visit.msstate.edu": [
        "How can I visit Mississippi State?",
        "What should prospective students know about visiting Mississippi State?",
        "Where can I find Mississippi State campus tour information?",
    ],
    "transportation.msstate.edu": [
        "What should I know about parking at Mississippi State?",
        "Where can I find Mississippi State transportation and parking information?",
        "What parking information does Mississippi State provide for students and visitors?",
    ],
}


@dataclass
class Page:
    url: str
    title: str
    text: str
    sections: list[tuple[str, str]]


def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "")
    return value.strip()


def normalize_heading(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"[^A-Za-z0-9 &/,.:'()-]+", "", value)
    return value.strip(" -")


def is_allowed_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == "msstate.edu" or host.endswith(".msstate.edu")


def fetch_html(url: str, timeout: int) -> str:
    if not is_allowed_url(url):
        raise ValueError(f"Refusing non-MSU URL: {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; MSU-GPT2-Student-Project/1.0; "
            "+https://www.msstate.edu)"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


SKIP_SECTION_HEADINGS = {
    "filters",
    "filters:",
    "priority date",
    "award information",
    "award criteria",
    "disclosures",
}


def extract_page(url: str, html: str, strip_chrome: bool = True) -> Page:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: beautifulsoup4. Install it with "
            "`pip install beautifulsoup4` before running this script."
        ) from exc

    soup = BeautifulSoup(html, "html.parser")
    removable = ["script", "style", "noscript", "svg", "form"]
    if strip_chrome:
        removable.extend(["nav", "header", "footer", "aside"])
    for tag in soup(removable):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = clean_text(soup.title.string)
    first_h1 = soup.find("h1")
    if first_h1:
        title = clean_text(first_h1.get_text(" ", strip=True)) or title
    title = title or url

    content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )
    content_nodes = content.find_all(["h1", "h2", "h3", "p", "li", "th", "td"])
    chunks = []
    sections = []
    current_heading = title
    current_parts = []

    for node in content_nodes:
        text = clean_text(node.get_text(" ", strip=True))
        if not text:
            continue
        name = node.name.lower()
        if name in {"h1", "h2", "h3"}:
            if current_parts:
                sections.append((current_heading, clean_text(" ".join(current_parts))))
                current_parts = []
            current_heading = normalize_heading(text) or title
            chunks.append(current_heading)
        else:
            chunks.append(text)
            current_parts.append(text)

    if current_parts:
        sections.append((current_heading, clean_text(" ".join(current_parts))))

    page_text = clean_text(" ".join(chunks))
    return Page(url=url, title=title, text=page_text, sections=sections)


def trim_to_sentence(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(".", 1)[0].strip()
    if len(clipped) < max_chars * 0.45:
        clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return clipped.rstrip(".") + "."


def host_for(url: str) -> str:
    return urlparse(url).netloc.lower()


def page_questions(page: Page) -> list[str]:
    host = host_for(page.url)
    for key, questions in QUESTION_HINTS.items():
        if host.endswith(key):
            return questions
    return [
        f"What should I know about {page.title} at Mississippi State?",
        f"Where can I find information about {page.title} at Mississippi State?",
    ]


def make_answer(body: str, url: str, max_chars: int) -> str:
    answer = trim_to_sentence(body, max_chars)
    return f"{answer} Source: {url}"


def examples_from_page(page: Page, max_chars: int) -> list[str]:
    examples = []

    for question in page_questions(page):
        answer = make_answer(page.text, page.url, max_chars)
        examples.append(f"User: {question}\nAssistant: {answer}\n\n")

    seen_headings = set()
    for heading, body in page.sections:
        heading = normalize_heading(heading)
        heading_key = heading.lower()
        if (
            not heading
            or heading_key in seen_headings
            or heading_key in SKIP_SECTION_HEADINGS
            or len(body) < 120
        ):
            continue
        seen_headings.add(heading_key)
        question = f"What does Mississippi State say about {heading}?"
        answer = make_answer(body, page.url, max_chars)
        examples.append(f"User: {question}\nAssistant: {answer}\n\n")

    return examples


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare an official Mississippi State University GPT-2 fine-tuning dataset."
    )
    parser.add_argument("--out_dir", type=str, default=os.path.join("data", "msstate_chat"))
    parser.add_argument("--text_out", type=str, default="input.txt")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max_answer_chars", type=int, default=900)
    parser.add_argument("--min_page_chars", type=int, default=300)
    parser.add_argument("--url", action="append", default=[], help="Additional official msstate.edu URL.")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    urls = list(dict.fromkeys(SOURCE_URLS + args.url))
    pages = []
    failures = []

    for url in urls:
        try:
            html = fetch_html(url, timeout=args.timeout)
            page = extract_page(url, html)
            if len(page.text) < args.min_page_chars:
                page = extract_page(url, html, strip_chrome=False)
            if len(page.text) < args.min_page_chars:
                failures.append({"url": url, "error": "too little extracted text"})
                continue
            pages.append(page)
            print(f"fetched: {url} ({len(page.text):,} chars)")
        except Exception as exc:
            failures.append({"url": url, "error": str(exc)})
            print(f"failed: {url} ({exc})")

    examples = []
    for page in pages:
        examples.extend(examples_from_page(page, max_chars=args.max_answer_chars))

    examples = list(dict.fromkeys(examples))
    if args.repeat > 1:
        examples = examples * args.repeat
    random.shuffle(examples)

    if len(examples) < 20:
        raise RuntimeError(
            f"Only built {len(examples)} examples. Check network access or source extraction."
        )

    split_idx = int(args.train_ratio * len(examples))
    split_idx = max(1, min(split_idx, len(examples) - 1))
    train_text = "".join(examples[:split_idx])
    val_text = "".join(examples[split_idx:])
    all_text = train_text + val_text

    with open(args.text_out, "w", encoding="utf-8") as f:
        f.write(all_text)

    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_text)
    val_ids = enc.encode_ordinary(val_text)

    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    meta_path = os.path.join(args.out_dir, "meta.pkl")
    sources_path = os.path.join(args.out_dir, "sources.jsonl")
    failures_path = os.path.join(args.out_dir, "failures.jsonl")

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    meta = {
        "vocab_size": enc.n_vocab,
        "tokenizer": "gpt2",
        "dataset": "official_msstate_pages",
        "num_pages": len(pages),
        "num_examples": len(examples),
        "train_ratio": args.train_ratio,
        "repeat": args.repeat,
        "source_urls": [page.url for page in pages],
        "text_out": args.text_out,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    write_jsonl(
        sources_path,
        (
            {
                "url": page.url,
                "title": page.title,
                "chars": len(page.text),
                "sections": len(page.sections),
            }
            for page in pages
        ),
    )
    write_jsonl(failures_path, failures)

    print(f"saved text: {args.text_out} ({len(all_text):,} chars)")
    print(f"saved: {train_path} ({train_arr.size:,} tokens)")
    print(f"saved: {val_path} ({val_arr.size:,} tokens)")
    print(f"saved: {meta_path}")
    print(f"saved: {sources_path}")
    if failures:
        print(f"saved failures: {failures_path} ({len(failures)} failed URLs)")


if __name__ == "__main__":
    main()
