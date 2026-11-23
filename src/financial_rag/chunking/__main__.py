"""Command-line entry point for chunking rendered reports."""

import argparse
import json
import logging
from pathlib import Path

from financial_rag.chunking.chunker import ChunkingConfig, RecursiveTokenChunker

DEFAULT_INPUT_DIR = Path("data/processed/rendered_reports")
DEFAULT_OUTPUT_DIR = Path("data/processed/chunks")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split rendered reports into page-bounded JSONL chunks."
    )
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-Embedding-0.6B")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        config = ChunkingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            tokenizer_model=args.tokenizer_model,
        )
        if args.reports:
            paths = args.reports
        else:
            if not args.input_dir.is_dir():
                raise ValueError(f"Rendered report directory not found: {args.input_dir}")
            paths = sorted(args.input_dir.glob("*.json"))
        if not paths:
            raise ValueError("No rendered reports found")

        chunker = RecursiveTokenChunker(config)
        output_dir = config.output_directory(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_count = 0
        for path in paths:
            report = json.loads(path.read_text(encoding="utf-8"))
            document_id = report["metadata"]["document_id"]
            if document_id != path.stem:
                raise ValueError(f"{path}: document_id must match the filename stem")
            chunks = chunker.chunk_report(report)
            output_path = output_dir / f"{document_id}.jsonl"
            with output_path.open("w", encoding="utf-8", newline="\n") as output:
                for chunk in chunks:
                    output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            chunk_count += len(chunks)
            logging.info("Chunked %s: %d chunks", path.name, len(chunks))
    except (OSError, ValueError, KeyError, TypeError) as error:
        logging.error("%s", error)
        return 1

    print(f"Wrote {chunk_count} chunks from {len(paths)} report(s) into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
