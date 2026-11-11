from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, model_validator
from transformers import AutoTokenizer

_MAX_OVERLAP_PERCENT = 30


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(
        default=384,
        ge=32,
        le=2048,
        description="The size of the chunk to create"
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="The overlap between chunks"
    )
    tokenizer_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="The tokenizer model to use"
    )

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap > self.chunk_size * _MAX_OVERLAP_PERCENT // 100:
            raise ValueError(
                f"Chunk overlap {self.chunk_overlap} is greater than {_MAX_OVERLAP_PERCENT}% of chunk size {self.chunk_size}"
            )
        return self

    def output_directory(self, base_dir: Path) -> Path:
        """Return the configuration-specific directory under ``base_dir``."""
        return Path(base_dir) / f"c{self.chunk_size}_o{self.chunk_overlap}"


class RecursiveTokenChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_model,
            trust_remote_code=True
        )

        self._splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self._tokenizer,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
            keep_separator="end"
        )

    def chunk_report(self, report: dict) -> list[dict]:
        """Split rendered pages independently, preserving source character offsets.

        Chunk indexes are zero-based within each page. Character offsets use
        Python string indexing, with an inclusive start and exclusive end.
        Token counts exclude special tokens.
        """
        document_id = report["metadata"]["document_id"]
        chunks = []
        for page in report["pages"]:
            page_number = page["page"]
            page_text = page["text"]
            spans = self._locate_spans(
                page_text,
                self._splitter.split_text(page_text),
                document_id=document_id,
                page_number=page_number,
            )
            for chunk_index, (text, char_start, char_end) in enumerate(spans):
                chunks.append({
                    "metadata": {
                        "chunk_id": f"{document_id}:p{page_number}:c{chunk_index}",
                        "document_id": document_id,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "token_count": len(self._tokenizer.encode(
                            text, add_special_tokens=False
                        )),
                        "char_start": char_start,
                        "char_end": char_end,
                    },
                    "text": text,
                })
        return chunks

    def _locate_spans(
        self,
        page_text: str,
        parts: list[str],
        *,
        document_id: str,
        page_number: int,
    ) -> list[tuple[str, int, int]]:
        """Recover a source span for each splitter part.

        The window opens at ``previous_end - len(text)``. A candidate that still
        reaches the previous end cannot start farther back than its own length,
        so a hit there overlaps ``page_text[start:previous_end]``. Overlap is
        measured with ``tokenize``, the splitter's length function. Earlier parts
        take the leftmost in-budget hit. The last part is anchored to its final
        occurrence, including a repeated table delimiter, and kept only when
        that overlap is in budget.
        """
        located: list[tuple[str, int, int]] = []
        search_start = 0
        previous_end = 0
        last = len(parts) - 1
        for index, text in enumerate(parts):
            window = max(search_start, previous_end - len(text))
            char_start = self._match_start(
                page_text,
                text,
                window,
                previous_end,
                is_last=index == last,
            )
            if char_start == -1:
                raise ValueError(
                    f"Chunk {index} on page {page_number} of "
                    f"{document_id} could not be located in its source text"
                )
            char_end = char_start + len(text)
            located.append((text, char_start, char_end))
            search_start = char_start + 1
            previous_end = char_end
        return located

    def _match_start(
        self,
        page_text: str,
        text: str,
        window: int,
        previous_end: int,
        *,
        is_last: bool,
    ) -> int:
        if is_last:
            char_start = page_text.rfind(text, window)
            if (
                char_start == -1
                or char_start >= previous_end
                or self._within_overlap(page_text, char_start, previous_end)
            ):
                return char_start
            return -1

        char_start = page_text.find(text, window)
        while char_start != -1 and char_start < previous_end:
            if self._within_overlap(page_text, char_start, previous_end):
                return char_start
            char_start = page_text.find(text, char_start + 1)
        return char_start

    def _within_overlap(
        self,
        page_text: str,
        char_start: int,
        previous_end: int,
    ) -> bool:
        overlap_tokens = len(self._tokenizer.tokenize(
            page_text[char_start:previous_end]
        ))
        return overlap_tokens <= self.config.chunk_overlap
