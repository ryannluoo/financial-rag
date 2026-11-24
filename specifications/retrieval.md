# Retrieval specification

## Scope

The retrieval stage (`financial_rag.retrieval`) performs online multi-stage retrieval over an indexed Qdrant collection. It consists of three in-memory components:

1. **Document routing (`router.py`)** — maps company names mentioned in a question to report `document_id`s.
2. **First-stage retrieval (`retriever.py`)** — searches Qdrant using dense vectors, optionally fusing the dense ranking with BM25 full-text search. Either mode supports filtering by `document_id`.
3. **Second-stage reranking (`reranker.py`)** — scores `(query, chunk.text)` pairs with a cross-encoder and returns the top reranked chunks.

Retrieval outputs structured chunk objects (`RetrievedChunk` → `RerankedChunk`) with citation metadata and scores. It does not format LLM prompts or write disk artifacts; prompt assembly belongs to `financial_rag.generation`.

## Document routing

`load_company_to_document_id(catalog_path)` reads `data/raw/document_catalog.json`, where each top-level key is a `document_id` mapping to an object with a non-empty `company_name` string. Case-folded duplicate company names are rejected.

`resolve_document_ids(question, company_to_document_id)` returns deduplicated `document_id`s in order of first mention in `question`:

- Matching case-folds both company names and the question and uses word boundaries (`(?<!\w)...(?!\w)`).
- Candidate company names are ordered longest-first so a longer name takes precedence over a prefix match.
- Empty/whitespace questions or empty catalogs return `[]`.

## First-stage retrieval

`build_document_filter(document_ids)` builds a Qdrant `Filter` matching `document_id` via `MatchAny` with deduplicated IDs, or returns `None` when `document_ids` is empty.

### Configuration and output

`RetrievalConfig` (frozen Pydantic model):

| Field | Default | Requirement |
| --- | --- | --- |
| `collection_name` | Required | Non-empty Qdrant collection name. |
| `top_k` | `20` | Positive integer (`>= 1`). Maximum chunks returned per query. |
| `vector_name` | `"dense"` | Non-empty named vector key in Qdrant. |
| `batch_size` | `32` | Positive integer (`>= 1`) per `query_batch_points` RPC. |
| `bm25_weight` | `0.0` | Finite, nonnegative BM25 fusion weight. `0.0` selects dense search. Positive values enable hybrid search with a fixed dense weight of `1.0`. |

`RetrievedChunk` (frozen Pydantic model) carries flat citation payload fields (`chunk_id`, `document_id`, `page`, `chunk_index`, `token_count`, `char_start`, `char_end`, `text`) plus the Qdrant `score`. The indexed cosine collection returns cosine similarity in dense mode and the weighted reciprocal rank fusion (RRF) score in hybrid mode.

### Behavior

`QdrantRetriever(config, client, embedder)` exposes:

- `retrieve(query, *, top_k=None, query_filter=None) -> list[RetrievedChunk]`
- `retrieve_batch(queries, *, top_k=None, query_filters=None) -> list[list[RetrievedChunk]]`

Queries are encoded via `embedder.embed_query(queries)` and verified to have shape `(len(queries), embedder.dimension)` with finite values. A per-call `top_k` overrides the configured limit.

- **Dense mode** sends one vector query per question with the selected vector name, report filter, and effective `top_k`.
- **Hybrid mode** uses two Qdrant prefetches. The first searches the dense vector. The second searches the original question against the BM25 index on the `text` payload. Each prefetch uses the effective `top_k` and the same report filter. Qdrant fuses the two ranked lists using weights `[1.0, bm25_weight]` and returns up to `top_k` distinct chunks. `bm25_weight=1.0` gives equal weights. Collections built by the [indexer](indexing.md) already have the required BM25 text index.

Both request types use `with_vector=False` and `with_payload=True`. The retriever sends requests in batches of `config.batch_size` and validates each returned point payload into `RetrievedChunk`. If fewer eligible chunks are available, the result can contain fewer than `top_k`. Empty `queries` returns `[]` without invoking the embedder or Qdrant.

## Second-stage reranking

### Supported models

`RerankingModel` (`StrEnum`):

| Enum member | Hugging Face model ID | `trust_remote_code` |
| --- | --- | --- |
| `QWEN3_0_6B` | `Qwen/Qwen3-Reranker-0.6B` | `False` |
| `ZERANK_1_SMALL` | `zeroentropy/zerank-1-small-reranker` | `True` |

Both models are loaded through `sentence_transformers.CrossEncoder` (`sentence-transformers==6.2.0`).

Run ZeRank and Qwen in separate Python processes. ZeRank's loader replaces `CrossEncoder.predict()` and `.to()` process-wide. It also loads its scoring weights on the first `predict()` call, so warm up before measuring steady-state latency. See the [ZeRank loader](https://huggingface.co/zeroentropy/zerank-1-small-reranker/blob/main/modeling_zeranker.py).

### Configuration and output

`RerankingConfig` (frozen Pydantic model):

| Field | Default | Requirement |
| --- | --- | --- |
| `reranking_model` | `RerankingModel.QWEN3_0_6B` | Member of `RerankingModel`. |
| `top_n` | `5` | Positive integer (`>= 1`). |
| `batch_size` | `16` | Positive integer (`>= 1`) passed to `CrossEncoder.predict`. ZeRank batches internally. |
| `device` | `None` | Optional device string (e.g., `"cuda"`, `"cpu"`). |
| `model_dtype` | `None` | Optional `"float32"` or `"float16"` model loader dtype. ZeRank loads its scoring weights with its own automatic dtype. |

`RerankedChunk` (frozen Pydantic model) composes `chunk: RetrievedChunk` and `rerank_score: float`, preserving the first-stage `chunk.score` alongside the cross-encoder score.

### Behavior

`CrossEncoderReranker(config)` exposes:

- `rerank(query, candidates, *, top_n=None) -> list[RerankedChunk]`

Empty `candidates` returns `[]` without calling `predict()`. Otherwise, it scores `(query, candidate.text)` pairs with `show_progress_bar=False` and `convert_to_numpy=True`, converts the scores to an array, flattens `(N, 1)` outputs to `(N,)`, verifies shape and finiteness, and sorts in descending order of `rerank_score`, preserving input order on ties. A per-call `top_n` overrides the configured limit.

## Validation and error handling

All three modules fail fast on invalid inputs or backend responses:

- The retriever and reranker reject blank queries, invalid result limits, and non-finite or wrong-shape model outputs with `ValueError`. Mismatched `query_filters` lengths and invalid Qdrant payloads also raise `ValueError`.
- The catalog loader and filter builder reject invalid catalog entries and blank document IDs with `ValueError`. The router returns an empty list for a blank question.
- Passing a single `str` where a sequence/list is required (`build_document_filter`, `retrieve_batch`) raises `TypeError`.
- A Qdrant response count mismatch in `query_batch_points` raises `RuntimeError`.
