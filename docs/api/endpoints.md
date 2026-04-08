# API Endpoints

## Upload a Document

```http
POST /jobs
Content-Type: multipart/form-data
X-API-Key: your-key

file: <binary>
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "queued",
  "source_file": "report.pdf"
}
```

---

## Get Job Status

```http
GET /jobs/{job_id}
X-API-Key: your-key
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "ready_for_review",
  "total_pages": 12,
  "total_blocks": 89,
  "total_chunks": 134
}
```

---

## List Blocks

```http
GET /jobs/{job_id}/blocks?status=pending&page=1&limit=50
X-API-Key: your-key
```

**Query params:**

| Param | Description |
|---|---|
| `status` | Filter by review status |
| `type` | Filter by block type |
| `page` | Filter by page number |
| `skip` | Pagination offset |
| `limit` | Max results (default 100) |

---

## Update a Block

```http
PATCH /jobs/{job_id}/blocks/{block_id}
Content-Type: application/json
X-API-Key: your-key

{
  "status": "edited",
  "edited_text": "Corrected content.",
  "edited_type": "paragraph",
  "reviewer_note": "Fixed OCR error on page 3",
  "version": 1
}
```

!!! note "Optimistic Locking"
    Include `version` matching the current block version to prevent concurrent edit conflicts (returns `409` on mismatch).

---

## Finalize

```http
POST /jobs/{job_id}/finalize
Content-Type: application/json
X-API-Key: your-key

{
  "finalize_policy": "auto_approve_pending"
}
```

---

## Embed

```http
POST /jobs/{job_id}/embed
Content-Type: application/json
X-API-Key: your-key

{
  "provider": "openai",
  "model": "text-embedding-3-small",
  "vector_db": "chroma",
  "collection_name": "my_docs"
}
```

---

## Search

```http
POST /search
Content-Type: application/json
X-API-Key: your-key

{
  "job_id": "abc123",
  "query": "quarterly revenue breakdown",
  "top_k": 5,
  "filters": {
    "chunk_type": "table"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "chunk_id": "c_xyz",
      "text": "Q3 revenue: $4.2M...",
      "score": 0.92,
      "page_numbers": [7],
      "chunk_type": "table"
    }
  ],
  "total": 1
}
```

---

## Chat

```http
POST /chat
Content-Type: application/json
X-API-Key: your-key

{
  "session_id": "sess_abc",
  "job_id": "abc123",
  "question": "What were the Q3 highlights?",
  "require_approval": false,
  "config": {
    "llm_provider": "openai",
    "llm_model": "gpt-5.3",
    "top_k": 5
  }
}
```

**Response:**
```json
{
  "session_id": "sess_abc",
  "turn_id": "turn_001",
  "answer": "The Q3 highlights include...",
  "sources": [
    {
      "chunk_id": "c_xyz",
      "text": "Q3 revenue...",
      "page_numbers": [7]
    }
  ],
  "status": "complete"
}
```
