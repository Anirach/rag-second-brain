PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  title TEXT,
  source_type TEXT,
  origin TEXT,
  url TEXT,
  checksum TEXT NOT NULL,
  language TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  section TEXT,
  text TEXT NOT NULL,
  char_start INTEGER,
  char_end INTEGER,
  page_start INTEGER,
  page_end INTEGER,
  checksum TEXT,
  FOREIGN KEY (source_id) REFERENCES sources(source_id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id);

CREATE TABLE IF NOT EXISTS objects (
  object_id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  title TEXT NOT NULL,
  slug TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL,
  confidence REAL,
  path TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(type);
CREATE INDEX IF NOT EXISTS idx_objects_status ON objects(status);

CREATE TABLE IF NOT EXISTS evidence_links (
  object_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  relation TEXT DEFAULT 'supports',
  PRIMARY KEY (object_id, chunk_id, relation),
  FOREIGN KEY (object_id) REFERENCES objects(object_id),
  FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_evidence_links_chunk_id ON evidence_links(chunk_id);

CREATE TABLE IF NOT EXISTS object_links (
  from_object_id TEXT NOT NULL,
  to_object_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  PRIMARY KEY (from_object_id, to_object_id, relation)
);

CREATE TABLE IF NOT EXISTS claims (
  claim_id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  claim_type TEXT,
  status TEXT NOT NULL,
  confidence REAL,
  source_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS review_items (
  review_id TEXT PRIMARY KEY,
  target_type TEXT NOT NULL,
  target_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  severity TEXT,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_items_status ON review_items(status);
