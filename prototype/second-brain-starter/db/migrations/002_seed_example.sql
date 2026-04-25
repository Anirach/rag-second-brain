INSERT OR IGNORE INTO sources (
  source_id, title, source_type, origin, url, checksum, language, created_at, updated_at, metadata_json
) VALUES (
  'src_20260425_001',
  'Attention Is All You Need',
  'paper',
  'seed-example',
  'https://arxiv.org/abs/1706.03762',
  'sha256:example-paper-checksum',
  'en',
  '2026-04-25T14:00:00+07:00',
  '2026-04-25T14:00:00+07:00',
  '{"authors":["Vaswani et al."],"tags":["transformers","nlp"]}'
);

INSERT OR IGNORE INTO chunks (
  chunk_id, source_id, section, text, char_start, char_end, page_start, page_end, checksum
) VALUES (
  'chk_src_20260425_001_0001',
  'src_20260425_001',
  'Abstract',
  'The Transformer is a neural architecture built around attention mechanisms and avoids recurrence entirely.',
  0,
  99,
  1,
  1,
  'sha256:chunk-example'
);

INSERT OR IGNORE INTO objects (
  object_id, type, title, slug, status, confidence, path, created_at, updated_at
) VALUES
  ('ent_transformer', 'entity', 'Transformer', 'transformer', 'reviewed', 0.88, 'knowledge/entities/transformer.md', '2026-04-25T14:10:00+07:00', '2026-04-25T14:10:00+07:00'),
  ('cpt_self_attention', 'concept', 'Self-Attention', 'self-attention', 'synthesized', 0.81, 'knowledge/concepts/self-attention.md', '2026-04-25T14:12:00+07:00', '2026-04-25T14:12:00+07:00'),
  ('syn_transformer_architecture_overview', 'synthesis', 'Transformer Architecture Overview', 'transformer-architecture-overview', 'draft', NULL, 'knowledge/synthesis/transformer-architecture-overview.md', '2026-04-25T14:20:00+07:00', '2026-04-25T14:20:00+07:00'),
  ('qst_attention_open_questions', 'question', 'Open Questions on Attention', 'attention-open-questions', 'draft', NULL, 'knowledge/questions/attention-open-questions.md', '2026-04-25T14:21:00+07:00', '2026-04-25T14:21:00+07:00');

INSERT OR IGNORE INTO evidence_links (object_id, chunk_id, relation) VALUES
  ('ent_transformer', 'chk_src_20260425_001_0001', 'supports'),
  ('cpt_self_attention', 'chk_src_20260425_001_0001', 'supports');

INSERT OR IGNORE INTO review_items (
  review_id, target_type, target_id, reason, severity, status, created_at, resolved_at
) VALUES (
  'rev_20260425_001',
  'claim',
  'clm_src_20260425_001_01',
  'Promote claim to trusted only after checking source coverage across full paper.',
  'medium',
  'open',
  '2026-04-25T14:25:00+07:00',
  NULL
);
