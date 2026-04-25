# Schema Rules

## Canonical object types
- entity
- concept
- synthesis
- question
- claim

## Status values
- draft
- synthesized
- reviewed
- trusted
- deprecated

## Link relations
- supports
- contradicts
- mentions
- expands
- summarizes
- derived_from

## Authoring rules
- Every non-raw object must reference at least one source or evidence chunk.
- Human edits in `knowledge/` are preserved; automated updates should merge, not overwrite blindly.
- Claims without evidence remain draft or enter review.
