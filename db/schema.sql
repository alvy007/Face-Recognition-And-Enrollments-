-- Run once in Supabase SQL Editor, before enabling persistent enrollment.
-- Only server-side code using a private PostgreSQL connection can access rows.
-- No photos are stored. Face embeddings are Fernet-encrypted in Python.
CREATE TABLE IF NOT EXISTS public.visionid_faces (
  label TEXT PRIMARY KEY CHECK (char_length(label) BETWEEN 1 AND 40),
  embedding_ciphertext BYTEA NOT NULL,
  consent_confirmed BOOLEAN NOT NULL DEFAULT FALSE CHECK (consent_confirmed),
  consented_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 days')
);
CREATE INDEX IF NOT EXISTS visionid_faces_expiry_idx ON public.visionid_faces (expires_at);
ALTER TABLE public.visionid_faces ENABLE ROW LEVEL SECURITY;
-- Prevent access via public Supabase REST roles: no RLS policies are created.
REVOKE ALL ON TABLE public.visionid_faces FROM PUBLIC, anon, authenticated;
