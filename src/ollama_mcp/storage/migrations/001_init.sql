CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY,
  applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evals (
  id              TEXT PRIMARY KEY,
  schema_version  INTEGER NOT NULL,
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  prompt          TEXT NOT NULL,
  prompt_hash     TEXT NOT NULL,
  models          TEXT NOT NULL,
  task_type       TEXT,
  tags            TEXT,
  winner          TEXT,
  criteria        TEXT,
  scores          TEXT,
  judge_model     TEXT,
  notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_evals_task_type ON evals(task_type);
CREATE INDEX IF NOT EXISTS idx_evals_winner ON evals(winner);
CREATE INDEX IF NOT EXISTS idx_evals_prompt_hash ON evals(prompt_hash);

CREATE TABLE IF NOT EXISTS routing_history (
  id          TEXT PRIMARY KEY,
  changed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  task        TEXT NOT NULL,
  old_model   TEXT,
  new_model   TEXT NOT NULL,
  reason      TEXT
);
