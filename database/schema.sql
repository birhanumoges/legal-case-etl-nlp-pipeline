-- database/schema.sql — PostgreSQL DDL for legal_nlp

CREATE TABLE IF NOT EXISTS legal_cases (
    id                   SERIAL PRIMARY KEY,
    case_id              VARCHAR(64)   NOT NULL UNIQUE,
    case_name            VARCHAR(512),
    source_folder        VARCHAR(128),
    year                 VARCHAR(8),
    court                VARCHAR(256),
    case_text            TEXT,
    case_text_length     INTEGER,

    verdict              VARCHAR(128),
    case_type            VARCHAR(64),
    sub_type             VARCHAR(64),

    verdict_mapped       VARCHAR(32),
    case_type_mapped     VARCHAR(32),
    sub_type_mapped      VARCHAR(64),

    num_citations        INTEGER DEFAULT 0,
    legal_citations      TEXT,
    statutes             TEXT,

    text_length          INTEGER,
    word_count           INTEGER,
    year_numeric         FLOAT,

    created_at           TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_year_type  ON legal_cases (year, case_type_mapped);
CREATE INDEX IF NOT EXISTS ix_verdict    ON legal_cases (verdict_mapped);
CREATE INDEX IF NOT EXISTS ix_court      ON legal_cases (court);
