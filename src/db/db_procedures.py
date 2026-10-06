from psycopg2.extras import execute_batch  # type: ignore
from psycopg2.extensions import connection  # type: ignore
from db.db_models import DocumentChunk
from settings import settings

def insert_chunks(conn: connection, chunks: list[DocumentChunk]):
    with conn.cursor() as cur:
        execute_batch(
            cur,
            """
            INSERT INTO document_chunks (
                id, doc_title, page_number, chunk_id, text, embedding
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk.id,
                    chunk.doc_title,
                    chunk.page_number,
                    chunk.chunk_id,
                    chunk.text,
                    chunk.embedding,
                )
                for chunk in chunks
            ],
        )
    conn.commit()


def query_similar_chunks(conn: connection, query_embedding, top_k=5):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                doc_title,
                page_number,
                chunk_id,
                text,
                1 - (embedding <=> (%s::vector)) AS cosine_similarity
            FROM document_chunks
            ORDER BY embedding <=> (%s::vector)
            LIMIT %s;
            """,
            (query_embedding, query_embedding, top_k),
        )

        return cur.fetchall()


def read_pgpass_entry(path: str, index: int = 0) -> tuple[str, int, str, str, str]:
    """Read one entry (host, port, dbname, user, password) from a libpq-style .pgpass file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            host, port, db, user, pw = line.split(":")
            entries.append((host, int(port), db, user, pw))
    return entries[index]


def get_db_credentials() -> tuple[str, int, str, str, str]:
    """Resolve database credentials: .pgpass if configured, otherwise PG_* settings."""
    if settings.pgpass_path:
        return read_pgpass_entry(settings.pgpass_path, settings.pgpass_index)
    return (
        settings.pg_host,
        settings.pg_port,
        settings.pg_database,
        settings.pg_user,
        settings.pg_password,
    )


def create_chunk_table(conn: connection):
    sql = """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id UUID PRIMARY KEY,
            doc_title TEXT NOT NULL,
            page_number INT NOT NULL,
            chunk_id INT NOT NULL,
            text TEXT NOT NULL,
            embedding vector(768)
        );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def drop_table(conn: connection):
    sql = """
    DROP TABLE IF EXISTS document_chunks;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
