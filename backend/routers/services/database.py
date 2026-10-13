"""
LaborLens – Database Service
Async MySQL connection pool using aiomysql.
Creates all tables automatically on startup.
"""

import aiomysql
from pymysql.err import MySQLError
from typing import Optional, List, Dict, Any
from config import settings
from routers.services.kb_categories import DEFAULT_KB_CATEGORY, LEGACY_KB_CATEGORY_MAP
import logging
from routers.services.performance import measure

logger = logging.getLogger(__name__)
_pool: Optional[aiomysql.Pool] = None
_confidence_reason_column_type = "varchar(255)"


def confidence_reason_for_storage(reason):
    """Bound only the stored diagnostic, using the schema verified at startup."""
    from routers.services.diagnostic_persistence import confidence_reason_value
    return confidence_reason_value(reason, _confidence_reason_column_type)


async def ensure_confidence_reason_capacity(cur):
    """Widen the legacy diagnostic column during normal schema initialization.

    Existing TEXT-or-larger columns remain unchanged. If an existing database
    forbids schema changes, its actual capacity remains available to the write
    helper so a long diagnostic cannot break a chat insert.
    """
    global _confidence_reason_column_type
    await cur.execute("SHOW COLUMNS FROM chat_messages LIKE 'confidence_reason'")
    column = await cur.fetchone()
    if not column:
        await cur.execute("ALTER TABLE chat_messages ADD COLUMN confidence_reason TEXT NULL AFTER confidence_level")
        _confidence_reason_column_type = "text"
        return
    column_type = str(column.get("Type") or "") if isinstance(column, dict) else str(column[1] or "")
    _confidence_reason_column_type = column_type.strip().lower() or "varchar(255)"
    if _confidence_reason_column_type.startswith(("varchar(", "char(")) or _confidence_reason_column_type == "tinytext":
        try:
            await cur.execute("ALTER TABLE chat_messages MODIFY COLUMN confidence_reason TEXT NULL")
        except MySQLError as exc:
            logger.warning("Confidence reason schema could not be widened (%s); respecting existing column capacity", type(exc).__name__)
        else:
            _confidence_reason_column_type = "text"


async def init_db():
    global _pool
    _pool = await aiomysql.create_pool(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        db=settings.DB_NAME,
        autocommit=True,
        minsize=2,
        maxsize=10,
        charset="utf8mb4",
    )
    await create_tables()
    logger.info("✅ MySQL connected and tables verified.")


async def get_pool() -> aiomysql.Pool:
    global _pool
    if _pool is None:
        await init_db()
    return _pool


async def create_tables():
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100) NOT NULL UNIQUE,
                    display_name VARCHAR(100) NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    role ENUM('user','admin') DEFAULT 'user',
                    language_preference ENUM('auto','en','fil','hil') NOT NULL DEFAULT 'auto',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            await cur.execute("SHOW COLUMNS FROM users LIKE 'language_preference'")
            language_column = await cur.fetchone()
            if language_column:
                column_type = (
                    str(language_column.get("Type") or "")
                    if isinstance(language_column, dict)
                    else str(language_column[1] or "")
                )
                if "'auto'" not in column_type.lower():
                    await cur.execute(
                        "ALTER TABLE users MODIFY COLUMN language_preference "
                        "ENUM('auto','en','fil','hil') NOT NULL DEFAULT 'auto'"
                    )

            await cur.execute("SHOW COLUMNS FROM users LIKE 'display_name'")
            if not await cur.fetchone():
                await cur.execute(
                    "ALTER TABLE users ADD COLUMN display_name VARCHAR(100) NULL AFTER username"
                )

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id INT NOT NULL,
                    title VARCHAR(255) DEFAULT 'New Conversation',
                    category VARCHAR(100) NULL,
                    conversation_summary TEXT NULL,
                    conversation_state JSON NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE KEY uq_chat_session_owner (id, user_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            for column_name, column_sql in (
                ("category", "ADD COLUMN category VARCHAR(100) NULL AFTER title"),
                ("conversation_summary", "ADD COLUMN conversation_summary TEXT NULL AFTER title"),
                ("conversation_state", "ADD COLUMN conversation_state JSON NULL AFTER conversation_summary"),
            ):
                await cur.execute("SHOW COLUMNS FROM chat_sessions LIKE %s", (column_name,))
                if not await cur.fetchone():
                    await cur.execute(f"ALTER TABLE chat_sessions {column_sql}")
            await cur.execute("""
                SELECT COUNT(*)
                FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA=DATABASE()
                  AND TABLE_NAME='chat_sessions'
                  AND INDEX_NAME='uq_chat_session_owner'
            """)
            if (await cur.fetchone())[0] == 0:
                await cur.execute(
                    "ALTER TABLE chat_sessions ADD UNIQUE KEY uq_chat_session_owner (id, user_id)"
                )

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    user_id INT NOT NULL,
                    role ENUM('user','assistant','system') NOT NULL,
                    content TEXT NOT NULL,
                    language ENUM('en','fil','hil') DEFAULT 'en',
                    sources JSON,
                    intent VARCHAR(100),
                    confidence_level VARCHAR(20),
                    confidence_reason TEXT NULL,
                    confidence_score FLOAT NULL,
                    confidence_label VARCHAR(40) NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    KEY idx_chat_message_owner_history (user_id, session_id, created_at),
                    CONSTRAINT fk_chat_message_session_owner
                        FOREIGN KEY (session_id, user_id)
                        REFERENCES chat_sessions(id, user_id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            for column_name, column_sql in (
                ("admin_review_status", "ADD COLUMN admin_review_status VARCHAR(20) NOT NULL DEFAULT 'unreviewed'"),
                ("admin_query_status", "ADD COLUMN admin_query_status VARCHAR(20) NULL"),
                ("admin_review_note", "ADD COLUMN admin_review_note TEXT NULL"),
                ("admin_reviewed_by", "ADD COLUMN admin_reviewed_by INT NULL"),
                ("admin_reviewed_at", "ADD COLUMN admin_reviewed_at DATETIME NULL"),
                ("confidence_level", "ADD COLUMN confidence_level VARCHAR(20) NULL AFTER intent"),
                ("confidence_reason", "ADD COLUMN confidence_reason TEXT NULL AFTER confidence_level"),
                ("confidence_score", "ADD COLUMN confidence_score FLOAT NULL AFTER confidence_reason"),
                ("confidence_label", "ADD COLUMN confidence_label VARCHAR(40) NULL AFTER confidence_score"),
            ):
                await cur.execute("SHOW COLUMNS FROM chat_messages LIKE %s", (column_name,))
                if not await cur.fetchone():
                    await cur.execute(f"ALTER TABLE chat_messages {column_sql}")
            await ensure_confidence_reason_capacity(cur)
            await cur.execute("""
                SELECT COUNT(*)
                FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA=DATABASE()
                  AND TABLE_NAME='chat_messages'
                  AND INDEX_NAME='idx_chat_message_owner_history'
            """)
            if (await cur.fetchone())[0] == 0:
                await cur.execute(
                    "ALTER TABLE chat_messages "
                    "ADD KEY idx_chat_message_owner_history (user_id, session_id, created_at)"
                )

            await cur.execute("""
                SELECT COUNT(*)
                FROM information_schema.TABLE_CONSTRAINTS
                WHERE CONSTRAINT_SCHEMA=DATABASE()
                  AND TABLE_NAME='chat_messages'
                  AND CONSTRAINT_NAME='fk_chat_message_session_owner'
                  AND CONSTRAINT_TYPE='FOREIGN KEY'
            """)
            owner_constraint_exists = (await cur.fetchone())[0] > 0
            if not owner_constraint_exists:
                await cur.execute("""
                    SELECT COUNT(*)
                    FROM chat_messages AS message
                    LEFT JOIN chat_sessions AS session
                      ON session.id=message.session_id
                     AND session.user_id=message.user_id
                    WHERE session.id IS NULL
                """)
                mismatched_owner_rows = (await cur.fetchone())[0]
                if mismatched_owner_rows == 0:
                    await cur.execute("""
                        ALTER TABLE chat_messages
                        ADD CONSTRAINT fk_chat_message_session_owner
                        FOREIGN KEY (session_id, user_id)
                        REFERENCES chat_sessions(id, user_id)
                        ON DELETE CASCADE
                    """)
                else:
                    logger.error(
                        "Cannot add chat ownership constraint: %s message rows have a mismatched session owner",
                        mismatched_owner_rows,
                    )

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS uploaded_documents (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(120) NOT NULL,
                    file_size INT,
                    category VARCHAR(100) DEFAULT 'Core Labor Laws',
                    secondary_categories JSON NULL,
                    document_title VARCHAR(255),
                    document_type VARCHAR(120),
                    legal_role VARCHAR(120),
                    implements_json JSON NULL,
                    topics_json JSON NULL,
                    sub_intents_json JSON NULL,
                    source_name VARCHAR(255),
                    source_url VARCHAR(500),
                    law_number VARCHAR(120),
                    article_section VARCHAR(255),
                    issuing_agency VARCHAR(255),
                    date_issued VARCHAR(120),
                    effective_date VARCHAR(120),
                    date_downloaded VARCHAR(120),
                    document_year INT NULL,
                    superseding_document VARCHAR(255),
                    law_version VARCHAR(120),
                    version_status VARCHAR(20) DEFAULT 'active',
                    is_active BOOLEAN DEFAULT TRUE,
                    uploaded_by INT,
                    status ENUM('pending','processing','indexed','error') DEFAULT 'pending',
                    processing_stage VARCHAR(40) DEFAULT 'uploaded',
                    chunk_count INT DEFAULT 0,
                    error_message TEXT,
                    last_indexed_at DATETIME NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (uploaded_by) REFERENCES users(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            await cur.execute("SHOW COLUMNS FROM uploaded_documents LIKE 'category'")
            if not await cur.fetchone():
                await cur.execute("""
                    ALTER TABLE uploaded_documents
                    ADD COLUMN category VARCHAR(100) DEFAULT 'Core Labor Laws' AFTER file_size
                """)
            await cur.execute("SHOW COLUMNS FROM uploaded_documents LIKE 'updated_at'")
            if not await cur.fetchone():
                await cur.execute("ALTER TABLE uploaded_documents ADD COLUMN updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)")
            await cur.execute(
                f"""
                ALTER TABLE uploaded_documents
                MODIFY COLUMN category VARCHAR(100) DEFAULT '{DEFAULT_KB_CATEGORY}'
                """
            )
            await cur.execute("SHOW COLUMNS FROM uploaded_documents LIKE 'file_type'")
            file_type_column = await cur.fetchone()
            if file_type_column and "varchar(50)" in str(file_type_column).lower():
                await cur.execute("""
                    ALTER TABLE uploaded_documents
                    MODIFY COLUMN file_type VARCHAR(120) NOT NULL
                """)
            await cur.execute(
                "UPDATE uploaded_documents SET category=%s WHERE category IS NULL OR category=''",
                (DEFAULT_KB_CATEGORY,),
            )
            for legacy_category, current_category in LEGACY_KB_CATEGORY_MAP.items():
                await cur.execute(
                    "UPDATE uploaded_documents SET category=%s WHERE category=%s",
                    (current_category, legacy_category),
                )
            for column_name, column_sql in (
                ("secondary_categories", "ADD COLUMN secondary_categories JSON NULL AFTER category"),
                ("document_title", "ADD COLUMN document_title VARCHAR(255) NULL AFTER category"),
                ("document_type", "ADD COLUMN document_type VARCHAR(120) NULL AFTER document_title"),
                ("legal_role", "ADD COLUMN legal_role VARCHAR(120) NULL AFTER document_type"),
                ("implements_json", "ADD COLUMN implements_json JSON NULL AFTER legal_role"),
                ("topics_json", "ADD COLUMN topics_json JSON NULL AFTER implements_json"),
                ("sub_intents_json", "ADD COLUMN sub_intents_json JSON NULL AFTER topics_json"),
                ("source_name", "ADD COLUMN source_name VARCHAR(255) NULL AFTER document_title"),
                ("source_url", "ADD COLUMN source_url VARCHAR(500) NULL AFTER source_name"),
                ("law_number", "ADD COLUMN law_number VARCHAR(120) NULL AFTER source_url"),
                ("article_section", "ADD COLUMN article_section VARCHAR(255) NULL AFTER law_number"),
                ("issuing_agency", "ADD COLUMN issuing_agency VARCHAR(255) NULL AFTER article_section"),
                ("date_issued", "ADD COLUMN date_issued VARCHAR(120) NULL AFTER issuing_agency"),
                ("effective_date", "ADD COLUMN effective_date VARCHAR(120) NULL AFTER date_issued"),
                ("date_downloaded", "ADD COLUMN date_downloaded VARCHAR(120) NULL AFTER effective_date"),
                ("document_year", "ADD COLUMN document_year INT NULL AFTER date_downloaded"),
                ("superseding_document", "ADD COLUMN superseding_document VARCHAR(255) NULL AFTER document_year"),
                ("law_version", "ADD COLUMN law_version VARCHAR(120) NULL AFTER superseding_document"),
                ("version_status", "ADD COLUMN version_status VARCHAR(20) DEFAULT 'active' AFTER law_version"),
                ("is_active", "ADD COLUMN is_active BOOLEAN DEFAULT TRUE AFTER version_status"),
                ("processing_stage", "ADD COLUMN processing_stage VARCHAR(40) DEFAULT 'uploaded' AFTER status"),
                ("index_version", "ADD COLUMN index_version INT DEFAULT 1 AFTER processing_stage"),
                ("replacement_notes", "ADD COLUMN replacement_notes TEXT NULL AFTER index_version"),
                ("replaced_at", "ADD COLUMN replaced_at DATETIME NULL AFTER replacement_notes"),
                ("file_hash", "ADD COLUMN file_hash CHAR(64) NULL AFTER replaced_at"),
                ("mime_type", "ADD COLUMN mime_type VARCHAR(120) NULL AFTER file_hash"),
                ("last_indexed_at", "ADD COLUMN last_indexed_at DATETIME NULL AFTER error_message"),
            ):
                await cur.execute("SHOW COLUMNS FROM uploaded_documents LIKE %s", (column_name,))
                if not await cur.fetchone():
                    await cur.execute(f"ALTER TABLE uploaded_documents {column_sql}")
            await cur.execute(
                "UPDATE uploaded_documents SET is_active=TRUE WHERE is_active IS NULL"
            )
            await cur.execute(
                "UPDATE uploaded_documents SET version_status='active' WHERE version_status IS NULL OR version_status=''"
            )
            await cur.execute(
                "UPDATE uploaded_documents SET secondary_categories=JSON_ARRAY() WHERE secondary_categories IS NULL"
            )
            await cur.execute(
                """
                UPDATE uploaded_documents
                SET processing_stage=
                    CASE
                        WHEN status='indexed' THEN 'ready'
                        WHEN status='error' THEN 'error'
                        WHEN status='processing' THEN 'indexing'
                        ELSE 'uploaded'
                    END
                WHERE processing_stage IS NULL OR processing_stage=''
                """
            )

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    session_id VARCHAR(36),
                    query_text TEXT,
                    intent VARCHAR(100),
                    response_time_ms INT,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    token_hash CHAR(64) NOT NULL UNIQUE,
                    expires_at DATETIME NOT NULL,
                    used_at DATETIME NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS faqs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    category VARCHAR(100),
                    language ENUM('en','fil','hil') DEFAULT 'en',
                    view_count INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            await cur.execute("SHOW COLUMNS FROM faqs LIKE 'updated_at'")
            if not await cur.fetchone():
                await cur.execute(
                    "ALTER TABLE faqs ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at"
                )

            # Existing FAQs remain drafts until an administrator reviews and publishes them.
            for column_name, definition in {
                "publication_status": "VARCHAR(20) NOT NULL DEFAULT 'draft'",
                "source_document_ids": "TEXT NULL",
                "source_versions": "LONGTEXT NULL",
                "source_citation": "TEXT NULL",
            }.items():
                await cur.execute("SHOW COLUMNS FROM faqs LIKE %s", (column_name,))
                if not await cur.fetchone():
                    await cur.execute(f"ALTER TABLE faqs ADD COLUMN {column_name} {definition}")

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_feedback (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    message_id INT NOT NULL,
                    user_id INT NOT NULL,
                    rating ENUM('yes','no') NOT NULL,
                    comment TEXT,
                    review_status VARCHAR(20) DEFAULT 'pending',
                    admin_notes TEXT,
                    reviewed_by INT NULL,
                    reviewed_at DATETIME NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_feedback_message_user (message_id, user_id),
                    FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            for column_name, column_sql in (
                ("review_status", "ADD COLUMN review_status VARCHAR(20) DEFAULT 'pending' AFTER comment"),
                ("admin_notes", "ADD COLUMN admin_notes TEXT NULL AFTER review_status"),
                ("reviewed_by", "ADD COLUMN reviewed_by INT NULL AFTER admin_notes"),
                ("reviewed_at", "ADD COLUMN reviewed_at DATETIME NULL AFTER reviewed_by"),
            ):
                await cur.execute("SHOW COLUMNS FROM chat_feedback LIKE %s", (column_name,))
                if not await cur.fetchone():
                    await cur.execute(f"ALTER TABLE chat_feedback {column_sql}")
            await cur.execute(
                "UPDATE chat_feedback SET review_status='pending' WHERE review_status IS NULL OR review_status=''"
            )

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS hiligaynon_glossary (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    term_en VARCHAR(255) NOT NULL,
                    term_hil VARCHAR(255) NOT NULL,
                    notes TEXT,
                    sample_usage TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_audit_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    admin_user_id INT NOT NULL,
                    action VARCHAR(120) NOT NULL,
                    target_type VARCHAR(80),
                    target_id VARCHAR(120),
                    details JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (admin_user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS kb_replacement_jobs (
                    job_id VARCHAR(80) PRIMARY KEY,
                    document_id INT NOT NULL,
                    admin_user_id INT NOT NULL,
                    status VARCHAR(40) DEFAULT 'queued',
                    progress INT DEFAULT 0,
                    message VARCHAR(500),
                    replacement_notes TEXT NULL,
                    old_filename VARCHAR(255),
                    new_filename VARCHAR(255),
                    old_file_size INT,
                    new_file_size INT,
                    old_chunk_count INT DEFAULT 0,
                    new_chunk_count INT DEFAULT 0,
                    warning_message TEXT NULL,
                    error_message TEXT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME NULL,
                    INDEX idx_kb_replacement_document (document_id, status),
                    FOREIGN KEY (document_id) REFERENCES uploaded_documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (admin_user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)


            await cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_corrections (
                    message_id INT PRIMARY KEY,
                    stage VARCHAR(32) NOT NULL DEFAULT 'needs_kb_update',
                    document_id INT NULL,
                    note TEXT NOT NULL,
                    retest_note TEXT NOT NULL,
                    revision INT NOT NULL DEFAULT 1,
                    updated_by INT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE,
                    FOREIGN KEY (document_id) REFERENCES uploaded_documents(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_coverage_reviews (
                    topic VARCHAR(80) NOT NULL,
                    document_id INT NOT NULL,
                    note TEXT NOT NULL,
                    reviewed_by INT NOT NULL,
                    reviewed_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
                    PRIMARY KEY (topic, document_id),
                    FOREIGN KEY (document_id) REFERENCES uploaded_documents(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)


async def execute_affected(query: str, params=None) -> int:
    """Execute a conditional write and report its affected row count."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params or ())
            return cur.rowcount


class AdminRoleChangeConflict(Exception):
    """Raised when a role change could lock administrators out of the system."""


async def execute_admin_role_change(user_id: int, role: str, acting_admin_id: int):
    """Change a user role while serializing changes that could remove admin access."""
    pool = await get_pool()
    lock_name = "laborlens_admin_role_change"
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute("SELECT GET_LOCK(%s, 5) AS acquired", (lock_name,))
            lock = await cur.fetchone()
            if not lock or int(lock.get("acquired") or 0) != 1:
                raise AdminRoleChangeConflict("Administrator role changes are busy. Please retry.")
            try:
                await conn.begin()
                await cur.execute(
                    "SELECT id,role,is_active FROM users WHERE id=%s FOR UPDATE",
                    (user_id,),
                )
                user = await cur.fetchone()
                if not user:
                    await conn.rollback()
                    return None
                if user_id == acting_admin_id and role != "admin":
                    raise AdminRoleChangeConflict("You cannot remove your own administrator access.")
                if user["role"] == "admin" and role != "admin" and bool(user.get("is_active", 1)):
                    await cur.execute(
                        "SELECT id FROM users WHERE role='admin' AND is_active=1 FOR UPDATE"
                    )
                    active_admins = await cur.fetchall()
                    if len(active_admins) <= 1:
                        raise AdminRoleChangeConflict("At least one active administrator must remain.")
                await cur.execute("UPDATE users SET role=%s WHERE id=%s", (role, user_id))
                await conn.commit()
                return user
            except Exception:
                await conn.rollback()
                raise
            finally:
                await cur.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
                await cur.fetchone()


async def execute_query(query: str, params=None) -> List[Dict[str, Any]]:
    with measure("database_read_ms"):
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, params or ())
                return list(await cur.fetchall())


async def execute_one(query: str, params=None) -> Optional[Dict[str, Any]]:
    with measure("database_read_ms"):
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, params or ())
                return await cur.fetchone()


async def execute_insert(query: str, params=None) -> int:
    stage = "session_update_ms" if "update chat_sessions" in query.lower() else "database_write_ms"
    with measure(stage):
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params or ())
                return cur.lastrowid
