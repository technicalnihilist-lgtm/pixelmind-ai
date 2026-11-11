package ai.pixelmind.data

/**
 * Every SQL statement the gallery and the prompt history run, as plain strings.
 *
 * They live outside [PixelMindDb] because the JVM unit tests cannot open an Android
 * `SQLiteDatabase` — on the unit-test classpath it is a stub that returns defaults. Keeping the
 * exact statement text here lets the tests execute *the same SQL* against a real SQLite engine
 * (see `GallerySqlTest`), which is the only way to prove that a query plan uses an index, that a
 * transaction rolls back, or that a prompt upsert is idempotent. A statement copy-pasted into a
 * test would prove nothing about the app.
 */
internal object GallerySql {

    const val TABLE_GENERATIONS = "generations"
    const val TABLE_PROMPTS = "prompts"

    val CREATE_GENERATIONS = """
        CREATE TABLE $TABLE_GENERATIONS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            negative_prompt TEXT NOT NULL DEFAULT '',
            style TEXT NOT NULL,
            aspect TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            steps INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            favorite INTEGER NOT NULL DEFAULT 0
        )
    """.trimIndent()

    /**
     * The only index the gallery needs: every read is `ORDER BY created_at DESC`, and this index
     * lets SQLite walk it backwards instead of sorting the whole table in a temporary B-tree.
     *
     * There is deliberately no index on `favorite`. It was created on day one and never used: no
     * statement filters on it — the favourites filter is applied to the already-loaded list in
     * `GalleryScreen` — and a two-value column cannot narrow a scan anyway. It cost one index write
     * on every insert and every favourite toggle, and it is dropped in schema v2.
     */
    const val CREATE_INDEX_CREATED =
        "CREATE INDEX idx_created ON $TABLE_GENERATIONS (created_at DESC)"

    const val DROP_INDEX_FAVORITE = "DROP INDEX IF EXISTS idx_favorite"

    val CREATE_PROMPTS = """
        CREATE TABLE $TABLE_PROMPTS (
            prompt TEXT PRIMARY KEY,
            last_used INTEGER NOT NULL,
            uses INTEGER NOT NULL DEFAULT 1
        )
    """.trimIndent()

    val INSERT_GENERATION =
        "INSERT INTO $TABLE_GENERATIONS (prompt, negative_prompt, style, aspect, width, height, " +
            "steps, seed, file_path, created_at, duration_ms, favorite) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"

    /**
     * Prompt history upsert, safe to run any number of times for the same prompt.
     *
     * `last_used` is `max(now, highest existing + 1)` rather than plain `now`: generation is
     * serialised but two records can still land in the same millisecond, and history was ordered
     * by `last_used` alone — so two prompts written in the same millisecond came back in whatever
     * order SQLite happened to return them, and the list visibly reshuffled between reads. The
     * bump makes the value strictly increasing per insert, which also keeps history sane if the
     * device clock is moved backwards.
     *
     * The conflict clause updates the row instead of inserting a second one, so a prompt generated
     * repeatedly stays one row with a growing `uses` count.
     */
    const val REMEMBER_PROMPT =
        "INSERT INTO $TABLE_PROMPTS (prompt, last_used, uses) " +
            "VALUES (?, MAX(?, IFNULL((SELECT MAX(last_used) FROM $TABLE_PROMPTS), 0) + 1), 1) " +
            "ON CONFLICT(prompt) DO UPDATE SET last_used = excluded.last_used, " +
            "uses = $TABLE_PROMPTS.uses + 1"

    /** `rowid` breaks the remaining ties so the order never depends on storage order. */
    const val RECENT_PROMPTS =
        "SELECT prompt FROM $TABLE_PROMPTS ORDER BY last_used DESC, rowid DESC LIMIT ?"

    const val SELECT_GENERATIONS =
        "SELECT id, prompt, negative_prompt, style, aspect, width, height, steps, seed, " +
            "file_path, created_at, duration_ms, favorite " +
            "FROM $TABLE_GENERATIONS ORDER BY created_at DESC"

    const val TOGGLE_FAVORITE = "UPDATE $TABLE_GENERATIONS SET favorite = ? WHERE id = ?"

    const val DELETE_GENERATION = "DELETE FROM $TABLE_GENERATIONS WHERE id = ?"
}
