package ai.pixelmind.data

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

/**
 * Plain SQLite. Deliberately not an ORM: the schema is four columns deep, and skipping
 * Room/KSP keeps both the APK and the build small.
 */
class PixelMindDb(context: Context) : SQLiteOpenHelper(context, NAME, null, VERSION) {

    init {
        // Write-ahead logging. The gallery reads the whole record list on the main thread at
        // start-up and after every delete while a generation writes its row on the IO dispatcher;
        // under the default rollback journal the writer blocks every reader for the duration of
        // the commit, and the reader blocks the writer. Under WAL they do not wait for each other.
        // Set on the helper, before the database is opened, which is the documented place for it.
        setWriteAheadLoggingEnabled(true)
    }

    override fun onConfigure(db: SQLiteDatabase) {
        // Runs before onCreate, so the page size below is the one a fresh file is created with.
        //
        // SQLite's Java API exposes no "allocate N bytes now", so the file cannot literally be
        // pre-sized; what this does is remove the two things that made it grow and shrink under
        // the app: a 4 KB page (the default is 1 KB on some builds, which quadruples the number of
        // page allocations for the same rows) and auto-vacuum, which returns freed pages to the
        // filesystem and makes the next insert allocate again.
        runCatching { db.execSQL("PRAGMA page_size = 4096") }
        runCatching { db.execSQL("PRAGMA auto_vacuum = NONE") }
        // Bound the -wal file. Left alone it grows until a checkpoint and can outgrow the database
        // itself; 4 MB is far more than a burst of gallery writes needs.
        runCatching { db.execSQL("PRAGMA journal_size_limit = 4194304") }
        // Safe under WAL: a committed transaction survives a process kill, and only an OS crash
        // can lose the last commits. It removes an fsync from every insert.
        runCatching { db.execSQL("PRAGMA synchronous = NORMAL") }
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
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
        )
        db.execSQL("CREATE INDEX idx_created ON $TABLE_GENERATIONS (created_at DESC)")
        db.execSQL("CREATE INDEX idx_favorite ON $TABLE_GENERATIONS (favorite)")
        db.execSQL(
            """
            CREATE TABLE $TABLE_PROMPTS (
                prompt TEXT PRIMARY KEY,
                last_used INTEGER NOT NULL,
                uses INTEGER NOT NULL DEFAULT 1
            )
            """.trimIndent()
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // Local, disposable data: a schema change rebuilds rather than migrating history.
        db.execSQL("DROP TABLE IF EXISTS $TABLE_GENERATIONS")
        db.execSQL("DROP TABLE IF EXISTS $TABLE_PROMPTS")
        onCreate(db)
    }

    fun insertGeneration(record: GenerationRecord): Long {
        val values = ContentValues().apply {
            put("prompt", record.prompt)
            put("negative_prompt", record.negativePrompt)
            put("style", record.style)
            put("aspect", record.aspect)
            put("width", record.width)
            put("height", record.height)
            put("steps", record.steps)
            put("seed", record.seed)
            put("file_path", record.filePath)
            put("created_at", record.createdAt)
            put("duration_ms", record.durationMs)
            put("favorite", if (record.favorite) 1 else 0)
        }
        return writableDatabase.insert(TABLE_GENERATIONS, null, values)
    }

    fun rememberPrompt(prompt: String) {
        val db = writableDatabase
        db.execSQL(
            "INSERT INTO $TABLE_PROMPTS (prompt, last_used, uses) VALUES (?, ?, 1) " +
                "ON CONFLICT(prompt) DO UPDATE SET last_used = excluded.last_used, uses = uses + 1",
            arrayOf(prompt.trim(), System.currentTimeMillis())
        )
    }

    fun recentPrompts(limit: Int): List<String> =
        readableDatabase.rawQuery(
            "SELECT prompt FROM $TABLE_PROMPTS ORDER BY last_used DESC LIMIT $limit",
            null
        ).use { cursor ->
            buildList { while (cursor.moveToNext()) add(cursor.getString(0)) }
        }

    companion object {
        private const val NAME = "pixelmind.db"
        private const val VERSION = 1
        const val TABLE_GENERATIONS = "generations"
        const val TABLE_PROMPTS = "prompts"
    }
}
