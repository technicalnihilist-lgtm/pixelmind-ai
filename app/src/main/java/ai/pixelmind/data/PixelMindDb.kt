package ai.pixelmind.data

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

/**
 * Plain SQLite. Deliberately not an ORM: the schema is four columns deep, and skipping
 * Room/KSP keeps both the APK and the build small.
 *
 * The statements themselves live in [GallerySql] so the JVM tests can run the exact same SQL
 * against a real SQLite engine.
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
        db.execSQL(GallerySql.CREATE_GENERATIONS)
        db.execSQL(GallerySql.CREATE_INDEX_CREATED)
        db.execSQL(GallerySql.CREATE_PROMPTS)
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // v1 -> v2 drops the unused `favorite` index and nothing else; the rows are kept.
        // Earlier versions rebuilt the tables on any schema change, which was acceptable while
        // the only change was to the shape of the data — this one is an index, and throwing the
        // user's gallery away to remove it would be a worse bug than the one being fixed.
        if (oldVersion < 2) db.execSQL(GallerySql.DROP_INDEX_FAVORITE)
    }

    /**
     * Runs [block] inside one transaction, so a group of statements commits or rolls back as a
     * unit and costs one journal write instead of one per statement.
     */
    fun <T> transaction(block: (SQLiteDatabase) -> T): T {
        val db = writableDatabase
        db.beginTransaction()
        try {
            val result = block(db)
            db.setTransactionSuccessful()
            return result
        } finally {
            db.endTransaction()
        }
    }

    /**
     * Writes the gallery row and the prompt-history row in a single transaction.
     *
     * They used to be two independent statements, each its own commit: two journal writes and two
     * fsyncs per generation under WAL, and a window in which a kill could leave a gallery entry
     * whose prompt never reached the history. Both rows are now one unit.
     */
    fun recordGeneration(record: GenerationRecord): Long = transaction { db ->
        db.execSQL(
            GallerySql.INSERT_GENERATION,
            arrayOf(
                record.prompt,
                record.negativePrompt,
                record.style,
                record.aspect,
                record.width,
                record.height,
                record.steps,
                record.seed,
                record.filePath,
                record.createdAt,
                record.durationMs,
                if (record.favorite) 1 else 0
            )
        )
        rememberPrompt(db, record.prompt)
        lastInsertedId(db)
    }

    /**
     * Records that [prompt] was used, once per generation.
     *
     * A blank prompt is not a history entry; the upsert would otherwise put an empty row in the
     * suggestion list.
     */
    fun rememberPrompt(prompt: String) {
        rememberPrompt(writableDatabase, prompt)
    }

    private fun rememberPrompt(db: SQLiteDatabase, prompt: String) {
        val key = prompt.trim()
        if (key.isEmpty()) return
        db.execSQL(GallerySql.REMEMBER_PROMPT, arrayOf(key, System.currentTimeMillis()))
    }

    fun setFavorite(id: Long, favorite: Boolean) {
        writableDatabase.execSQL(GallerySql.TOGGLE_FAVORITE, arrayOf(if (favorite) 1 else 0, id))
    }

    fun deleteGeneration(id: Long) {
        writableDatabase.execSQL(GallerySql.DELETE_GENERATION, arrayOf(id))
    }

    fun recentPrompts(limit: Int): List<String> =
        readableDatabase.rawQuery(GallerySql.RECENT_PROMPTS, arrayOf(limit.toString())).use { cursor ->
            buildList { while (cursor.moveToNext()) add(cursor.getString(0)) }
        }

    fun queryAll(): List<GenerationRecord> =
        readableDatabase.rawQuery(GallerySql.SELECT_GENERATIONS, null).use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    add(
                        GenerationRecord(
                            id = cursor.getLong(0),
                            prompt = cursor.getString(1),
                            negativePrompt = cursor.getString(2),
                            style = cursor.getString(3),
                            aspect = cursor.getString(4),
                            width = cursor.getInt(5),
                            height = cursor.getInt(6),
                            steps = cursor.getInt(7),
                            seed = cursor.getLong(8),
                            filePath = cursor.getString(9),
                            createdAt = cursor.getLong(10),
                            durationMs = cursor.getLong(11),
                            favorite = cursor.getInt(12) == 1
                        )
                    )
                }
            }
        }

    private fun lastInsertedId(db: SQLiteDatabase): Long =
        db.rawQuery("SELECT last_insert_rowid()", null).use { cursor ->
            if (cursor.moveToFirst()) cursor.getLong(0) else -1L
        }

    companion object {
        private const val NAME = "pixelmind.db"

        /** v2: `idx_favorite` dropped (unused). See [GallerySql.CREATE_INDEX_CREATED]. */
        private const val VERSION = 2
        const val TABLE_GENERATIONS = GallerySql.TABLE_GENERATIONS
        const val TABLE_PROMPTS = GallerySql.TABLE_PROMPTS
    }
}
