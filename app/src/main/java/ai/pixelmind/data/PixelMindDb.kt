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
