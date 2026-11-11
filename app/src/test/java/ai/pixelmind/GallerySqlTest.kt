package ai.pixelmind

import ai.pixelmind.data.GallerySql
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.sql.Connection
import java.sql.DriverManager

/**
 * The gallery's SQL, executed against a real SQLite engine.
 *
 * Android's `SQLiteDatabase` is a stub on the unit-test classpath, so anything the app proves about
 * its database through a normal unit test is proved about a stub. These tests open an actual SQLite
 * file and run the statements in [GallerySql] — the same text the app runs — so the query plan, the
 * rollback behaviour and the upsert semantics under test are the ones the device will see.
 */
class GallerySqlTest {

    @get:Rule
    val temp = TemporaryFolder()

    private lateinit var conn: Connection

    @Before
    fun openDatabase() {
        Class.forName("org.sqlite.JDBC")
        conn = DriverManager.getConnection("jdbc:sqlite:" + File(temp.root, "gallery.db").absolutePath)
        conn.createStatement().use { statement ->
            statement.execute(GallerySql.CREATE_GENERATIONS)
            statement.execute(GallerySql.CREATE_INDEX_CREATED)
            statement.execute(GallerySql.CREATE_PROMPTS)
        }
    }

    @After
    fun closeDatabase() {
        conn.close()
    }

    // ---- item 42: index usage ---------------------------------------------------------------

    @Test
    fun `the gallery read walks the created_at index`() {
        val plan = queryPlan(GallerySql.SELECT_GENERATIONS)
        assertTrue("expected idx_created in the plan, got $plan", plan.any { it.contains("idx_created") })
    }

    @Test
    fun `the gallery read never sorts in a temporary b-tree`() {
        val plan = queryPlan(GallerySql.SELECT_GENERATIONS)
        assertFalse("the read sorts instead of using the index: $plan", plan.any { it.contains("TEMP B-TREE", true) })
    }

    @Test
    fun `the only index on the gallery table is the one the reads use`() {
        val indexes = stringColumn(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
            GallerySql.TABLE_GENERATIONS
        )
        assertEquals(listOf("idx_created"), indexes)
    }

    @Test
    fun `dropping the unused favorite index leaves the rows alone`() {
        insertGeneration("a red bicycle", createdAt = 1L)
        conn.createStatement().use { it.execute(GallerySql.DROP_INDEX_FAVORITE) }
        assertEquals(1, countOf(GallerySql.TABLE_GENERATIONS))
        assertEquals(listOf("idx_created"), stringColumn(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
            GallerySql.TABLE_GENERATIONS
        ))
    }

    // ---- item 49: prompt history is idempotent ----------------------------------------------

    @Test
    fun `remembering the same prompt fifty times leaves one row and counts the uses`() {
        repeat(50) { remember("a red bicycle", now = 1_700_000_000_000L) }
        assertEquals(1, countOf(GallerySql.TABLE_PROMPTS))
        assertEquals(50, usesOf("a red bicycle"))
    }

    @Test
    fun `a prompt written twice in the same millisecond keeps a stable newest-first order`() {
        val stamp = 1_700_000_000_000L
        (1..50).forEach { remember("prompt $it", now = stamp) }

        val first = recentPrompts(50)
        assertEquals((50 downTo 1).map { "prompt $it" }, first)
        // Re-reading must not reshuffle: the old statement ordered by a value 50 rows shared.
        assertEquals(first, recentPrompts(50))
    }

    @Test
    fun `last_used is strictly increasing even when the clock goes backwards`() {
        remember("later", now = 2_000L)
        remember("earlier", now = 1_000L)
        assertEquals(listOf("earlier", "later"), recentPrompts(10))
        assertEquals(2_001L, lastUsedOf("earlier"))
    }

    @Test
    fun `history respects its limit`() {
        (1..40).forEach { remember("prompt $it", now = 1_700_000_000_000L) }
        assertEquals(30, recentPrompts(30).size)
    }

    // ---- item 41: one transaction for the gallery row and the prompt ------------------------

    @Test
    fun `a rolled back generation write leaves neither a gallery row nor a prompt`() {
        conn.autoCommit = false
        insertGeneration("a red bicycle", createdAt = 1L)
        remember("a red bicycle", now = 1L)
        conn.rollback()
        conn.autoCommit = true

        assertEquals(0, countOf(GallerySql.TABLE_GENERATIONS))
        assertEquals(0, countOf(GallerySql.TABLE_PROMPTS))
    }

    @Test
    fun `a committed generation write stores both rows`() {
        conn.autoCommit = false
        insertGeneration("a red bicycle", createdAt = 1L)
        remember("a red bicycle", now = 1L)
        conn.commit()
        conn.autoCommit = true

        assertEquals(1, countOf(GallerySql.TABLE_GENERATIONS))
        assertEquals(1, countOf(GallerySql.TABLE_PROMPTS))
        assertEquals(listOf("a red bicycle"), recentPrompts(10))
    }

    // ---- helpers ----------------------------------------------------------------------------

    private fun queryPlan(sql: String, vararg args: Any?): List<String> =
        conn.prepareStatement("EXPLAIN QUERY PLAN $sql").use { statement ->
            args.forEachIndexed { index, value -> statement.setObject(index + 1, value) }
            statement.executeQuery().use { rows ->
                buildList { while (rows.next()) add(rows.getString("detail")) }
            }
        }

    private fun insertGeneration(prompt: String, createdAt: Long) {
        conn.prepareStatement(GallerySql.INSERT_GENERATION).use { statement ->
            val values = arrayOf<Any?>(
                prompt, "", "PHOTOREAL", "SQUARE", 512, 512, 8, 42L,
                "/data/user/0/ai.pixelmind/files/generations/pm_1_42.png",
                createdAt, 1234L, 0
            )
            values.forEachIndexed { index, value -> statement.setObject(index + 1, value) }
            statement.execute()
        }
    }

    private fun remember(prompt: String, now: Long) {
        conn.prepareStatement(GallerySql.REMEMBER_PROMPT).use { statement ->
            statement.setObject(1, prompt)
            statement.setObject(2, now)
            statement.execute()
        }
    }

    private fun recentPrompts(limit: Int): List<String> =
        conn.prepareStatement(GallerySql.RECENT_PROMPTS).use { statement ->
            statement.setObject(1, limit)
            statement.executeQuery().use { rows ->
                buildList { while (rows.next()) add(rows.getString(1)) }
            }
        }

    private fun countOf(table: String): Int =
        conn.createStatement().use { statement ->
            statement.executeQuery("SELECT COUNT(*) FROM $table").use { rows ->
                if (rows.next()) rows.getInt(1) else -1
            }
        }

    private fun usesOf(prompt: String): Int =
        conn.prepareStatement("SELECT uses FROM ${GallerySql.TABLE_PROMPTS} WHERE prompt = ?").use { statement ->
            statement.setObject(1, prompt)
            statement.executeQuery().use { rows -> if (rows.next()) rows.getInt(1) else -1 }
        }

    private fun lastUsedOf(prompt: String): Long =
        conn.prepareStatement("SELECT last_used FROM ${GallerySql.TABLE_PROMPTS} WHERE prompt = ?").use { statement ->
            statement.setObject(1, prompt)
            statement.executeQuery().use { rows -> if (rows.next()) rows.getLong(1) else -1L }
        }

    private fun stringColumn(sql: String, vararg args: Any?): List<String> =
        conn.prepareStatement(sql).use { statement ->
            args.forEachIndexed { index, value -> statement.setObject(index + 1, value) }
            statement.executeQuery().use { rows ->
                buildList { while (rows.next()) add(rows.getString(1)) }
            }
        }
}
