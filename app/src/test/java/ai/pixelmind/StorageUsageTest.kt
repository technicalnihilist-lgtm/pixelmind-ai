package ai.pixelmind

import ai.pixelmind.util.StorageUsage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.Locale

/**
 * The numbers on the Settings storage card.
 *
 * The card shows three rows: the generated images, the model weights, and a total. The total used
 * to be `images + weights`, which is not the app's data — the database, its journal and the
 * verification sidecar were not in it. These tests pin the arithmetic and the formatting so the
 * card and the walk that feeds it cannot drift apart again.
 */
class StorageUsageTest {

    @get:Rule
    val temp = TemporaryFolder()

    @Test
    fun `formatting matches what the card has always shown`() {
        assertEquals("0 B", StorageUsage.formatBytes(0))
        assertEquals("1023 B", StorageUsage.formatBytes(1023))
        assertEquals("1 KB", StorageUsage.formatBytes(1024))
        assertEquals("1.0 MB", StorageUsage.formatBytes(1_048_576))
        assertEquals("22.2 MB", StorageUsage.formatBytes(23_270_517))
        assertEquals("1.00 GB", StorageUsage.formatBytes(1_073_741_824))
        assertEquals("1.50 GB", StorageUsage.formatBytes(1_610_612_736))
    }

    @Test
    fun `formatting does not change with the device locale`() {
        val original = Locale.getDefault()
        try {
            Locale.setDefault(Locale.GERMANY)
            assertEquals("22.2 MB", StorageUsage.formatBytes(23_270_517))
            Locale.setDefault(Locale.forLanguageTag("ar-EG"))
            assertEquals("1.50 GB", StorageUsage.formatBytes(1_610_612_736))
        } finally {
            Locale.setDefault(original)
        }
    }

    @Test
    fun `the total is every file in the app's private directory`() {
        val files = temp.newFolder("files")
        File(files, "generations").mkdirs()
        File(files, "models").mkdirs()
        write(File(files, "generations/pm_1_7.png"), 100)
        write(File(files, "generations/pm_2_7.png"), 200)
        write(File(files, "models/unet.onnx"), 1000)
        write(File(files, "pixelmind.db"), 4096)

        val appData = StorageUsage.treeBytes(files)
        val images = StorageUsage.treeBytes(File(files, "generations"))
        val weights = StorageUsage.treeBytes(File(files, "models"))

        assertEquals(5396L, appData)
        assertEquals(300L, images)
        assertEquals(1000L, weights)
        // What the card shows: images, weights, and a total that is at least their sum.
        assertTrue("total $appData is less than its parts", appData >= images + weights)
        assertEquals(4096L, StorageUsage.otherBytes(appData, images, weights))
        assertEquals("5 KB", StorageUsage.formatBytes(appData))
    }

    @Test
    fun `a directory that does not exist is zero bytes, not an error`() {
        assertEquals(0L, StorageUsage.treeBytes(File(temp.root, "absent")))
    }

    @Test
    fun `other bytes never goes negative when the parts are stale`() {
        // The images and the weights are measured separately from the walk, so a save between the
        // two reads can make the parts add up to more than the total. The card must not show a
        // negative remainder, and nothing else may divide by it.
        assertEquals(0L, StorageUsage.otherBytes(appDataBytes = 100, imageBytes = 80, modelBytes = 40))
    }

    private fun write(file: File, bytes: Int) {
        file.parentFile?.mkdirs()
        file.writeBytes(ByteArray(bytes) { it.toByte() })
    }
}
