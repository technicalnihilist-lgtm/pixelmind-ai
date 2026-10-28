package ai.pixelmind

import android.content.pm.PackageManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The privacy promise, asserted rather than documented: the installed package must hold no
 * permissions at all, and must not be able to reach the network.
 */
@RunWith(AndroidJUnit4::class)
class OfflineIntegrityTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun app_declares_no_permissions() {
        val requested = context.packageManager
            .getPackageInfo(context.packageName, PackageManager.GET_PERMISSIONS)
            .requestedPermissions
        // AndroidX merges one app-scoped signature permission of its own
        // (<pkg>.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION). It is not a platform permission
        // and grants no capability, so it is excluded here — everything else must be absent.
        val platformPermissions = requested.orEmpty()
            .filterNot { it.startsWith(context.packageName) }
        assertTrue(
            "PixelMind must request no platform permissions, found: ${platformPermissions.joinToString()}",
            platformPermissions.isEmpty()
        )
    }

    @Test
    fun app_cannot_reach_the_network() {
        val granted = context.packageManager.checkPermission(
            "android.permission.INTERNET", context.packageName
        )
        assertFalse(
            "INTERNET must never be granted to PixelMind",
            granted == PackageManager.PERMISSION_GRANTED
        )
    }
}
