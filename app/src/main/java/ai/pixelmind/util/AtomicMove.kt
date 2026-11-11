package ai.pixelmind.util

import java.io.File
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * Moves a fully written staging file onto its final name, so the final name is never visible in a
 * half-written state.
 *
 * Both writers in the app (a generated PNG, a provisioned model component) used to fall back to
 * `copyTo(target)` when `File.renameTo` returned false. That fallback was worse than the failure it
 * was papering over: it re-created the very hazard the staging file exists to prevent, because the
 * copy writes straight into the visible path, and a process kill halfway through leaves a truncated
 * file that every reader treats as real. A rename either happens or it does not; a copy does not.
 *
 * [Files.move] with [StandardCopyOption.ATOMIC_MOVE] is the same `rename(2)` the platform offers,
 * and it reports whether the filesystem can do it. Where it cannot, a plain rename is still a
 * single directory-entry change on every filesystem this app runs on. If neither works the move
 * fails and the caller's failure path deletes the staging file — nothing partial is left behind.
 */
internal fun moveAtomically(staging: File, target: File) {
    if (!staging.exists()) {
        throw IOException("${staging.name} disappeared before it could be moved into place")
    }
    try {
        Files.move(staging.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE)
        return
    } catch (_: AtomicMoveNotSupportedException) {
        // Filesystems that cannot rename atomically (some FUSE/SD mounts) still rename.
    }
    Files.move(staging.toPath(), target.toPath())
}
