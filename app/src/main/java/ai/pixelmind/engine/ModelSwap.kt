package ai.pixelmind.engine

import java.io.File

/**
 * The three-directory dance that installs an imported model.
 *
 * `models` is the live tree, `models.staging` is where an import unpacks, and `models.previous`
 * holds the tree the import replaced so that a model which fails to load can be put back.
 *
 * The swap is two renames, and a process that dies between them leaves `models` absent while the
 * user's weights sit in `models.previous`, where nothing used to look. The next launch then found
 * no manifest, provisioned the bundled model into the empty directory and dropped the imported
 * weights without a word — a killed process quietly undoing a successful import. [recover] is the
 * repair for exactly that window, and it runs before anything else touches the directory.
 *
 * Kept apart from [ModelImporter] because it needs nothing from Android: it is a set of file
 * moves, and file moves can be tested.
 */
internal object ModelSwap {

    const val MODELS = "models"
    const val STAGING = "models.staging"
    const val PREVIOUS = "models.previous"

    fun root(filesDir: File): File = File(filesDir, MODELS)
    fun staging(filesDir: File): File = File(filesDir, STAGING)
    fun previous(filesDir: File): File = File(filesDir, PREVIOUS)

    /**
     * current -> previous, staging -> current.
     *
     * Returns null on success, otherwise the sentence to show. Every failure path leaves a usable
     * tree at `root`: either the new one or the old one, never neither.
     */
    fun install(root: File, staging: File, previous: File): String? {
        previous.deleteRecursively()
        if (root.exists() && !root.renameTo(previous)) {
            return "Could not move the current model aside."
        }
        if (!staging.renameTo(root)) {
            // Put the old tree back rather than leave the app with no model at all.
            previous.renameTo(root)
            return "Could not install the imported model."
        }
        return null
    }

    /** Puts the previous model back after a failed load. False when there was nothing to put back. */
    fun rollback(root: File, previous: File): Boolean {
        if (!previous.exists()) return false
        root.deleteRecursively()
        return previous.renameTo(root)
    }

    /**
     * Undoes an interrupted install: `root` is missing but a previous tree survived.
     *
     * True when a tree was moved back into place. Deliberately does nothing when `root` exists —
     * a present tree is either the new model or the old one, and neither should be second-guessed.
     */
    fun recover(root: File, previous: File): Boolean {
        if (root.exists()) return false
        if (!previous.exists()) return false
        return previous.renameTo(root)
    }
}
