package ai.pixelmind.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import ai.pixelmind.ui.theme.Radii

/**
 * A text field that behaves like an inset glass well: darker than the card it sits in, with a
 * hairline inner border so it reads as recessed rather than raised.
 */
@Composable
fun PromptField(
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: String,
    modifier: Modifier = Modifier,
    isDark: Boolean = true,
    minHeight: Dp = 96.dp,
    textStyle: TextStyle? = null,
    imeAction: ImeAction = ImeAction.Default,
    singleLine: Boolean = false,
    leading: (@Composable () -> Unit)? = null
) {
    val shape = RoundedCornerShape(Radii.control)
    Column(modifier = modifier.fillMaxWidth()) {
        if (leading != null) {
            Box(modifier = Modifier.padding(bottom = 8.dp)) { leading() }
        }
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = minHeight)
                .background(
                    brush = Brush.verticalGradient(
                        listOf(
                            Color.Black.copy(alpha = if (isDark) 0.28f else 0.04f),
                            Color.Black.copy(alpha = if (isDark) 0.16f else 0.02f)
                        )
                    ),
                    shape = shape
                )
                .border(
                    width = 1.dp,
                    color = Color.White.copy(alpha = if (isDark) 0.10f else 0.9f),
                    shape = shape
                )
                .padding(horizontal = 16.dp, vertical = 14.dp)
        ) {
            BasicTextField(
                value = value,
                onValueChange = onValueChange,
                modifier = Modifier.fillMaxWidth(),
                textStyle = (textStyle ?: MaterialTheme.typography.bodyLarge).copy(
                    color = MaterialTheme.colorScheme.onSurface
                ),
                cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                singleLine = singleLine,
                keyboardOptions = KeyboardOptions(imeAction = imeAction),
                decorationBox = { inner ->
                    if (value.isEmpty()) {
                        Text(
                            placeholder,
                            style = textStyle ?: MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.75f)
                        )
                    }
                    inner()
                }
            )
        }
    }
}
