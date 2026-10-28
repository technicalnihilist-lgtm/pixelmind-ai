# ONNX Runtime reaches its native bridge through JNI reflection.
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# JNI entry points for the optional native scheduler.
-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}
-keep class ai.pixelmind.engine.nativebridge.** { *; }

# Kotlin coroutines
-dontwarn kotlinx.coroutines.**
