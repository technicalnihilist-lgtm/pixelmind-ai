#include <jni.h>
#include <android/log.h>
#include <cmath>
#include <cstdint>

#define LOG_TAG "PixelMindNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {

/**
 * One latent-consistency update, per element:
 *
 *     eps      = cOut * model_output + cSkip * x
 *     x_prev   = (x - sqrt(1 - alpha_prev) * eps) / sqrt(alpha_prev)
 *
 * The scalars are computed once on the Kotlin side and passed in, which keeps this function a
 * pure elementwise kernel and guarantees it agrees with the reference implementation.
 */
inline void lcm_step_kernel(float* latent,
                            const float* model_output,
                            jsize length,
                            float c_skip,
                            float c_out,
                            float sqrt_one_minus_alpha_prev,
                            float sqrt_alpha_prev) {
    for (jsize i = 0; i < length; ++i) {
        const float x = latent[i];
        const float eps = c_out * model_output[i] + c_skip * x;
        latent[i] = (x - sqrt_one_minus_alpha_prev * eps) / sqrt_alpha_prev;
    }
}

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_ai_pixelmind_engine_NativeScheduler_lcmStepNative(
        JNIEnv* env,
        jobject /* thiz */,
        jfloatArray latent_array,
        jfloatArray model_output_array,
        jfloat c_skip,
        jfloat c_out,
        jfloat sqrt_one_minus_alpha_prev,
        jfloat sqrt_alpha_prev) {

    if (latent_array == nullptr || model_output_array == nullptr) return;

    const jsize latent_len = env->GetArrayLength(latent_array);
    const jsize output_len = env->GetArrayLength(model_output_array);
    if (latent_len != output_len) {
        LOGI("Length mismatch: %d vs %d", latent_len, output_len);
        return;
    }

    auto* latent = static_cast<float*>(env->GetPrimitiveArrayCritical(latent_array, nullptr));
    auto* output = static_cast<float*>(env->GetPrimitiveArrayCritical(model_output_array, nullptr));
    if (latent == nullptr || output == nullptr) {
        if (latent) env->ReleasePrimitiveArrayCritical(latent_array, latent, JNI_ABORT);
        if (output) env->ReleasePrimitiveArrayCritical(model_output_array, output, JNI_ABORT);
        return;
    }

    lcm_step_kernel(latent, output, latent_len,
                    c_skip, c_out, sqrt_one_minus_alpha_prev, sqrt_alpha_prev);

    // Mode 0: copy back and free the critical section.
    env->ReleasePrimitiveArrayCritical(latent_array, latent, 0);
    env->ReleasePrimitiveArrayCritical(model_output_array, output, JNI_ABORT);
}
