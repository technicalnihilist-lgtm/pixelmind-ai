plugins {
    id("com.android.asset-pack")
}

assetPack {
    packName = "modelpack"
    dynamicDelivery {
        // Installed together with the APK: no user-visible download, ever.
        deliveryType = "install-time"
    }
}
