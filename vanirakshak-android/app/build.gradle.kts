plugins {
    alias(libs.plugins.android.application)
    // All sources under app/src/*/kotlin are Kotlin — without this plugin
    // the Kotlin sources are never compiled and the build fails.
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.vericall"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.vericall.temp"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    buildFeatures {
        viewBinding = true
    }

    testOptions {
        unitTests.isIncludeAndroidResources = true
        unitTests.isReturnDefaultValues = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}


dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.okhttp)
    testImplementation(libs.junit)
    testImplementation(libs.androidx.test.core)
    testImplementation(libs.robolectric)
}
