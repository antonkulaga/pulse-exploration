import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
}

// Samsung Health Sensor SDK (tier "all") is optional and licensed separately: download it from
// developer.samsung.com/health/sensor and put the .aar in probe/libs/ (gitignored). With it, src/samsungSdk
// is compiled and the app uses the SDK whenever the watch allows it. Without it (or with
// -PsamsungSdk=false) src/noSamsungSdk is compiled instead and the app records tier "25hz" only.
val samsungSdk = fileTree("libs") { include("samsung-health-sensor-api-*.aar") }
val withSamsungSdk = !samsungSdk.isEmpty && providers.gradleProperty("samsungSdk").orNull != "false"
if (!withSamsungSdk) logger.warn("probe: building without the Samsung Health Sensor SDK, tier 25hz only")

android {
    namespace = "io.github.antonkulaga.pulseprobe"
    compileSdk = 36

    defaultConfig {
        applicationId = "io.github.antonkulaga.pulseprobe"
        minSdk = 33
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets.getByName("main").kotlin.srcDir(if (withSamsungSdk) "src/samsungSdk/java" else "src/noSamsungSdk/java")
}

kotlin {
    jvmToolchain(21)
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    if (withSamsungSdk) implementation(samsungSdk)
}
