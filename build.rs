fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_c_ref)");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch == "x86_64" || arch == "aarch64" {
        let mut b = cc::Build::new();
        b.file("c/ksw2_extz2_sse.c")
            .file("c/ksw2_ref_wrapper.c")
            .include("c")
            .define("__SSE2__", None)
            .warnings(false);

        if arch == "x86_64" {
            b.flag_if_supported("-msse4.1").define("__SSE4_1__", None);
        } else {
            // Build the SSE2-only code path and map intrinsics through sse2neon.
            b.include("c/sse2neon").define("KSW_SSE2_ONLY", None);
        }

        b.compile("ksw2cref");
        println!("cargo:rustc-cfg=has_c_ref");
    }
}
