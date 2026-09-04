fn main() {
    // This tells the Rust compiler to read your main.slint file
    // and compile the UI design directly into native Rust code ahead of time.
    slint_build::compile("src/ui/main.slint").unwrap();
}
