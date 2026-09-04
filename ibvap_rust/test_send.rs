use slint::SharedPixelBuffer;
use slint::Rgba8Pixel;

fn assert_send<T: Send>() {}

fn main() {
    assert_send::<SharedPixelBuffer<Rgba8Pixel>>();
    println!("SharedPixelBuffer is Send!");
}
