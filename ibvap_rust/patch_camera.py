import re

with open("src/main.rs", "r") as f:
    content = f.read()

impl_block = """
impl DiscoveredCamera {
    pub fn get_active_rtsp(&self) -> String {
        let custom_user = self.rtsp_user.clone().unwrap_or_default();
        let custom_pass = self.rtsp_pass.clone().unwrap_or_default();
        
        if !custom_user.is_empty() && !custom_pass.is_empty() && self.rtsp.starts_with("rtsp://") {
            let without_scheme = &self.rtsp[7..];
            let host_path = if let Some(idx) = without_scheme.find('@') {
                &without_scheme[idx + 1..]
            } else {
                without_scheme
            };
            format!("rtsp://{}:{}@{}", custom_user, custom_pass, host_path)
        } else {
            self.rtsp.clone()
        }
    }
}
"""

content = content.replace("pub struct DiscoveredCamera {\n", "pub struct DiscoveredCamera {\n")
# find where DiscoveredCamera struct ends, and insert the impl
struct_end = """    #[serde(default)]
    pub rtsp_pass: Option<String>,
}"""

content = content.replace(struct_end, struct_end + "\n" + impl_block)

# Now replace camera.rtsp.clone() with camera.get_active_rtsp() in all three places!
content = content.replace("camera.rtsp.clone(),\n                    frame_tx", "camera.get_active_rtsp(),\n                    frame_tx")
content = content.replace("cam.rtsp.clone(),\n                            frame_tx", "cam.get_active_rtsp(),\n                            frame_tx")
content = content.replace("cam.rtsp.clone(),\n                        frame_tx", "cam.get_active_rtsp(),\n                        frame_tx")

with open("src/main.rs", "w") as f:
    f.write(content)
