# ONVIF Discovery

- Discovery over LAN works.
- Connection to ONVIF supported Cameras work and frame can now be fetched from any camera

> Next: Error Handling and Async images

- Slint for the Local UI
- pyo3 to merge the python files in rust.

# Rust UI

- The UI should have a `Search for Cameras` button.
- It should have common username and password, for cameras and also, a way to check if the camera has ONVIF auth enabled.
- It should display details for each camera.
- then it should handle the web connection.
- A way to create regions, or the regions can be created directly in the remote server because regions does not matter for the same network.
- Renaming cameras or TAGs for the users to identify the cameras locally
- A locally hosted network website for those on the same network, store the mac addresses for the clients and the connection details on the local DB
- login on the local page.
- The EXE should handle the python version, packages and python errors.
- each camera should have an online or offline status, use the id from the ONVIF to identify the camera, local IP might change due to DHCP
- Add or remove camera feature is also, needed.
- Add remote updates so, the software can be updated when internet is available.
- A small local storage for event videos, and events, after the cloud update only the recent 50 events are stored locally
- Add a sync button that syncs with the cloud if not done automatically.
