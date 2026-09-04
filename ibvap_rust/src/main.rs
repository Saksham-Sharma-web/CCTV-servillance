use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde::Deserialize;
use slint::Model;
use std::rc::Rc;
use std::thread;
mod database;
use database::*;
use uuid;
slint::include_modules!();


// ============================================================
// Python -> Rust data
// ============================================================

#[derive(Debug, Deserialize, Clone)]
struct DiscoveredCamera {
    #[serde(default)]
    id: String,

    #[serde(default)]
    name: String,

    #[serde(default)]
    ip: String,

    #[serde(default)]
    rtsp: String,
}


// ============================================================
// Helpers
// ============================================================

fn generate_id(index:usize) -> String {
    format!("manual-{}", uuid::Uuid::new_v4())
}




fn discover_cameras_with_python() -> Result<Vec<DiscoveredCamera>, String> {

    Python::with_gil(|py| {

        // ----------------------------------------------------
        // Make the project directory available to Python
        // ----------------------------------------------------

        let sys = py
            .import("sys")
            .map_err(|e| e.to_string())?;

        let cwd = std::env::current_dir()
            .map_err(|e| e.to_string())?;

        sys.getattr("path")
            .map_err(|e| e.to_string())?
            .call_method1(
                "insert",
                (0, cwd.to_string_lossy().to_string()),
            )
            .map_err(|e| e.to_string())?;


        // ----------------------------------------------------
        // Import stream.py
        // ----------------------------------------------------

        let stream = PyModule::import(py, "stream")
            .map_err(|e| {
                format!("Failed to import stream.py:\n{}", e)
            })?;


        // ----------------------------------------------------
        // Run Python async main()
        // ----------------------------------------------------

        let asyncio = py
            .import("asyncio")
            .map_err(|e| e.to_string())?;

        let main_function = stream
            .getattr("main")
            .map_err(|e| e.to_string())?;

        let result = asyncio
            .call_method1(
                "run",
                (main_function.call0()
                    .map_err(|e| e.to_string())?,),
            )
            .map_err(|e| {
                format!("Python stream.main() failed:\n{}", e)
            })?;


        // ----------------------------------------------------
        // Python returns:
        //
        //     List[dict]
        //
        // ----------------------------------------------------

        let json = py
            .import("json")
            .map_err(|e| e.to_string())?;

        let json_string: String = json
            .getattr("dumps")
            .map_err(|e| e.to_string())?
            .call1((result,))
            .map_err(|e| e.to_string())?
            .extract()
            .map_err(|e| e.to_string())?;


        // ----------------------------------------------------
        // JSON -> Rust
        // ----------------------------------------------------

        serde_json::from_str::<Vec<DiscoveredCamera>>(
            &json_string
        )
        .map_err(|e| {
            format!(
                "Python returned invalid camera JSON:\n{}\n\nError: {}",
                json_string,
                e
            )
        })
    })
}


// ============================================================
// Convert Python camera -> Slint camera
// ============================================================

fn to_slint_camera(
    camera: DiscoveredCamera,
    index: usize,
) -> Camera {

    let id = if camera.id.is_empty() {
        generate_id(index)
    } else {
        camera.id
    };

    let name = if camera.name.is_empty() {
        format!("Camera {}", index + 1)
    } else {
        camera.name
    };

    Camera {
        id: id.into(),
        name: name.clone().into(),
        tag: name.into(),
        ip: camera.ip.into(),

        is_online: true,
        has_onvif: true,

        last_seen: "Just now".into(),
    }
}


// ============================================================
// Main
// ============================================================

fn main() -> Result<(), slint::PlatformError> {

    // --------------------------------------------------------
    // Initialize Python
    // --------------------------------------------------------

    pyo3::prepare_freethreaded_python();


    // --------------------------------------------------------
    // Create Slint application
    // --------------------------------------------------------

    let ui = AppWindow::new()?;
    let db = database::open()
        .expect("Failed to open cameras.db");




    // ========================================================
    // SEARCH CAMERAS
    // ========================================================

    let ui_weak = ui.as_weak();

ui.on_search_cameras(move || {

    let ui_weak = ui_weak.clone();

    // ----------------------------------------------------
    // Update UI immediately
    // ----------------------------------------------------

    if let Some(ui) = ui_weak.upgrade() {

        ui.set_is_scanning(true);

        ui.set_toast_message(
            "Searching for ONVIF cameras...".into()
        );

        ui.set_toast_kind(
            NotifKind::Info
        );

        ui.set_toast_visible(true);
    }


    // ----------------------------------------------------
    // Run Python in background
    // ----------------------------------------------------

    thread::spawn(move || {

        let result =
            discover_cameras_with_python();


        // ------------------------------------------------
        // Return to Slint UI thread
        // ------------------------------------------------

        let _ = slint::invoke_from_event_loop(
            move || {

                let Some(ui) =
                    ui_weak.upgrade()
                else {
                    return;
                };


                ui.set_is_scanning(false);


                match result {

                    // ====================================
                    // DISCOVERY SUCCESS
                    // ====================================

                    Ok(cameras) => {

                        // Open database for this operation
                        let db =
                            match database::open() {

                                Ok(db) => db,

                                Err(e) => {

                                    eprintln!(
                                        "Database error: {}",
                                        e
                                    );

                                    ui.set_toast_message(
                                        "Database error.".into()
                                    );

                                    ui.set_toast_kind(
                                        NotifKind::Alert
                                    );

                                    ui.set_toast_visible(true);

                                    return;
                                }
                            };


                        // --------------------------------
                        // Save discovered cameras
                        // --------------------------------

                        for camera in &cameras {

                            if let Err(e) =
                                database::upsert_camera(
                                    &db,
                                    camera
                                )
                            {

                                eprintln!(
                                    "Failed to save camera {}: {}",
                                    camera.id,
                                    e
                                );
                            }
                        }


                        // --------------------------------
                        // IMPORTANT:
                        //
                        // Reload the DATABASE after the
                        // discovery.
                        //
                        // This means the UI represents
                        // persistent state, not merely
                        // this discovery operation.
                        // --------------------------------

                        let stored =
                            match database::get_cameras(
                                &db
                            ) {

                                Ok(cams) => cams,

                                Err(e) => {

                                    eprintln!(
                                        "Failed loading cameras: {}",
                                        e
                                    );

                                    ui.set_toast_message(
                                        "Failed to load camera database."
                                            .into()
                                    );

                                    ui.set_toast_kind(
                                        NotifKind::Alert
                                    );

                                    ui.set_toast_visible(true);

                                    return;
                                }
                            };


                        // --------------------------------
                        // Convert DB cameras → Slint
                        // --------------------------------

                        let slint_cameras =
                            stored
                                .into_iter()
                                .enumerate()
                                .map(
                                    |(index, camera)| {

                                        to_slint_camera(
                                            camera,
                                            index
                                        )
                                    }
                                )
                                .collect::<Vec<_>>();


                        ui.set_cameras(
                            Rc::new(
                                slint::VecModel::from(
                                    slint_cameras
                                )
                            ).into()
                        );


                        // --------------------------------
                        // Notification
                        // --------------------------------

                        ui.set_toast_message(
                            format!(
                                "Discovery complete. {} camera(s) found.",
                                cameras.len()
                            ).into()
                        );

                        ui.set_toast_kind(
                            NotifKind::Info
                        );

                        ui.set_toast_visible(true);
                    }


                    // ====================================
                    // DISCOVERY FAILED
                    // ====================================

                    Err(error) => {

                        eprintln!(
                            "Python discovery error:\n{}",
                            error
                        );


                        ui.set_toast_message(
                            "Camera discovery failed.".into()
                        );

                        ui.set_toast_kind(
                            NotifKind::Alert
                        );

                        ui.set_toast_visible(true);
                    }
                }
            }
        );
    });
});


    // ========================================================
    // SYNC CLOUD
    // ========================================================

    let ui_weak = ui.as_weak();

    ui.on_sync_cloud(move || {

        let ui_weak = ui_weak.clone();

        if let Some(ui) = ui_weak.upgrade() {
            ui.set_is_syncing(true);
        }


        thread::spawn(move || {

            thread::sleep(
                std::time::Duration::from_secs(2)
            );


            let _ = slint::invoke_from_event_loop(
                move || {

                    let Some(ui) = ui_weak.upgrade() else {
                        return;
                    };

                    ui.set_is_syncing(false);

                    ui.set_toast_message(
                        "Synced with central cloud.".into()
                    );

                    ui.set_toast_kind(
                        NotifKind::Info
                    );

                    ui.set_toast_visible(true);
                }
            );
        });
    });


    // ========================================================
    // CHECK UPDATES
    // ========================================================

    let ui_weak = ui.as_weak();

    ui.on_check_updates(move || {

        let Some(ui) = ui_weak.upgrade() else {
            return;
        };

        ui.set_update_available(true);

        ui.set_toast_message(
            "Version 1.2.4 is available.".into()
        );

        ui.set_toast_kind(
            NotifKind::Update
        );

        ui.set_toast_visible(true);
    });


    // ========================================================
    // REMOVE CAMERA
    // ========================================================

    let ui_weak = ui.as_weak();

    ui.on_remove_camera(move |id| {

        let Some(ui) = ui_weak.upgrade() else {
            return;
        };


        let mut cameras: Vec<Camera> =
            ui.get_cameras()
                .iter()
                .collect();


        cameras.retain(|camera| {
            camera.id != id
        });


        ui.set_cameras(
            Rc::new(
                slint::VecModel::from(cameras)
            ).into()
        );
    });


    // ========================================================
    // RENAME CAMERA
    // ========================================================

    let ui_weak = ui.as_weak();

    ui.on_rename_camera(move |id, new_tag| {

        let Some(ui) = ui_weak.upgrade() else {
            return;
        };


        let mut cameras: Vec<Camera> =
            ui.get_cameras()
                .iter()
                .collect();


        for camera in &mut cameras {

            if camera.id == id {

                camera.tag = new_tag.clone();
            }
        }


        ui.set_cameras(
            Rc::new(
                slint::VecModel::from(cameras)
            ).into()
        );
    });


    // ========================================================
    // MANUALLY ADD CAMERA
    // ========================================================

    let ui_weak = ui.as_weak();

    ui.on_add_camera_manual(move |ip| {

        let Some(ui) = ui_weak.upgrade() else {
            return;
        };


        let mut cameras: Vec<Camera> =
            ui.get_cameras()
                .iter()
                .collect();


        let id = generate_id(
            cameras.len() + 1
        );


        cameras.push(Camera {

            id: id.into(),

            name: "Manual Camera".into(),

            tag: "Manual Camera".into(),

            ip,

            is_online: false,

            has_onvif: false,

            last_seen: "Never".into(),
        });


        ui.set_cameras(
            Rc::new(
                slint::VecModel::from(cameras)
            ).into()
        );
    });


    // ========================================================
    // DISMISS TOAST
    // ========================================================

    let ui_weak = ui.as_weak();

    ui.on_dismiss_toast(move || {

        if let Some(ui) = ui_weak.upgrade() {

            ui.set_toast_visible(false);
        }
    });


    // ========================================================
    // START APPLICATION
    // ========================================================

    println!(
        "Starting IBVAP Edge Command Center..."
    );

    ui.run()
}
