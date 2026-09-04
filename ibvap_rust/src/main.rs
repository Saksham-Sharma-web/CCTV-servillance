use slint::Model;
slint::include_modules!();
// This macro imports the generated Rust code from your .slint file

fn main() -> Result<(), slint::PlatformError> {
    // 1. Instantiate the AppWindow defined in main.slint
    let ui = AppWindow::new()?;

    // 2. We grab a weak reference to the UI to safely use inside callbacks
    let ui_handle = ui.as_weak();

    // 3. Register the callback for the "Search Cameras" button
    ui.on_search_cameras(move || {
        // Safe unwrap because the UI must exist if a button was clicked
        let ui = ui_handle.unwrap();

        println!("Button Clicked: Searching local network for ONVIF cameras...");

        // TODO: In the future, this is where you will send a message across
        // a channel to your PyO3 worker thread to run the discovery.py script.

        // For now, let's just log a dummy notification to prove the UI updates!
        let mut current_notifs: Vec<Notification> = ui.get_notifications().iter().collect();
        current_notifs.insert(0, Notification {
            time: slint::SharedString::from("NOW"),
            message: slint::SharedString::from("Initiating ONVIF network sweep..."),
            kind: NotifKind::Info,
        });

        // Update the UI model natively
        ui.set_notifications(std::rc::Rc::new(slint::VecModel::from(current_notifs)).into());
    });

    // Register the sync cloud callback
    ui.on_sync_cloud(|| {
        println!("Button Clicked: Syncing recent 50 events to the central cloud...");
    });

    // 4. Run the Slint event loop on the main thread
    println!("Starting IBVAP Edge Command Center UI...");
    ui.run()
}
