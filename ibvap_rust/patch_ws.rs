use std::fs;

fn main() {
    let mut code = fs::read_to_string("src/web_server.rs").unwrap();
    let target = r#"async fn handle_ws_events(mut socket: WebSocket, state: AppState) {
    let mut rx = state.ws_sender.subscribe();

    while let Ok(_notif) = rx.recv().await {
        if socket.send(Message::Text("update".to_string())).await.is_err() {
            // Client disconnected
            break;
        }
    }
}"#;

    let replacement = r#"async fn handle_ws_events(mut socket: WebSocket, state: AppState) {
    let mut rx = state.ws_sender.subscribe();

    loop {
        tokio::select! {
            msg = socket.recv() => {
                if msg.is_none() {
                    break;
                }
            }
            Ok(_notif) = rx.recv() => {
                if socket.send(Message::Text("update".to_string())).await.is_err() {
                    break;
                }
            }
        }
    }
}"#;

    if code.contains(target) {
        code = code.replace(target, replacement);
        fs::write("src/web_server.rs", code).unwrap();
        println!("Patched successfully");
    } else {
        println!("Target not found");
    }
}
