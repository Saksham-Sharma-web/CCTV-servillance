# Requirements

- Human detection and tracking
- Vehicle detection and classification
- Face detection
- Automatic Number Plate Recognition (ANPR)
- Virtual fence intrusion detection
- Suspicious activity detection
- Night-time movement detection
- Real-time alert generation and event logging
- Expected Solution The proposed system should leverage Artificial Intelligence, Machine Learning, Computer Vision, and Video Analytics to create a software-defined surveillance platform capable of extracting actionable intelligence from existing CCTV infrastructure.

# The Solution Must

- Eliminate dependence on expensive dedicated surveillance hardware.
- Enable intelligent monitoring through AI-powered video analytics.
- Provide real-time alerts for security incidents and border intrusions.
- Support facial recognition, vehicle identification, and behavioral analytics through software.
- Authorised person database...
- Human officer verification...
- virtual Boundary/ fence...
- Improve situational awareness and response time for border security forces.
- Support integration with existing command and control systems.
- The final solution should be cost-effective, scalable, and suitable for deployment across remote border locations and strategic installations.
- Possible Project Name IBVAP â€“ Intelligent Border Video Analytics Platform

## ANPR

- Manage whitelisting, blacklisting, and watchlist, frequently observed and unknown vehicles.

# IBVAP — Intelligent Border Video Analytics Platform

### System Architecture (Redesigned for Clarity)

> **Core principle:** *Process locally. Detect locally. Store locally. Synchronize centrally.*

---

## 1. One-Line Summary

Every border region runs its own **Edge Node** that watches its cameras, detects events with AI, and stores footage locally. Only lightweight metadata, alerts, and health pings travel to the **Central Platform** — full video never leaves the region unless someone specifically requests it.

---

## 2. High-Level System Overview

```mermaid
flowchart TB
    subgraph CENTRAL["☁️ CENTRAL IBVAP PLATFORM"]
        API["Central Backend / API<br/>• Auth & Users<br/>• Region Mgmt<br/>• Alert & Event Aggregation<br/>• Node Health Monitoring"]
        DASH["Web Dashboard<br/>• Live Events<br/>• Alerts<br/>• Node Status<br/>• Footage Requests"]
        API --> DASH
    end

    subgraph A["🌍 REGION A"]
        CAMA["IP Cameras"] --> EDGEA["IBVAP Edge Node<br/>AI Detection + Tracking"]
        EDGEA --> STOREA["Local Event Store<br/>(Metadata + Footage)"]
    end

    subgraph B["🌍 REGION B"]
        CAMB["IP Cameras"] --> EDGEB["IBVAP Edge Node<br/>AI Detection + Tracking"]
        EDGEB --> STOREB["Local Event Store<br/>(Metadata + Footage)"]
    end

    STOREA -- "Metadata / Alerts / Health" --> API
    STOREB -- "Metadata / Alerts / Health" --> API
```

**Key idea:** the Central Platform is a *thin coordinator*, not a video pipe. Each region is self-sufficient.

---

## 3. What Each Edge Node Does

```mermaid
flowchart LR
    CAM["📷 Camera Feed"] --> ING["Ingestion"]
    ING --> AI["AI Analytics Engine"]
    AI --> H["Human Detection"]
    AI --> V["Vehicle Detection"]
    AI --> F["Face Detection"]
    AI --> P["ANPR (Plates)"]
    AI --> FE["Virtual Fence"]
    AI --> SUS["Suspicious Activity"]
    AI --> N["Night Detection"]
    H & V & F & P & FE & SUS & N --> EV["Event Generator"]
    EV --> STORE[("Local Event Store")]
```

| Function | Purpose |
| --- | --- |
| Human / Vehicle Detection | Identify people & vehicles crossing camera view |
| Face Detection | Flag and log faces for review |
| ANPR | Read vehicle number plates automatically |
| Virtual Fence | Trigger alert when a defined boundary line is crossed |
| Suspicious Activity | Behavioral pattern flags (loitering, climbing, etc.) |
| Night Detection | Low-light / IR-based detection mode |

---

## 4. Regional Network Layout

```mermaid
flowchart LR
    subgraph VLAN20["VLAN 20 — Camera Segment"]
        CAM["IP Cameras / CCTV"]
    end
    subgraph VLAN10["VLAN 10 — Compute Segment"]
        EDGE["IBVAP Edge Node<br/>(AI + Analytics)"]
    end
    CAM -- "Video Stream (isolated)" --> EDGE
    EDGE --> LOCAL[("Local Event Storage")]
    LOCAL -- "WAN / Internet" --> CENTRAL(["Central Platform"])
```

Cameras live on an **isolated VLAN** — they can only talk to the Edge Node, never directly to the internet or the Central Platform.

---

## 5. Event Lifecycle

```mermaid
flowchart TD
    A["Camera detects activity"] --> B["Edge AI analyzes video"]
    B --> C{"Security event?"}
    C -- "No" --> D["Continue monitoring"]
    C -- "Yes" --> E["Create Event"]
    E --> F["Store locally:<br/>Timestamp • Camera ID • Type • Confidence • Footage"]
    F --> G["Send lightweight metadata"]
    G --> H["Central Platform"]
    H --> I["Dashboard Alert"]
```

---

## 6. Footage Retrieval (On-Demand Only)

```mermaid
sequenceDiagram
    participant U as Dashboard User
    participant C as Central Platform
    participant E as Edge Node
    participant S as Local Storage

    U->>C: Request event footage
    C->>E: Forward request
    E->>S: Retrieve stored clip
    S-->>E: Footage
    E-->>C: Send footage
    C-->>U: Deliver footage
```

Continuous video **never** streams to the center — only the specific clip requested, and only when authorized.

---

## 7. Offline Resilience

```mermaid
flowchart LR
    subgraph OFFLINE["🔌 Internet Lost"]
        C1["Cameras keep recording"] --> A1["Edge AI keeps detecting"] --> S1["Events stored locally"]
    end
    OFFLINE -- "Connection restored" --> SYNC["Edge Node Syncs:<br/>Missed Events • Metadata • Node Status • Requested Footage"]
    SYNC --> CENTRAL(["Central Platform"])
```

| While offline | Status |
| --- | --- |
| Cameras recording | ✅ Continues |
| AI detection | ✅ Continues |
| Event storage | ✅ Continues |
| Central sync | ⏸ Paused (auto-resumes) |

---

## 8. Security Model — Who Can Talk to Whom

```mermaid
flowchart LR
    CAM["Camera Network"] -->|"✅ Allowed"| EDGE["Edge Node"]
    EDGE -->|"✅ Allowed"| CENTRAL["Central Platform"]
    CAM -.->|"❌ Denied"| NET["Internet"]
    CAM -.->|"❌ Denied"| CENTRAL
    CENTRAL -.->|"❌ Denied"| CAM
    REGA["Region A"] -.->|"❌ Denied"| REGB["Region B"]
```

The **Edge Node is the only gatekeeper** between raw surveillance infrastructure and the outside world. No direct region-to-region access exists.

---

## 9. Secure Software Updates

```mermaid
flowchart TD
    U["Central Update Server"] -->|"Signed Package"| E["Edge Node"]
    E --> V{"Signature Valid?"}
    V -- "No" --> R["Reject Update"]
    V -- "Yes" --> I["Install Update"]
```

---

## 10. Prototype Demo vs. Real Deployment

| Stage | 🧪 Prototype Demo | 🏗️ Real Deployment |
| --- | --- | --- |
| Camera | Smartphone camera | Fixed IP CCTV camera |
| Network | Local Wi-Fi hotspot | Segmented VLAN CCTV network |
| Edge Node | Laptop running IBVAP software | Dedicated Edge processing server |
| Uplink | Internet (Wi-Fi/mobile) | Secure WAN / Internet |
| Platform | Same hosted Central Platform | Same hosted Central Platform |

The software stack is **identical** in both cases — only the hardware at the edges changes. This is the key point to make in a presentation: *the same platform scales from a laptop demo to a real CCTV deployment without redesign.*

---

## 11. Final Summary Diagram

```mermaid
flowchart TD
    A["Existing CCTV Infrastructure"] --> B["Local IP Network"]
    B --> C["IBVAP Edge Node"]
    C --> D["Real-Time AI Processing"]
    C --> E["Local Event Storage"]
    C --> F["Offline Operation"]
    D & E & F --> G["Secure Internet / WAN"]
    G --> H["Central IBVAP Platform"]
    H --> I["Auth • Dashboard • Alerts • Multi-Region Monitoring"]
```

---

## 12. Presentation Talking Points

- The live demo validates real software behavior using **smartphones as IP cameras** and isolated Wi-Fi as simulated regions.
- The same architecture maps directly onto a **segmented IP CCTV network** in a real deployment.
- AI inference keeps running at the Edge Node **even without internet** — only central sync pauses.
- The Central Platform receives **metadata, alerts, health status, and requested footage only** — never continuous raw video.
