Build a complete, production-quality frontend UI/UX from scratch for an AI-based Border CCTV Surveillance and Intelligent Video Analytics Command Center.

IMPORTANT:
This is ONLY the frontend UI/UX prototype at this stage.

Do NOT build or depend on a real backend, database, authentication server, AI inference server, RTSP server, or API.

Use realistic MOCK DATA throughout the application so that the entire dashboard looks fully operational.

I will connect my own backend, database, CCTV streams, AI models and APIs later.

The application must therefore be architected cleanly so that all mock data can later be replaced by API/database data without redesigning the UI.

==================================================
1. PRODUCT PURPOSE
==================================================

The system is an intelligent border surveillance platform that uses existing CCTV infrastructure and AI video analytics to monitor a large secured area.

The platform should allow security personnel to:

- Monitor live CCTV cameras
- View camera health/status
- Detect and track people
- Track a person across multiple cameras
- Track vehicles
- Perform license plate/ANPR visualization
- Detect suspicious activities
- Detect intrusion
- Detect restricted-zone violations
- Detect loitering
- Detect night movement
- Monitor virtual fences
- Visualize CCTV camera coverage
- Visualize blind spots
- Track the last known position of a person
- Predict the likely movement of a person after they enter a CCTV blind spot
- Visualize predicted movement paths on a digital border blueprint/map
- Generate and manage alerts
- Investigate incidents
- View historical events
- View person/vehicle movement history
- View camera analytics
- Maintain evidence and audit information
- Monitor system health
- Manage surveillance zones
- View AI analytics statistics

The overall product should feel like a real government/security command-and-control application.

==================================================
2. VISUAL DESIGN DIRECTION
==================================================

The most important design requirement:

DO NOT make this look like a generic AI-generated SaaS dashboard.

Do NOT use:
- Purple gradients
- Neon blue/purple futuristic colors
- Excessive glowing effects
- Glassmorphism
- Excessive transparency
- Huge rounded cards
- Excessive shadows
- Floating 3D elements
- Cyberpunk styling
- Futuristic sci-fi UI
- Excessive animations
- Overly playful illustrations
- Generic AI robot graphics
- Excessive emojis
- Marketing-style SaaS sections

Instead create a serious, institutional, government/security-oriented interface.

Design inspiration:
- Government of India digital portals
- Indian government command-and-control systems
- Defence/security operations rooms
- Police surveillance control rooms
- Border security monitoring systems
- Mission-critical enterprise applications
- GIS command centers
- National infrastructure monitoring dashboards

The UI should communicate:

SECURITY
AUTHORITY
RELIABILITY
CONTROL
PRECISION
TRUST
OPERATIONAL READINESS

==================================================
3. COLOR SYSTEM
==================================================

Use a restrained Indian government-inspired palette.

Primary:
- Deep Navy: #102A43
- Dark Navy: #0B1F33

Secondary:
- Government Blue: #1F4E79
- Steel Blue: #3E6078

Background:
- Very Light Warm Gray: #F4F5F3
- White: #FFFFFF

Borders:
- #D6DADF
- #C4CBD2

Text:
- Primary: #17202A
- Secondary: #5F6B76

Status colors:
- Critical Red: #B42318
- High/Warning Amber: #B54708
- Success Green: #18794E
- Information Blue: #175CD3

Use muted colors rather than highly saturated colors.

Indian tricolor colors may be used VERY subtly as accent details:
- Saffron #E87817
- White
- India Green #138808

Do not turn the entire UI orange/green.

Use navy as the dominant identity color.

==================================================
4. TYPOGRAPHY
==================================================

Use a professional government/enterprise typography system.

Prefer:
- Inter
- Source Sans 3
- IBM Plex Sans

Use clear hierarchy.

Do not use futuristic fonts.

Typography should prioritize readability and information density.

==================================================
5. OVERALL APPLICATION STRUCTURE
==================================================

Create a desktop-first command center application.

Main layout:

LEFT:
Persistent vertical navigation sidebar.

TOP:
Government/security style header with:
- Organization/application identity
- Global search
- Current operational status
- Notification indicator
- User profile
- System status

MAIN:
Page content.

The interface should work properly on:
- Desktop
- Large monitors
- Laptop
- Tablet

Desktop is the primary target because this is a command center application.

==================================================
6. APPLICATION BRANDING
==================================================

Create a professional application identity.

Suggested name:

"National Border Surveillance & Video Analytics System"

Short name:

"NB-SVAS"

Subtitle:

"Intelligent Border Surveillance Command Center"

Do NOT falsely claim that this is an official Government of India website.

Use a neutral government-style identity.

Do NOT use the official Ashoka emblem unless explicitly provided as an asset.

Instead use a simple professional geometric/security insignia.

The visual language should be inspired by Indian government systems without impersonating a real government department.

==================================================
7. SIDEBAR NAVIGATION
==================================================

Create a persistent sidebar with these sections:

1. Command Center
2. Live Surveillance
3. Situation Map
4. People Tracking
5. Vehicle & ANPR
6. Alerts & Incidents
7. AI Analytics
8. Zones & Virtual Fence
9. Blind Spot Analysis
10. Investigation
11. Camera Infrastructure
12. System Health
13. Evidence & Audit
14. Reports
15. User Management
16. Settings

Sidebar should have:
- Icon
- Label
- Active state
- Hover state
- Collapse/expand functionality

Use professional line icons.

==================================================
8. TOP HEADER
==================================================

Header should contain:

LEFT:
Application title and current module.

CENTER:
Global search bar.

Search placeholder:

"Search camera, person ID, vehicle, plate, incident..."

RIGHT:
- System operational indicator
- Notification icon
- Current date/time
- User profile
- Role
- Dropdown

Example:

SYSTEM STATUS
● OPERATIONAL

USER:
Duty Officer
Operations Control

Avoid overly modern SaaS styling.

==================================================
9. COMMAND CENTER / MAIN DASHBOARD
==================================================

The Command Center is the primary landing page.

It must immediately answer:

WHAT IS HAPPENING?
WHERE IS IT HAPPENING?
WHO IS INVOLVED?
HOW SERIOUS IS IT?
WHAT SHOULD THE OFFICER DO?

Create the following sections.

--------------------------------------------------
A. OPERATIONAL KPI CARDS
--------------------------------------------------

Create 6 compact information cards:

1. Cameras Online
Example:
48 / 50
96% operational

2. Active People
Example:
127

3. Vehicles Detected
Example:
42

4. Active Alerts
Example:
23

5. Critical Alerts
Example:
4

6. Active Blind Spot Events
Example:
3

Each card should include:
- Small icon
- Metric
- Label
- Comparison/status
- Small contextual information

Avoid oversized cards.

--------------------------------------------------
B. BORDER SITUATION MAP
--------------------------------------------------

This should be the MOST IMPORTANT visual component on the dashboard.

Create a large interactive-looking digital border blueprint/map.

Use a realistic fictional border facility layout.

The map should contain:

- Border line
- Fence
- Gates
- Roads
- Buildings
- Restricted zones
- Patrol zones
- CCTV cameras
- Camera coverage/FOV
- Blind spots
- People
- Vehicles
- Active incidents
- Entry/exit points

Display camera markers such as:

CAM-01
CAM-02
CAM-03
CAM-04
CAM-05

Show camera FOV as translucent cones/polygons.

Show blind spots using a subtle hatched/outlined region.

Show detected people as small markers.

Show active tracked person with:
- Person ID
- Direction arrow
- Movement trail

Example:

P-102
● → → →

If person enters a blind spot, show:

TRACK LOST
LAST SEEN
14:32:18

Then show a predicted trajectory.

Example:

CAM-07
   ↓
   ↓
BLIND SPOT
   ↓
Predicted Path
   ↓
CAM-09

Display:

Prediction Confidence:
81%

Likely Next Camera:
CAM-09

The map should support:
- Zoom
- Pan
- Layer toggles
- Camera layer
- Person layer
- Vehicle layer
- Alert layer
- Blind spot layer
- Zone layer

Create a map legend.

--------------------------------------------------
C. CRITICAL ALERT PANEL
--------------------------------------------------

Right-side panel showing most important live alerts.

Each alert should show:

Severity
Event type
Location
Camera
Person/Vehicle ID
Timestamp
Status

Examples:

CRITICAL
Unauthorized Access Detected
Main Gate - Entrance 2
CAM-07
P-102
2 min ago

HIGH
Restricted Zone Entry
Sector A
CAM-12
P-204
5 min ago

MEDIUM
Loitering Detected
Loading Bay
CAM-03
10 min ago

INFO
Vehicle Detected
Front Desk
CAM-03
15 min ago

Use:
- Red for critical
- Amber for high
- Yellow/neutral for medium
- Blue for information

Do NOT overuse red.

--------------------------------------------------
D. LIVE CAMERA WALL
--------------------------------------------------

Create a 2x2 live camera grid.

Use realistic surveillance/campus/border facility imagery as placeholder content.

Each camera card:

CAM-01
Main Gate - Entrance 1

LIVE indicator

FPS
24

AI
ACTIVE

People:
3

Vehicles:
1

Alerts:
0

Timestamp

Each camera should support:
- Fullscreen
- Mute
- Snapshot
- Camera details
- Open tracking
- Open map location

Use realistic CCTV imagery rather than generic colorful AI imagery.

--------------------------------------------------
E. EVENT TIMELINE
--------------------------------------------------

Create a real-time operational event timeline.

Example:

14:32:18
Person P-102 detected
CAM-07

14:31:04
Virtual fence violation
CAM-07

14:29:41
P-102 transitioned from CAM-05 → CAM-07

14:27:19
Vehicle V-204 detected
CAM-05

14:25:03
Loitering detected
CAM-03

Each event should be clickable.

==================================================
10. LIVE SURVEILLANCE PAGE
==================================================

Create a dedicated Live Surveillance page.

Features:

- Camera grid
- 2x2
- 3x3
- 4x4 view
- Fullscreen mode
- Search camera
- Filter by sector
- Filter by status
- Filter by AI activity
- Critical activity filter

Camera card must show:

- Camera ID
- Location
- Live status
- FPS
- Resolution
- AI status
- Detection count
- Current alerts
- Timestamp

Create realistic placeholder surveillance feeds.

==================================================
11. CAMERA DETAIL PAGE
==================================================

Clicking a camera opens detailed view.

Show:

Large live feed

Camera information:
- Camera ID
- Location
- Sector
- Resolution
- FPS
- Protocol
- Stream status
- AI processing status
- Last heartbeat
- Latency
- Coverage area
- FOV
- Blind spots

Below:

Recent events
- Person detected
- Vehicle detected
- Intrusion
- Loitering
- etc.

==================================================
12. SITUATION MAP PAGE
==================================================

Create a full-screen map interface.

This is the advanced version of the dashboard map.

Controls:

Layers:
[✓] Cameras
[✓] People
[✓] Vehicles
[✓] Alerts
[✓] Blind Spots
[✓] Restricted Zones
[✓] Virtual Fences

Map controls:
- Zoom
- Search location
- Center on active incident
- Fullscreen

Clicking a person opens:

PERSON P-102

Status:
TRACKING

Current Camera:
CAM-07

Direction:
North-East

Speed:
1.7 m/s

Confidence:
94%

Clicking a camera opens camera details.

Clicking a blind spot opens:

BLIND SPOT BS-03

Risk:
HIGH

Affected Cameras:
CAM-07, CAM-08

Last Person:
P-102

==================================================
13. PEOPLE TRACKING PAGE
==================================================

Create a professional person tracking interface.

Top:

Search:
"Search Person ID"

Filters:
- Active
- Lost
- Reacquired
- Critical
- Last hour
- Last 24 hours

Table:

Person ID
Status
First Seen
Last Seen
Current Camera
Direction
Speed
Risk
Confidence

Example:

P-102
TRACKING
14:21
14:32
CAM-07
NE
1.7 m/s
HIGH
94%

Clicking a person opens a detailed investigation-style page.

==================================================
14. PERSON TRACK DETAIL
==================================================

Show:

Person ID:
P-102

Current Status:
TRACK LOST / TRACKING / REACQUIRED

Last Known Camera:
CAM-07

Last Seen:
14:32:18

Direction:
North-East

Speed:
1.7 m/s

Risk Score:
82 / 100

Then show:

CAMERA TRANSITION HISTORY

CAM-02
↓
CAM-04
↓
CAM-05
↓
CAM-07
↓
BLIND SPOT
↓
PREDICTED
↓
CAM-09

Show trajectory visually on map.

--------------------------------------------------
BLIND SPOT PREDICTION
--------------------------------------------------

Create a dedicated panel:

TRACK LOST

Reason:
Person entered known CCTV blind spot.

Last known location:
Sector A / North Corridor

Direction:
North-East

Estimated speed:
1.7 m/s

Predicted location:
Sector B

Likely next cameras:

CAM-09
81%

CAM-10
62%

CAM-11
41%

Show predicted route on the map.

==================================================
15. VEHICLE & ANPR PAGE
==================================================

Create vehicle surveillance section.

Show:

Vehicle ID
Type
Color
License Plate
Camera
Timestamp
Direction
Confidence
Status

Example:

V-204
SUV
White
UP16AB1234
CAM-07
14:31
North
96%

Create ANPR section.

Recent Plates:

UP16AB1234
DL09XY7788
UP14CD2201

Add watchlist status:

NORMAL
WATCHLIST
UNKNOWN
FLAGGED

==================================================
16. ALERTS & INCIDENTS PAGE
==================================================

Create full incident management system.

Tabs:

All
Critical
High
Medium
Low
Resolved

Each incident:

Incident ID
Severity
Type
Camera
Location
Person/Vehicle
Timestamp
Status
Assigned Officer

Statuses:

NEW
ACKNOWLEDGED
INVESTIGATING
RESOLVED

Create professional incident cards/table.

==================================================
17. ALERT DETAIL PAGE
==================================================

When clicking an incident:

Show:

Incident ID
Severity
Event type
Location
Camera
Person ID
Vehicle ID
Timestamp
AI confidence
Risk score

Show large evidence image/video placeholder.

Show:

EVENT TIMELINE

Show:

TRACK HISTORY

Show:

MAP LOCATION

Actions:

ACKNOWLEDGE
ASSIGN
INVESTIGATE
RESOLVE
EXPORT REPORT

==================================================
18. AI ANALYTICS PAGE
==================================================

Create an AI analytics monitoring dashboard.

Show modules:

Person Detection
Vehicle Detection
Person Tracking
Face Detection
ANPR
Intrusion Detection
Loitering Detection
Suspicious Activity
Night Movement

Each module shows:

Status
Detections
Confidence
Processing FPS
Latency

Example:

PERSON DETECTION

Status:
ACTIVE

Detections:
127

Average Confidence:
94.2%

Processing:
24 FPS

Latency:
82 ms

Also create charts:

- Detections over time
- Incidents over time
- Person activity
- Vehicle activity
- Camera activity
- AI event distribution

Charts should be professional and restrained.

==================================================
19. ZONES & VIRTUAL FENCE PAGE
==================================================

Create zone management UI.

Show:

Restricted Zones
Patrol Zones
High Security Zones
Entry Zones
Exit Zones

Allow frontend simulation of:

- Create zone
- Edit zone
- Delete zone
- Enable/disable zone
- Set severity
- Set rules

Visualize zones on map.

Virtual fence:

Show line/polygon.

Example:

VIRTUAL FENCE VF-04

Status:
ACTIVE

Camera:
CAM-07

Zone:
Restricted Sector A

Trigger:
Person crosses boundary

Severity:
CRITICAL

==================================================
20. BLIND SPOT ANALYSIS PAGE
==================================================

Create dedicated blind spot analytics.

Show:

Total Blind Spots
12

Critical:
3

High:
4

Medium:
5

List:

BS-01
North Corridor
HIGH

BS-02
Storage Area
MEDIUM

BS-03
Sector A
CRITICAL

Each blind spot should show:

- Location
- Area
- Affected cameras
- Risk level
- Coverage gap
- Nearby cameras
- Last activity
- Recommended camera coverage

Map should highlight blind spots.

==================================================
21. INVESTIGATION PAGE
==================================================

This should feel like a professional investigation workstation.

Search by:

- Person ID
- Vehicle ID
- License plate
- Camera
- Incident ID
- Date/time

After selecting a person:

Show:

IDENTITY / TRACK ID

CAMERA HISTORY

TIMELINE

MAP TRAJECTORY

ALERT HISTORY

VEHICLE ASSOCIATIONS

FACE MATCHES

BLIND SPOT EVENTS

PREDICTED MOVEMENT

EVIDENCE

The entire investigation should be connected visually.

==================================================
22. CAMERA INFRASTRUCTURE PAGE
==================================================

Create camera management.

Show table:

Camera ID
Location
Sector
Status
FPS
Resolution
AI
Last Heartbeat
Latency
Coverage

Example:

CAM-01
Main Gate
Sector A
ONLINE
24 FPS
1080p
ACTIVE
14:32:21
82 ms

Status types:

ONLINE
OFFLINE
DEGRADED
MAINTENANCE

==================================================
23. SYSTEM HEALTH PAGE
==================================================

Create infrastructure health dashboard.

Metrics:

Camera availability
AI processing
Network status
Database status
Storage
Stream health
API status

Example:

CAMERAS
48 / 50 ONLINE

AI SERVICES
8 / 8 ACTIVE

DATABASE
OPERATIONAL

STORAGE
72%

NETWORK
STABLE

Create system health timeline.

==================================================
24. EVIDENCE & AUDIT PAGE
==================================================

Create evidence management.

Each incident may contain:

- Snapshot
- Video clip
- Person track
- Vehicle track
- Camera ID
- Timestamp
- Location
- AI detection
- Incident record

Show:

Evidence ID
Incident ID
Type
Created
Hash
Integrity
Status

Example:

EV-2031
Video
INC-1024
14:32
8f7a...91bc
VERIFIED

Create an audit log:

Timestamp
User
Action
Resource
Result

Example:

14:34
Duty Officer
Acknowledged Incident
INC-1024
SUCCESS

==================================================
25. REPORTS PAGE
==================================================

Create report generation UI.

Report types:

Daily Surveillance Report
Incident Report
Camera Health Report
AI Analytics Report
Vehicle Detection Report
Person Movement Report
Blind Spot Report

Show date range selector.

Buttons:

GENERATE REPORT
EXPORT PDF
EXPORT CSV

Use mock functionality.

==================================================
26. USER MANAGEMENT
==================================================

Create role-based frontend UI.

Roles:

Administrator
Commander
Security Officer
Investigator
Viewer

Show:

User
Role
Status
Last Login
Permissions

Permissions include:

View Cameras
View Alerts
Investigate
Manage Zones
Manage Cameras
Export Evidence
Manage Users

==================================================
27. SETTINGS PAGE
==================================================

Create settings for:

General
Notifications
Alert thresholds
Camera preferences
Map preferences
Display
Security
User preferences

==================================================
28. GLOBAL SEARCH
==================================================

The global search should be functional on the frontend using mock data.

Search:

Camera
Person ID
Vehicle ID
License plate
Incident ID
Location

Example:

Search:
P-102

Results:

Person P-102
Current Camera: CAM-07
Last Seen: 14:32
Risk: HIGH

==================================================
29. MOCK DATA ARCHITECTURE
==================================================

Create structured mock data files/modules.

Do NOT hardcode data directly inside UI components.

Separate mock data into:

- cameras
- persons
- vehicles
- alerts
- incidents
- events
- zones
- blindSpots
- tracks
- systemHealth
- users

The UI should consume this mock data through clean reusable structures.

I will replace these later with backend API calls.

==================================================
30. COMPONENT ARCHITECTURE
==================================================

Create reusable components:

- Sidebar
- Header
- KPI Card
- Alert Card
- Camera Card
- Camera Grid
- Map
- Map Marker
- Person Marker
- Vehicle Marker
- Blind Spot Overlay
- Zone Overlay
- Event Timeline
- Status Badge
- Severity Badge
- Data Table
- Search
- Filter Bar
- Modal
- Drawer
- Detail Panel
- Chart
- Evidence Viewer
- Track Timeline

Do not duplicate UI code unnecessarily.

==================================================
31. INTERACTION REQUIREMENTS
==================================================

This must NOT be a static mockup.

Implement frontend interactions.

Examples:

Click camera:
→ Open camera detail.

Click person:
→ Open person tracking.

Click alert:
→ Open incident detail.

Click blind spot:
→ Open blind spot details.

Click map person:
→ Show tracking information.

Click predicted route:
→ Show prediction information.

Click "Track on Map":
→ Open map and focus on person.

Click "Investigate":
→ Open investigation page.

Click "Acknowledge":
→ Change alert state.

Click "Resolve":
→ Change incident state.

Filters should actually filter mock data.

Search should actually search mock data.

Tabs should work.

Sidebar navigation should work.

Modals/drawers should work.

==================================================
32. REALISTIC MOCK DATA
==================================================

Do NOT use repetitive dummy data such as:

Person 1
Person 2
Person 3

Instead create realistic identifiers:

P-102
P-117
P-204
P-309

Cameras:

CAM-01
CAM-02
CAM-03
CAM-04
CAM-07
CAM-09
CAM-12

Vehicles:

V-102
V-204
V-311

Incidents:

INC-2026-001
INC-2026-002

Evidence:

EV-2031
EV-2032

Locations:

Main Gate
North Corridor
Sector A
Sector B
Loading Area
Perimeter Fence
Restricted Zone
Watch Tower 03

Use realistic timestamps.

==================================================
33. SURVEILLANCE IMAGERY
==================================================

For CCTV placeholder imagery, use realistic surveillance/security camera footage imagery.

Avoid:
- Stock photos of smiling people
- Marketing photography
- AI-generated futuristic scenes

Prefer:
- CCTV-style images
- Security camera viewpoints
- Entrance gates
- Industrial facilities
- Border infrastructure
- Roads
- Fences
- Checkpoints
- Warehouses
- Security perimeters

The imagery should look like it comes from surveillance cameras.

==================================================
34. DATA VISUALIZATION
==================================================

Charts should include:

Incident trend
Detection trend
Camera activity
People detected
Vehicles detected
Alert distribution
Hourly activity
Night activity
Blind spot events

Charts should be simple and professional.

Avoid excessive gradients and decorative charts.

==================================================
35. ALERT PRIORITY VISUAL LANGUAGE
==================================================

Critical:
Dark red accent

High:
Amber

Medium:
Muted yellow/orange

Low:
Blue/gray

Normal:
Green

Do not make every component colorful.

Most of the interface should remain:
White
Warm gray
Navy
Dark text
Muted borders

==================================================
36. GOVERNMENT / COMMAND CENTER VISUAL DETAILS
==================================================

Use subtle details that make the interface feel institutional:

- Thin borders
- Compact spacing
- Structured information hierarchy
- Section labels
- Small uppercase metadata labels
- Official-looking tables
- Dense but readable information
- Minimal decorative elements
- Clear operational status indicators
- Professional iconography
- Consistent alignment
- Conservative corner radius

Cards should have approximately 6–10px radius, not huge 24–32px radius.

Avoid excessive floating cards.

==================================================
37. MAP STYLE
==================================================

The map should NOT look like Google Maps.

Create a professional operational GIS-style map.

Use:

- Dark or neutral blueprint base
- Thin infrastructure lines
- Sector boundaries
- Camera FOV
- Restricted zones
- Blind spots
- Movement trajectories
- Alert markers

The map should look like a surveillance operations map.

==================================================
38. RESPONSIVENESS
==================================================

Desktop:
Primary command-center layout.

Tablet:
Collapse sidebar and reorganize cards.

Mobile:
Do not attempt to show the full command center exactly as desktop.

Use:
- Collapsible navigation
- Scrollable camera feeds
- Stacked alerts
- Simplified map
- Responsive tables

==================================================
39. ACCESSIBILITY
==================================================

Use:

- Strong contrast
- Readable typography
- Clear status indicators
- Do not rely only on color
- Keyboard-friendly controls
- Proper labels
- Tooltips for icons
- Clear focus states

==================================================
40. PERFORMANCE
==================================================

Keep the frontend performant.

Do not load unnecessary large assets.

Use lazy loading where appropriate.

Do not create excessive animations.

The application should feel like mission-critical software rather than a marketing website.

==================================================
41. ANIMATIONS
==================================================

Use VERY subtle animations only where operationally useful.

Allowed:

- New alert highlight
- Status change
- Map marker movement
- Camera loading
- Panel transition
- Modal transition

Do NOT use:
- Floating animations everywhere
- Gradient animations
- Parallax
- Excessive hover effects
- Glowing elements

==================================================
42. IMPORTANT EMPTY / ERROR / OFFLINE STATES
==================================================

Design all states.

Camera offline:

CAM-17
OFFLINE

Last heartbeat:
14:28:11

Stream unavailable.

Person tracking lost:

TRACK LOST

Last known location:
Sector A

Last seen:
14:32:18

Blind spot prediction available.

No alerts:

NO ACTIVE ALERTS

System operational.

Loading state:

Use professional skeleton loaders.

Error state:

Unable to load surveillance feed.

==================================================
43. DEMO MODE
==================================================

Add a frontend-only DEMO MODE.

Demo mode should simulate:

- New person detection
- Camera transition
- Alert generation
- Person entering blind spot
- Track lost
- Predicted movement
- Track reacquired

Example demo sequence:

CAM-04 detects P-102

↓

P-102 moves to CAM-05

↓

P-102 enters restricted zone

↓

Critical alert generated

↓

P-102 moves toward CAM-07

↓

P-102 enters blind spot

↓

Tracking temporarily lost

↓

System predicts route

↓

Prediction points toward CAM-09

↓

CAM-09 detects P-102

↓

Track reacquired

The dashboard should visually update during the demo.

This will be extremely useful for presentation purposes.

==================================================
44. FRONTEND TECHNOLOGY
==================================================

Use a modern frontend architecture.

Preferred:

React / Next.js

Use:
- TypeScript
- Tailwind CSS or clean CSS architecture
- Reusable components
- Component-based architecture
- Proper routing
- Mock service/data layer

For charts use a suitable charting library.

For maps, use a frontend map solution or a custom SVG/blueprint implementation if necessary.

The application must run locally without a backend.

==================================================
45. DESIGN QUALITY BAR
==================================================

The final result should feel like a serious operational application that could be shown to:

- Government officials
- Security agencies
- Defence personnel
- Police/security departments
- SIH judges
- Technical evaluators

It should NOT feel like:

- A student CRUD dashboard
- A generic admin panel
- A SaaS analytics template
- An AI-generated UI template
- A futuristic cyber-security landing page

It should feel like:

"Mission-critical government surveillance command software."

==================================================
46. MOST IMPORTANT HOME SCREEN PRIORITY
==================================================

On the Command Center page, prioritize visual hierarchy as follows:

1. Border Situation Map
2. Critical Alerts
3. Live CCTV Monitoring
4. Person/Vehicle Activity
5. Event Timeline
6. System/Camera Health
7. Analytics

Do not allow analytics charts to visually dominate the live operational information.

==================================================
47. FINAL USER EXPERIENCE
==================================================

When an officer opens the system, they should immediately understand:

SYSTEM STATUS:
OPERATIONAL

CAMERAS:
48 / 50 ONLINE

ACTIVE ALERTS:
23

CRITICAL:
4

ACTIVE PEOPLE:
127

ACTIVE VEHICLES:
42

Then they should see the border situation map.

They should immediately be able to identify:

- Where the cameras are
- What cameras cover
- Where people are
- Where vehicles are
- Where alerts are occurring
- Where blind spots exist
- Where a tracked person was last seen
- Where the person may have moved
- Which camera is likely to detect them next

The entire UI should support the operational workflow:

DETECT
→ TRACK
→ ANALYZE
→ ALERT
→ PREDICT
→ INVESTIGATE
→ RESOLVE
→ AUDIT

==================================================
48. FINAL REQUIREMENT
==================================================

Build the entire application, not just the home dashboard.

Every sidebar item must lead to a properly designed page.

Every major page must contain realistic mock data.

Every major interaction must work in frontend demo mode.

Do not leave placeholder pages saying:

"Coming Soon"

Do not create empty sections.

Do not create a generic dashboard template.

Create the complete UI/UX system with realistic operational data and interconnected interactions.

The final application should look like a mature government/security surveillance command platform ready to have a real backend connected to it.

Start by implementing the complete application structure, design system, navigation, Command Center dashboard, Situation Map, Live Surveillance, People Tracking, Vehicle/ANPR, Alerts, AI Analytics, Zones, Blind Spots, Investigation, Camera Infrastructure, System Health, Evidence/Audit, Reports, Users and Settings.

Prioritize visual consistency and operational usability across every page.

REFINEMENT INSTRUCTION:

The current design still feels too much like a generic modern SaaS/AI dashboard.

Redesign the visual language to feel more like a Government of India security command-and-control application.

Reduce:
- rounded cards
- gradients
- excessive whitespace
- decorative elements
- bright colors
- glass effects
- futuristic UI
- unnecessary animations

Increase:
- information density
- structured tables
- thin borders
- navy/steel blue institutional colors
- warm white/gray surfaces
- compact controls
- operational labels
- clear status indicators
- map-centric workflow
- surveillance terminology
- command-center hierarchy

The application should look serious, mature, institutional and mission-critical.

Imagine that this software is being used by a security officer sitting in a national border surveillance control room.

It should look like software built for an actual government/security organization, not a startup pitch deck.

Keep the interface modern and polished, but conservative and authoritative.

Do not make it visually old-fashioned either.

Target:
"Modern Government Command Center"
rather than
"Modern AI SaaS Dashboard".