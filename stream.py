import discovery
import connection

devices = discovery.discover()
for device in devices:
    connection.connect(device)

