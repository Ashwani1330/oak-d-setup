import depthai as dai

# List all devices (unbooted or bootloader)
infos = dai.DeviceBootloader.getAllAvailableDevices()
print("Found devices:", infos)

if not infos:
    raise Exception("No OAK-D cameras detected!")

# Pick the first
deviceInfo = infos[0]
print("Using device:", deviceInfo.name, "state:", deviceInfo.state)
