import cupy as cp

# Get the number of available devices
device_count = cp.cuda.runtime.getDeviceCount()
print(f"Total number of CUDA devices available: {device_count}")

# Print properties of all devices
for i in range(device_count):
    props = cp.cuda.runtime.getDeviceProperties(i)
    print(f"Device ID {i}:")
    print(f"  Name: {props['name'].decode()}") # Name of the GPU
    cc = None
    if 'compute_capability' in props:
        try:
            cc = props['compute_capability'].decode()
        except Exception:
            cc = str(props['compute_capability'])
    elif 'major' in props and 'minor' in props:
        cc = f"{props['major']}.{props['minor']}"
    else:
        cc = 'unknown'
    print(f"  Compute Capability: {cc}")


# create arrays on GPU
x = cp.array([1, 2, 3, 4])
y = cp.array([10, 20, 30, 40])

# elementwise operations
z = x + y
print(z)