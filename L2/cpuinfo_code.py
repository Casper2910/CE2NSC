import psutil
from L2.cpuinfo_code import get_cpu_info

def bytes_to_mb(x):
    return f"{x / (1024 ** 2):.1f} MB"

def bytes_to_gb(x):
    return f"{x / (1024 ** 3):.2f} GB"

cpu = get_cpu_info()

# CPU cores
physical_cores = psutil.cpu_count(logical=False)
logical_cores = psutil.cpu_count(logical=True)

# RAM
ram = psutil.virtual_memory().total

# Cache info (may be missing on some CPUs/OSes)
l1 = cpu.get("l1_data_cache_size") or cpu.get("l1_cache_size")
l2 = cpu.get("l2_cache_size")
l3 = cpu.get("l3_cache_size")

print("CPU:")
print(f"  Brand           : {cpu.get('brand_raw', 'unknown')}")
print(f"  Physical cores  : {physical_cores}")
print(f"  Logical cores   : {logical_cores}")

print("\nMemory:")
print(f"  RAM             : {bytes_to_gb(ram)}")

print("\nCache:")
print(f"  L1 cache        : {bytes_to_mb(l1) if l1 else 'unknown'}")
print(f"  L2 cache        : {bytes_to_mb(l2) if l2 else 'unknown'}")
print(f"  L3 cache        : {bytes_to_mb(l3) if l3 else 'unknown'}")


