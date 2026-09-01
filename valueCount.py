import os
from collections import Counter, defaultdict

LABELS_DIR = r"C:\Urbanity-data\PhoneCall\labels"

class_counter = Counter()
image_counter = defaultdict(set)

total_files = 0
empty_files = 0

for file_name in os.listdir(LABELS_DIR):
    if not file_name.endswith(".txt"):
        continue

    total_files += 1
    file_path = os.path.join(LABELS_DIR, file_name)

    with open(file_path, "r") as f:
        lines = f.readlines()

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            class_counter[class_id] += 1
            image_counter[class_id].add(file_name)

class_names = {
    0: 'Rider', 
    1: 'car', 
    2: 'no_helmet', 
    3: 'helmet', 
    4: 'number_plate', 
    5: 'pillion', 
    6: 'bike_with_rider',
    7:'callling_posture'
}

print("\n Class Statistics")
print("--------------------------------------------")
print("Class Name        | Objects | Images")
print("--------------------------------------------")

for cls in sorted(class_counter):
    print(
        f"{class_names.get(cls,'unknown'):16} | "
        f"{class_counter[cls]:7} | "
        f"{len(image_counter[cls]):6}"
    )

print("\n Dataset Summary")

print("--------------------------------------------")
print(f"Total label files : {total_files}")
print(f"Empty label files : {empty_files}")
print(f"Total objects     : {sum(class_counter.values())}")
