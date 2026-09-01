import os

LABELS_DIR = r"C:/Users/softlabs_group/Downloads/project-18-at-2026-02-23-17-41-9de2dcc9/labels"
REMOVE_CLASS_ID = 5  # rider_standby

removed_count = 0
affected_files = 0

for file_name in os.listdir(LABELS_DIR):
    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join(LABELS_DIR, file_name)

    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    file_changed = False

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        class_id = int(parts[0])

        if class_id == REMOVE_CLASS_ID:
            removed_count += 1
            file_changed = True
            continue  # skip this line

        new_lines.append(line)

    if file_changed:
        affected_files += 1
        with open(file_path, "w") as f:
            f.writelines(new_lines)

print("Cleanup completed")
print(f"Removed rider_standby objects : {removed_count}")
print(f"Affected label files          : {affected_files}")
