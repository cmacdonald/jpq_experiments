import os
import csv
import argparse
import sys

def record_files_to_csv(folder_path, csv_name="file_inventory.csv"):
    if not os.path.isdir(folder_path):
        print(f"Skipping (not a directory): {folder_path}", file=sys.stderr)
        return

    csv_path = os.path.join(folder_path, csv_name)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "size_bytes"])

        for entry in os.scandir(folder_path):
            if entry.is_file():
                writer.writerow([entry.name, entry.stat().st_size])

    print(f"Created {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Record filenames and sizes (non-recursive) into a CSV per directory"
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more directories to scan"
    )
    parser.add_argument(
        "--csv-name",
        default="file_inventory.csv",
        help="Name of CSV file to create in each directory"
    )

    args = parser.parse_args()

    for folder in args.directories:
        record_files_to_csv(folder, args.csv_name)


if __name__ == "__main__":
    main()

