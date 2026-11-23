"""Join the published film metadata to the original SerenLens movie CSV."""

import argparse
import csv
from pathlib import Path


FIELDS = ("item_id", "corrected_title", "description")


def load_metadata(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"Metadata CSV must have columns {', '.join(FIELDS)}.")
        metadata = {}
        for row in reader:
            item_id = row["item_id"]
            if not item_id or item_id in metadata:
                raise ValueError(f"Missing or duplicate metadata item_id on line {reader.line_num}.")
            metadata[item_id] = (row["corrected_title"], row["description"])
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="Original SerenLens_Movies.csv")
    parser.add_argument("metadata", type=Path, help="Downloaded serenlens_movie_metadata.csv")
    parser.add_argument("output", type=Path, help="Path for the merged CSV")
    args = parser.parse_args()

    csv.field_size_limit(16 * 1024 * 1024)
    try:
        metadata = load_metadata(args.metadata)
        with args.original.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames or []
            if "item_id" not in fields or any(field in fields for field in FIELDS[1:]):
                raise ValueError("Use the original SerenLens CSV with item_id and no metadata columns.")
            rows = matched = 0
            with args.output.open("x", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination)
                writer.writerow([*fields, *FIELDS[1:]])
                for row in reader:
                    values = metadata.get(row["item_id"])
                    matched += values is not None
                    rows += 1
                    writer.writerow([*(row[field] for field in fields), *(values or ("", ""))])
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"{error}\n")
    print(f"Joined metadata for {matched:,} of {rows:,} source rows. Output: {args.output}")


if __name__ == "__main__":
    main()
