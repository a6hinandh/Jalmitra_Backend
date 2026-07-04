# check_uuids.py

import json

def check_uuids(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔍 Checking UUIDs in {json_file}")
    print(f"📊 Total entries: {len(data)}")

    missing_uuids = []
    valid_uuids = []

    for i, entry in enumerate(data):
        location_name = entry.get('locationName', f'Entry_{i}')
        uuid = entry.get('locationUUID')

        if uuid is None or uuid == "":
            missing_uuids.append(f"  {i+1}. {location_name}: UUID is {uuid}")
        else:
            valid_uuids.append(f"  {i+1}. {location_name}: {uuid}")

    print(f"\n✅ Entries with valid UUIDs: {len(valid_uuids)}")
    if len(valid_uuids) <= 5:  # Show all if few
        for entry in valid_uuids:
            print(entry)
    else:  # Show first few
        for entry in valid_uuids[:3]:
            print(entry)
        print(f"  ... and {len(valid_uuids) - 3} more")

    print(f"\n❌ Entries with missing UUIDs: {len(missing_uuids)}")
    for entry in missing_uuids:
        print(entry)

if __name__ == "__main__":
    check_uuids("data/states/KERALA.json")
