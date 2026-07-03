# debug_data.py

import json

def debug_json_structure(json_file):
    print(f"🔍 Debugging {json_file}")
    
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"✅ File loaded successfully")
        print(f"📊 Total entries in file: {len(data)}")
        
        # Check the structure of first few entries
        print(f"\n🔬 First 3 entries structure:")
        for i, entry in enumerate(data[:3]):
            print(f"\nEntry {i+1}:")
            print(f"  Keys: {list(entry.keys())}")
            if 'loctype' in entry:
                print(f"  loctype: {entry['loctype']}")
            if 'locationName' in entry:
                print(f"  locationName: {entry['locationName']}")
        
        # Count different loctypes
        loctype_counts = {}
        for entry in data:
            loctype = entry.get('loctype', 'Unknown')
            loctype_counts[loctype] = loctype_counts.get(loctype, 0) + 1
        
        print(f"\n📈 Loctype distribution:")
        for loctype, count in loctype_counts.items():
            print(f"  {loctype}: {count}")
        
        # Show STATE entries specifically
        state_entries = [entry for entry in data if entry.get('loctype') == 'STATE']
        print(f"\n🏛️ Found {len(state_entries)} STATE entries")
        
        if state_entries:
            print(f"First STATE entry example:")
            state = state_entries[0]
            for key, value in state.items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_json_structure("data/states/KERALA.json")