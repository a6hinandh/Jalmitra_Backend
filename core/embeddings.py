# embeddings.py

import os
import json
from sentence_transformers import SentenceTransformer

def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    ids = []
    metadatas=[]

    for entry in data:
        # Since there's no loctype field, we'll process all entries as states
        # They all seem to be state-level data based on the locationName values

        location_name = entry.get('locationName', 'Unknown')

        # Extract nested data safely
        area_data = entry.get('area', {})
        total_area = None
        recharge_area = None
        recharge_command_area = None
        recharge_non_command_area = None
        hilly_area = None
        non_recharge_area = None
        if isinstance(area_data, dict) and 'total' in area_data:
            if isinstance(area_data['total'], dict):
                total_area = area_data['total'].get('totalArea')
                hilly_area = area_data['total'].get('hillyArea')

            else:
                total_area = area_data.get('total')
            if isinstance(area_data['recharge_worthy'],dict):
                recharge_area = area_data['recharge_worthy'].get('totalArea')
                recharge_command_area = area_data['recharge_worthy'].get('commandArea')
                recharge_non_command_area = area_data['recharge_worthy'].get('nonCommandArea')
            if isinstance(area_data['non_recharge_worthy'],dict):
                non_recharge_area = area_data['non_recharge_worthy'].get('totalArea')

        total_loss = None
        loss_data = entry.get("loss",{})
        if isinstance(loss_data, dict):
            total_loss = loss_data.get("total")

        # future_use = None
        # future_use_data = entry.get("gwProjectedUtilAllocationDynamicAquifer",{})
        # if isinstance(future_use_data, dict):
        #     future_use = future_use_data["total"].get("total")

        avaialable_groundwater = None
        avaialable_groundwater_command = None
        avaialable_groundwater_non_command = None
        groundwater_data = entry.get("totalGWAvailability",{})
        if isinstance(groundwater_data, dict):
            avaialable_groundwater = groundwater_data.get("total")
            avaialable_groundwater_command = groundwater_data.get("non_command")
            avaialable_groundwater_non_command = groundwater_data.get("command")

        rainfall_data = entry.get('rainfall', {})
        rainfall_total = None
        if isinstance(rainfall_data, dict):
            rainfall_total = rainfall_data.get('total')

        recharge_data = entry.get('rechargeData', {})
        recharge_total = None
        rainfall_recharge_total = None
        streamChannel_recharge_total = None
        canal_recharge_total = None
        surfaceWater_recharge_total = None
        groundWater_recharge_total = None
        waterConversationStructures_recharge_total = None
        tanksAndPonds_recharge_total = None

        if isinstance(recharge_data, dict) and 'total' in recharge_data:
            if isinstance(recharge_data['total'], dict):
                recharge_total = recharge_data.get('total',"")
                rainfall_recharge_total = recharge_data.get("rainfall","")
                streamChannel_recharge_total = recharge_data.get("agriculture","")
                canal_recharge_total = recharge_data.get("canal","")
                surfaceWater_recharge_total=recharge_data.get("surface_irrigation","")
                groundWater_recharge_total=recharge_data.get("gw_irrigation","")
                waterConversationStructures_recharge_total=recharge_data.get("artificial_structure","")
                tanksAndPonds_recharge_total=recharge_data.get("water_body","")
            else:
                recharge_total = recharge_data.get('total')

        draft_data = entry.get('draftData', {})
        draft_total = None
        if isinstance(draft_data, dict) and 'total' in draft_data:
            if isinstance(draft_data['total'], dict):
                draft_total = draft_data['total'].get('total')
            else:
                draft_total = draft_data.get('total')

        future_data = entry.get('availabilityForFutureUse', {})
        future_total = None
        if isinstance(future_data, dict) and 'total' in draft_data:
            if isinstance(future_data['total'], dict):
                future_total = future_data['total'].get('total')
            else:
                future_total = future_data.get('total')

        stage_extraction = entry.get('stageOfExtraction', {})
        stage_total = None
        if isinstance(stage_extraction, dict):
            stage_total = stage_extraction.get('total')

        category = entry.get('category', 'Unknown')


        metadata = {
            "State": location_name,
            "Total area" : total_area,
            "Rainfall" : rainfall_total,
            "Total ground water available" : avaialable_groundwater,
            "Total extractable ground water" : draft_total,
            "Available Groundwater for future use" : future_total,
            "Stage of ground water extraction" : stage_total

        }

        # Create text representation
        text = f"""
        Ground water report for {location_name}:
        Total area: {total_area}
        Rainfall: {rainfall_total}
        Total Recharge worthy area: {recharge_area}
        Total Recharge worthy command area: {recharge_command_area}
        Total Recharge worthy non command area: {recharge_non_command_area}
        Total Non recharge worthy area: {non_recharge_area}
        Total Hilly area : {hilly_area}
        Total loss in ground water : {total_loss}
        Total available ground water : {avaialable_groundwater}
        Total available ground water in command area : {avaialable_groundwater_command}
        Total available ground water in non command area : {avaialable_groundwater_non_command}
        Ground water recharge data : {
            f"""Total : {recharge_total},
            "Rainfall" : {rainfall_recharge_total},
            "Stream channel" : {streamChannel_recharge_total},
            "Canal" : {canal_recharge_total},
            "Surface water irrigation": {surfaceWater_recharge_total},
            "Ground water irrigation" : {groundWater_recharge_total},
            "Water Conversation Structures": {waterConversationStructures_recharge_total},
            "Tanks and Ponds" : {tanksAndPonds_recharge_total},
            """
        }
        Total extractable ground water : {draft_total}
        Ground water extraction data : {draft_data}
        Available Groundwater for future use : {future_total}
        Stage of ground water extraction : {stage_total}
        Category: {category}
        """

        texts.append(text)
        metadatas.append(metadata)
        # Use locationUUID as ID, or fallback to a generated ID
        entry_id = entry.get('locationUUID')
        if entry_id is None or entry_id == "":
            # Generate a safe ID using the location name
            safe_name = location_name.replace(" ", "_").replace("-", "_").lower()
            entry_id = f"district_{safe_name}_{len(ids)}"
        ids.append(str(entry_id))  # Ensure it's always a string


    return ids, texts, metadatas

def create_embeddings(texts):
    # Must match EMBED_MODEL used at query time in core/graphrag.py, otherwise the
    # vector dimensions won't match the Pinecone index.
    model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

if __name__ == "__main__":
    json_file = "data/output/india.json"
    ids, texts, metadatas = load_data(json_file)

    # Print first few texts to verify
    print("📝 Sample processed texts:")
    for i, text in enumerate(texts[:3]):
        print(f"  {i+1}. {text}")

    embeddings = create_embeddings(texts)
    print(f"✅ Created embeddings for {len(texts)} states.")
