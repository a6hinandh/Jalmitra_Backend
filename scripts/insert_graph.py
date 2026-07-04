from neo4j import GraphDatabase
import json

from dotenv import load_dotenv
import os

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# --- Utility to clean None values ---
# --- Utility to clean None values safely ---
def clean_dict(d):
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items() if v is not None and not isinstance(v, (dict, list))}


# --- Main insert function ---
def insert_all(tx, data):
    for state in data:
        loc = state["locationName"]
        uuid = state.get("locationUUID") or "total"

        # Create (or update) State node with UUID
        tx.run("""
            MERGE (s:State {uuid:$uuid})
            SET s.name = $loc
        """, loc=loc, uuid=uuid)

        # Connect to Country
        tx.run("""
            MERGE (c:Country {name:$country})
            WITH c
            MATCH (s:State {uuid:$uuid})
            MERGE (c)-[:HAS_STATE]->(s)
        """, country="India", uuid=uuid)

        # Areas
        for area_type, v in state["area"].items():
            tx.run("""
                MATCH (l:State {uuid:$uuid})
                MERGE (a:Area {type:$type,uuid:$uuid})
                SET a += $vals
                MERGE (l)-[:HAS_AREA]->(a)
            """, type=area_type, loc=loc, uuid=uuid, vals=clean_dict(v))

        # Loss
        tx.run("""
            MATCH (l:State {uuid:$uuid})
            MERGE (loss:Loss {State:$loc})
            SET loss += $loss
            MERGE (l)-[:HAS_LOSS]->(loss)
        """, loc=loc, uuid=uuid, loss=clean_dict(state["loss"]))

        # Block Summary (make key unique per State)
        for k, v in state["reportSummary"].items():
            if k != "total":
                continue
            block = clean_dict(v.get("BLOCK", {}))
            tx.run("""
                MATCH (l:State {uuid:$uuid})
                MERGE (b:BlockSummary {uuid:$uuid+"_"+$bid})
                SET b += $block
                MERGE (l)-[:HAS_BLOCK_SUMMARY]->(b)
            """, bid=k, loc=loc, uuid=uuid, block=block)

        # Recharge Data
        recharge = {k: v.get("total") for k, v in state["rechargeData"].items() if isinstance(v, dict)}
        tx.run("""
            MATCH (l:State {uuid:$uuid})
            MERGE (r:Recharge {State:$loc})
            SET r += $recharge
            MERGE (l)-[:HAS_RECHARGE]->(r)
        """, loc=loc, uuid=uuid, recharge=clean_dict(recharge))

        # Draft Data
        draft = {k: v.get("total") for k, v in state["draftData"].items() if isinstance(v, dict)}
        tx.run("""
            MATCH (l:State {uuid:$uuid})
            MERGE (d:Draft {State:$loc})
            SET d += $draft
            MERGE (l)-[:HAS_DRAFT]->(d)
        """, loc=loc, uuid=uuid, draft=clean_dict(draft))

        # Allocation
        alloc = {k: (v.get("total") if isinstance(v, dict) else v) for k, v in state["gwallocation"].items()}
        tx.run("""
            MATCH (l:State {uuid:$uuid})
            MERGE (a:Allocation {State:$loc})
            SET a += $alloc
            MERGE (l)-[:HAS_ALLOCATION]->(a)
        """, loc=loc, uuid=uuid, alloc=clean_dict(alloc))

        # Availability
        tx.run("""
            MATCH (l:State {uuid:$uuid})
            MERGE (av:Availability {State:$loc})
            SET av += $availability
            MERGE (l)-[:HAS_AVAILABILITY]->(av)
        """, loc=loc, uuid=uuid, availability=clean_dict(state["totalGWAvailability"]))

        # Stage of Extraction
        tx.run("""
            MATCH (l:State {uuid:$uuid})
            MERGE (s:StageOfExtraction {State:$loc})
            SET s += $stage
            MERGE (l)-[:HAS_STAGE]->(s)
        """, loc=loc, uuid=uuid, stage=clean_dict(state["stageOfExtraction"]))

        # Rainfall
        tx.run("""
            MATCH (l:State {uuid:$uuid})
            MERGE (rf:Rainfall {State:$loc})
            SET rf += $rainfall
            MERGE (l)-[:HAS_RAINFALL]->(rf)
        """, loc=loc, uuid=uuid, rainfall=clean_dict(state["rainfall"]))

        #Ground Water Availability
        tx.run("""
            MATCH (l:State {uuid:$uuid})
            MERGE (gw:GroundWaterAvailability {State:$loc})
            SET gw += $groundwater
            MERGE (l)-[:HAS_GROUND_WATER]->(gw)
        """, loc=loc, uuid=uuid, groundwater=clean_dict(state["totalGWAvailability"]))

        #Future Use
        tx.run("""
            MATCH (l:State {uuid:$uuid})
            MERGE (f:FutureUse {State:$loc})
            SET f += $future
            MERGE (l)-[:HAS_FUTURE_USE]->(f)
        """, loc=loc, uuid=uuid, future=clean_dict(state["availabilityForFutureUse"]))


        # Additional Recharge
        add_recharge = {k: v.get("total") for k, v in state["additionalRecharge"].items() if isinstance(v, dict)}
        tx.run("""
            MATCH (l:State {uuid:$uuid})
            MERGE (ar:AdditionalRecharge {State:$loc})
            SET ar += $ar
            MERGE (l)-[:HAS_ADDITIONAL_RECHARGE]->(ar)
        """, loc=loc, uuid=uuid, ar=clean_dict(add_recharge))

        # Aquifer Business Data (loop all aquifers dynamically)
        for aq_type, aq_vals in state["aquiferBusinessData"].items():
            if aq_vals and isinstance(aq_vals, dict):
                aq_props = {}
                for k, v in aq_vals.items():
                    if isinstance(v, dict):
                        aq_props[k] = v.get(k) or v.get("total")  # get actual number
                    else:
                        aq_props[k] = v
                tx.run("""
                    MATCH (l:State {uuid:$uuid})
                    MERGE (aq:Aquifer {type:$type, State:$loc})
                    SET aq += $aq
                    MERGE (l)-[:HAS_AQUIFER]->(aq)
                """, type=aq_type, loc=loc, uuid=uuid, aq=clean_dict(aq_props))


# --- Run the insert ---
if __name__ == "__main__":
    with open("data/output/india.json", "r") as f:
        data = json.load(f)

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))
    )

    with driver.session() as session:
        session.execute_write(insert_all, data)

    driver.close()
    print("✅ Data stored in Neo4j successfully")
