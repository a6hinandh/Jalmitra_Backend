from neo4j import GraphDatabase
import json

# --- Utility to clean None values ---
# --- Utility to clean None values safely ---
def clean_dict(d):
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items() if v is not None and not isinstance(v, (dict, list))}


# --- Main insert function ---
def insert_all(tx, data):
    for District in data:
        loc = District["locationName"]
        uuid = District.get("locationUUID") or "total"

        # Create (or update) District node with UUID
        tx.run("""
            MERGE (s:District {uuid:$uuid})
            SET s.name = $loc
        """, loc=loc, uuid=uuid)

        # Connect to Country
        tx.run("""
            MERGE (c:State {name:$state})
            WITH c
            MATCH (s:District {uuid:$uuid})
            MERGE (c)-[:HAS_District]->(s)
        """, state="KERALA", uuid=uuid)

        # Areas
        for area_type, v in District["area"].items():
            tx.run("""
                MATCH (l:District {uuid:$uuid})
                MERGE (a:Area {type:$type,uuid:$uuid})
                SET a += $vals
                MERGE (l)-[:HAS_AREA]->(a)
            """, type=area_type, loc=loc, uuid=uuid, vals=clean_dict(v))

        # Loss
        tx.run("""
            MATCH (l:District {uuid:$uuid})
            MERGE (loss:Loss {District:$loc})
            SET loss += $loss
            MERGE (l)-[:HAS_LOSS]->(loss)
        """, loc=loc, uuid=uuid, loss=clean_dict(District["loss"]))

        # Block Summary (make key unique per District)
        for k, v in District["reportSummary"].items():
            if k != "total": 
                continue
            block = clean_dict(v.get("BLOCK", {}))
            tx.run("""
                MATCH (l:District {uuid:$uuid})
                MERGE (b:BlockSummary {uuid:$uuid+"_"+$bid})
                SET b += $block
                MERGE (l)-[:HAS_BLOCK_SUMMARY]->(b)
            """, bid=k, loc=loc, uuid=uuid, block=block)

        # Recharge Data
        recharge = {k: v.get("total") for k, v in District["rechargeData"].items() if isinstance(v, dict)}
        tx.run("""
            MATCH (l:District {uuid:$uuid})
            MERGE (r:Recharge {District:$loc})
            SET r += $recharge
            MERGE (l)-[:HAS_RECHARGE]->(r)
        """, loc=loc, uuid=uuid, recharge=clean_dict(recharge))

        # Draft Data
        draft = {k: v.get("total") for k, v in District["draftData"].items() if isinstance(v, dict)}
        tx.run("""
            MATCH (l:District {uuid:$uuid})
            MERGE (d:Draft {District:$loc})
            SET d += $draft
            MERGE (l)-[:HAS_DRAFT]->(d)
        """, loc=loc, uuid=uuid, draft=clean_dict(draft))

        # Allocation
        alloc = {k: (v.get("total") if isinstance(v, dict) else v) for k, v in District["gwallocation"].items()}
        tx.run("""
            MATCH (l:District {uuid:$uuid})
            MERGE (a:Allocation {District:$loc})
            SET a += $alloc
            MERGE (l)-[:HAS_ALLOCATION]->(a)
        """, loc=loc, uuid=uuid, alloc=clean_dict(alloc))

        # Availability
        tx.run("""
            MATCH (l:District {uuid:$uuid})
            MERGE (av:Availability {District:$loc})
            SET av += $availability
            MERGE (l)-[:HAS_AVAILABILITY]->(av)
        """, loc=loc, uuid=uuid, availability=clean_dict(District["totalGWAvailability"]))

        # Stage of Extraction
        tx.run("""
            MATCH (l:District {uuid:$uuid})
            MERGE (s:StageOfExtraction {District:$loc})
            SET s += $stage
            MERGE (l)-[:HAS_STAGE]->(s)
        """, loc=loc, uuid=uuid, stage=clean_dict(District["stageOfExtraction"]))

        # Rainfall
        tx.run("""
            MATCH (l:District {uuid:$uuid})
            MERGE (rf:Rainfall {District:$loc})
            SET rf += $rainfall
            MERGE (l)-[:HAS_RAINFALL]->(rf)
        """, loc=loc, uuid=uuid, rainfall=clean_dict(District["rainfall"]))

        #Ground Water Availability
        tx.run("""
            MATCH (l:District {uuid:$uuid})
            MERGE (gw:GroundWaterAvailability {District:$loc})
            SET gw += $groundwater
            MERGE (l)-[:HAS_GROUND_WATER]->(gw)
        """, loc=loc, uuid=uuid, groundwater=clean_dict(District["totalGWAvailability"]))

        #Future Use
        tx.run("""
            MATCH (l:District {uuid:$uuid})
            MERGE (f:FutureUse {District:$loc})
            SET f += $future
            MERGE (l)-[:HAS_FUTURE_USE]->(f)
        """, loc=loc, uuid=uuid, future=clean_dict(District["availabilityForFutureUse"]))
        

        # Additional Recharge
        add_recharge = {k: v.get("total") for k, v in District["additionalRecharge"].items() if isinstance(v, dict)}
        tx.run("""
            MATCH (l:District {uuid:$uuid})
            MERGE (ar:AdditionalRecharge {District:$loc})
            SET ar += $ar
            MERGE (l)-[:HAS_ADDITIONAL_RECHARGE]->(ar)
        """, loc=loc, uuid=uuid, ar=clean_dict(add_recharge))

        # Aquifer Business Data (loop all aquifers dynamically)
        for aq_type, aq_vals in District["aquiferBusinessData"].items():
            if aq_vals and isinstance(aq_vals, dict):
                aq_props = {}
                for k, v in aq_vals.items():
                    if isinstance(v, dict):
                        aq_props[k] = v.get(k) or v.get("total")  # get actual number
                    else:
                        aq_props[k] = v
                tx.run("""
                    MATCH (l:District {uuid:$uuid})
                    MERGE (aq:Aquifer {type:$type, District:$loc})
                    SET aq += $aq
                    MERGE (l)-[:HAS_AQUIFER]->(aq)
                """, type=aq_type, loc=loc, uuid=uuid, aq=clean_dict(aq_props))


# --- Run the insert ---
if __name__ == "__main__":
    with open("data/states/KERALA.json", "r") as f:
        data = json.load(f)

    driver = GraphDatabase.driver(
        "neo4j+s://03882640.databases.neo4j.io", 
        auth=("03882640", "jacLTqC7FH6l_qgbf6gQJp7wfXBq3b39EqV6KMPhiT4")
    )

    with driver.session() as session:
        session.execute_write(insert_all, data)

    driver.close()
    print("✅ Data stored in Neo4j successfully")
