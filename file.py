from flask import Flask, jsonify
from pymongo import MongoClient
from datetime import datetime, timedelta

app = Flask(__name__)

client = MongoClient("mongodb://admin:cisco123@13.234.241.103:27017/?authSource=iotdb&readPreference=primary&ssl=false")
db = client['iotdb']
overall_data = db["overall_data"]  # Replace with your collection name

@app.route("/api/orgchart", methods=["GET"])
def get_orgchart_data():
    try:
        # Fetch distinct plants
        plants = overall_data.distinct("dataItemMap.Plant")
        
        # Initialize the hierarchy structure
        org_chart = []

        # Calculate yesterday's date
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

        for plant in plants:
            # Initialize plant-level data
            plant_data = {
                "name": plant,
                "title": "Plant",
                "image": "../../public/assets/images/plant.png",  # Replace with your plant image path
                "children": [],
                "p_abd_sum": 0  # Initialize cumulative P_abd
            }

            # Get all serial numbers (sn) associated with the plant
            sn_list = overall_data.distinct("dataItemMap.sn", {"dataItemMap.Plant": plant})

            for sn in sn_list:
                # Initialize sn-level data
                sn_data = {
                    "name": sn,
                    "title": "Serial Number",
                    "image": "../../public/assets/images/inverter.png",  # Replace with your SN image path
                    "children": [],
                    "p_abd_sum": 0  # Initialize cumulative P_abd
                }

                # Get all MPPTs associated with the sn
                mppt_list = overall_data.distinct("dataItemMap.MPPT", {"dataItemMap.sn": sn})

                for mppt in mppt_list:
                    # Initialize MPPT-level data
                    mppt_data = {
                        "name": mppt,
                        "title": "MPPT",
                        "image": "../../public/assets/images/mppt.png",  # Replace with your MPPT image path
                        "children": [],
                        "p_abd_sum": 0  # Initialize cumulative P_abd
                    }

                    # Get all Strings associated with the MPPT
                    string_docs = overall_data.find(
                        {"dataItemMap.MPPT": mppt, "dataItemMap.sn": sn},
                        {"dataItemMap.Strings": 1, "dataItemMap.Watt/String": 1, "timestamp": 1, "dataItemMap.P_abd": 1}
                    )

                    # Process each string
                    strings_data = {}
                    for doc in string_docs:
                        string_name = doc["dataItemMap"]["Strings"]
                        watt_string = doc["dataItemMap"]["Watt/String"]
                        timestamp = doc["timestamp"]
                        p_abd = doc["dataItemMap"].get("P_abd", 0)

                        # Add string data or update the sum of P_abd
                        if string_name not in strings_data:
                            strings_data[string_name] = {
                                "name": string_name,
                                "title": "String",
                                "image": "../../public/assets/images/solar-mon.png",  # Replace with your string image path
                                "watt_string": watt_string,
                                "p_abd_sum": 0
                            }

                        # Only sum P_abd for yesterday
                        if timestamp == yesterday:
                            strings_data[string_name]["p_abd_sum"] += p_abd

                    # Append each string to MPPT's children and add its P_abd to MPPT's total
                    for string in strings_data.values():
                        mppt_data["children"].append(string)
                        mppt_data["p_abd_sum"] += string["p_abd_sum"]

                    # Append MPPT data to sn's children and add its P_abd to SN's total
                    sn_data["children"].append(mppt_data)
                    sn_data["p_abd_sum"] += mppt_data["p_abd_sum"]

                # Append sn data to plant's children and add its P_abd to Plant's total
                plant_data["children"].append(sn_data)
                plant_data["p_abd_sum"] += sn_data["p_abd_sum"]

            # Append plant data to the hierarchy
            org_chart.append(plant_data)

        return jsonify({"status": "success", "data": org_chart})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
