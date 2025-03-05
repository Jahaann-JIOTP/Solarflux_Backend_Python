from flask import Flask, request, jsonify, Response
from pymongo import MongoClient
from datetime import datetime, timedelta
from flask_cors import CORS
from collections import defaultdict
import pyodbc
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from dtaidistance import dtw
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import optuna
app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/": {"origins": ""}})
server = 'DESKTOP-FMELRIN\\ION'
database = 'JAHAANN_DB'
driver = '{ODBC Driver 17 for SQL Server}'
model = joblib.load('Best_RF_Model_CocaCola.pkl')
# client = MongoClient("mongodb://localhost:27017/")
client = MongoClient("mongodb://admin:cisco123@13.234.241.103:27017/?authSource=iotdb&readPreference=primary&ssl=false")
db = client['iotdb']
collection2 = db['final_format']
db = client['iotdb']
collection = db['hourly_plant']
device_data_collection = db['device_data']
device_hour_collection = db['String_Hour']
device_day_collection = db['String_Day']
String_Five_Minute = db['String_Five_Minute']
overall_data = db['overall_data']
GM_Hourly = db['GM_Hourly']
GM_Day = db['GM_Day']
GT_Hourly = db['GT_Hour']
Plant_Day = db['Plant_Day']

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the SolarFlux Backend API!"
    })
@app.route("/api/orgchart", methods=["GET"])
def get_orgchart_data():
    try:
        # Fetch distinct plants
        plants = overall_data.distinct("dataItemMap.Plant")
        
        # Initialize the hierarchy structure
        org_chart = []

        # Calculate the date 40 days ago
        target_date = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")

        # Aggregate query to get average temperature by sn and Day_Hour
        temperature_avg_pipeline = [
            {
                "$match": {
                    "Day_Hour": {"$regex": f"^{target_date}"}  # Match documents where Day_Hour starts with the target date
                }
            },
            {
                "$group": {
                    "_id": {"sn": "$sn", "Day_Hour": {"$substr": ["$Day_Hour", 0, 10]}},  # Extract the date part
                    "average_temp": {"$avg": {"$round": ["$temperature", 2]}}  # Round to 2 decimal places
                }
            }
        ]
        avg_temperatures = list(GT_Hourly.aggregate(temperature_avg_pipeline))
        
        # Create a dictionary for avg temperatures to use during data aggregation
        avg_temp_dict = {f"{item['_id']['sn']}_{item['_id']['Day_Hour']}": item['average_temp'] for item in avg_temperatures}

        for plant in plants:
            # Initialize plant-level data
            plant_data = {
                "name": plant,
                "title": f"Power: 0 KW",  # Initialize title with P_abd
                "image": "/assets/images/plant.png",  # Replace with your plant image path
                "children": [],
                "p_abd_sum": 0  # Initialize cumulative P_abd
            }

            # Get all serial numbers (sn) associated with the plant
            sn_list = overall_data.distinct("dataItemMap.sn", {"dataItemMap.Plant": plant})

            for sn in sn_list:
                # Initialize sn-level data
                sn_data = {
                    "name": sn,
                    "title": f"Power: 0 KW",  # Initialize title with P_abd
                    "image": "/assets/images/inv.png",  # Replace with your SN image path
                    "children": [],
                    "p_abd_sum": 0,  # Initialize cumulative P_abd
                    "avg_temp": avg_temp_dict.get(f"{sn}_{target_date}", 0)  # Fetch average temperature
                }
                # Calculate total Watt/String using the function
                watt_string_info = sum_watt_string(sn)
                sn_data['inverter'] = watt_string_info["total_watt_string"]/1000
                print('uuu', watt_string_info["total_watt_string"])
                # Add total_watt_string to title
                total_watt_string = watt_string_info["total_watt_string"]
                sn_data["title"] = f"Power: {sn_data['p_abd_sum']} KW<br> Avg Temp: {sn_data['avg_temp']}°C"

                # Get all MPPTs associated with the sn
                mppt_list = overall_data.distinct("dataItemMap.MPPT", {"dataItemMap.sn": sn})

                for mppt in mppt_list:
                    # Initialize MPPT-level data
                    mppt_data = {
                        "name": mppt,
                        "title": f"Power: 0 KW",  # Initialize title with P_abd
                        "image": "/assets/images/mppt.png",  # Replace with your MPPT image path
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
                                "title": f"Power: 0 KW",  # Initialize title with P_abd
                                "image": "/assets/images/solar-mon.png",  # Replace with your string image path
                                "watt_string": watt_string,
                                "p_abd_sum": 0
                            }

                        # Only sum P_abd for the target date
                        if timestamp == target_date:
                            strings_data[string_name]["p_abd_sum"] += p_abd

                    # Append each string to MPPT's children and add its rounded P_abd to MPPT's total
                    for string in strings_data.values():
                        string["p_abd_sum"] = round(string["p_abd_sum"], 2)  # Round the P_abd_sum
                        
                        string["title"] = f"Power: {string['p_abd_sum']} KW<br> Capacity: {string['watt_string']} W"   # Update title with rounded P_abd
                        mppt_data["children"].append(string)
                        mppt_data["p_abd_sum"] += string["p_abd_sum"]

                    # Round and update MPPT title with P_abd
                    mppt_data["p_abd_sum"] = round(mppt_data["p_abd_sum"], 2)
                    mppt_data["title"] = f"Power: {mppt_data['p_abd_sum']} KW"
                    # Append MPPT data to sn's children and add its P_abd to SN's total
                    sn_data["children"].append(mppt_data)
                    sn_data["p_abd_sum"] += mppt_data["p_abd_sum"]

                # Round and update SN title with P_abd and avg_temp
                sn_data["p_abd_sum"] = round(sn_data["p_abd_sum"], 2)
                sn_data["avg_temp"] = round(sn_data["avg_temp"], 2)
                sn_data["title"] = f"Power: {sn_data['p_abd_sum']} KW<br> Avg Temp: {sn_data['avg_temp']}°C"

                # Append sn data to plant's children and add its P_abd to Plant's total
                plant_data["children"].append(sn_data)
                plant_data["p_abd_sum"] += sn_data["p_abd_sum"]

            # Round and update Plant title with P_abd
            plant_data["p_abd_sum"] = round(plant_data["p_abd_sum"], 2)
            plant_data["title"] = f"Power: {plant_data['p_abd_sum']} KW"
            # Append plant data to the hierarchy
            org_chart.append(plant_data)

        return jsonify({"status": "success", "data": org_chart})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# @app.route("/api/orgchart", methods=["GET"])
# def get_orgchart_data():
#     try:
#         # Fetch distinct plants
#         plants = overall_data.distinct("dataItemMap.Plant")
        
#         # Initialize the hierarchy structure
#         org_chart = []

#         # Calculate the date 40 days ago
#         target_date = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")

#         for plant in plants:
#             # Initialize plant-level data
#             plant_data = {
#                 "name": plant,
#                 "title": f"Power: 0 KW",  # Initialize title with P_abd
#                 "image": "/assets/images/plant.png",  # Replace with your plant image path
#                 "children": [],
#                 "p_abd_sum": 0  # Initialize cumulative P_abd
#             }

#             # Get all serial numbers (sn) associated with the plant
#             sn_list = overall_data.distinct("dataItemMap.sn", {"dataItemMap.Plant": plant})

#             for sn in sn_list:
#                 # Initialize sn-level data
#                 sn_data = {
#                     "name": sn,
#                     "title": f"Power: 0 KW",  # Initialize title with P_abd
#                     "image": "/assets/images/inv.png",  # Replace with your SN image path
#                     "children": [],
#                     "p_abd_sum": 0  # Initialize cumulative P_abd
#                 }

#                 # Get all MPPTs associated with the sn
#                 mppt_list = overall_data.distinct("dataItemMap.MPPT", {"dataItemMap.sn": sn})

#                 for mppt in mppt_list:
#                     # Initialize MPPT-level data
#                     mppt_data = {
#                         "name": mppt,
#                         "title": f"Power: 0 KW",  # Initialize title with P_abd
#                         "image": "/assets/images/mppt.png",  # Replace with your MPPT image path
#                         "children": [],
#                         "p_abd_sum": 0  # Initialize cumulative P_abd
#                     }

#                     # Get all Strings associated with the MPPT
#                     string_docs = overall_data.find(
#                         {"dataItemMap.MPPT": mppt, "dataItemMap.sn": sn},
#                         {"dataItemMap.Strings": 1, "dataItemMap.Watt/String": 1, "timestamp": 1, "dataItemMap.P_abd": 1}
#                     )

#                     # Process each string
#                     strings_data = {}
#                     for doc in string_docs:
#                         string_name = doc["dataItemMap"]["Strings"]
#                         watt_string = doc["dataItemMap"]["Watt/String"]
#                         timestamp = doc["timestamp"]
#                         p_abd = doc["dataItemMap"].get("P_abd", 0)

#                         # Add string data or update the sum of P_abd
#                         if string_name not in strings_data:
#                             strings_data[string_name] = {
#                                 "name": string_name,
#                                 "title": f"Power: 0 KW",  # Initialize title with P_abd
#                                 "image": "/assets/images/solar-mon.png",  # Replace with your string image path
#                                 "watt_string": watt_string,
#                                 "p_abd_sum": 0
#                             }

#                         # Only sum P_abd for the target date
#                         if timestamp == target_date:
#                             strings_data[string_name]["p_abd_sum"] += p_abd

#                     # Append each string to MPPT's children and add its rounded P_abd to MPPT's total
#                     for string in strings_data.values():
#                         string["p_abd_sum"] = round(string["p_abd_sum"], 2)  # Round the P_abd_sum
#                         string["title"] = f"Power: {string['p_abd_sum']} KW<br> Watt/String: {string['watt_string']}"   # Update title with rounded P_abd
#                         mppt_data["children"].append(string)
#                         mppt_data["p_abd_sum"] += string["p_abd_sum"]

#                     # Round and update MPPT title with P_abd
#                     mppt_data["p_abd_sum"] = round(mppt_data["p_abd_sum"], 2)
#                     mppt_data["title"] = f"Power: {mppt_data['p_abd_sum']} KW"
#                     # Append MPPT data to sn's children and add its P_abd to SN's total
#                     sn_data["children"].append(mppt_data)
#                     sn_data["p_abd_sum"] += mppt_data["p_abd_sum"]

#                 # Round and update SN title with P_abd
#                 sn_data["p_abd_sum"] = round(sn_data["p_abd_sum"], 2)
#                 sn_data["title"] = f"Power: {sn_data['p_abd_sum']} KW"
#                 # Append sn data to plant's children and add its P_abd to Plant's total
#                 plant_data["children"].append(sn_data)
#                 plant_data["p_abd_sum"] += sn_data["p_abd_sum"]

#             # Round and update Plant title with P_abd
#             plant_data["p_abd_sum"] = round(plant_data["p_abd_sum"], 2)
#             plant_data["title"] = f"Power: {plant_data['p_abd_sum']} KW"
#             # Append plant data to the hierarchy
#             org_chart.append(plant_data)

#         return jsonify({"status": "success", "data": org_chart})

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)})

@app.route('/get_hourly_values_inter', methods=['POST'])
def get_hourly_values_inter():
    # Parse payload
    payload = request.json
    date = payload.get("date")
    plant = payload.get("plant")
    inverter = payload.get("inverter")
    mppt = payload.get("mppt")
    string = payload.get("string")
    plant1 = payload.get("plant1")
    inverter1 = payload.get("inverter1")
    mppt1 = payload.get("mppt1")
    string1 = payload.get("string1")

    # Convert date to string for filtering
    date_filter = date[:10]  # Extract YYYY-MM-DD

    def build_query(plant, inverter, mppt, string):
        """Helper function to build query for a specific set."""
        query = {"Day_Hour": {"$regex": f"^{date_filter}"}}
        if plant:
            query["Plant"] = plant
        if inverter:
            query["sn"] = inverter
        if mppt:
            query["MPPT"] = mppt
        if string:
            query["Strings"] = string
        return query

    def build_pipeline(query):
        """Helper function to build aggregation pipeline for a query."""
        grouping_conditions = {
            "date": {"$substr": ["$Day_Hour", 0, 10]},
            "hour": {"$toInt": {"$substr": ["$Day_Hour", 11, 2]}}
        }
        if "Plant" in query:
            grouping_conditions["plant"] = "$Plant"
        if "sn" in query:
            grouping_conditions["inverter"] = "$sn"
        if "MPPT" in query:
            grouping_conditions["mppt"] = "$MPPT"
        if "Strings" in query:
            grouping_conditions["string"] = "$Strings"

        return [
            {"$match": query},
            {
                "$group": {
                    "_id": grouping_conditions,
                    "avg_u": {"$sum": "$P_abd"}  # Calculate the sum of field 'P_abd'
                }
            },
            {
                "$group": {
                    "_id": "$_id.date",
                    "hourly_values": {
                        "$push": {
                            "hour": "$_id.hour",
                            "value": "$avg_u"
                        }
                    }
                }
            },
            {"$sort": {"_id": 1}}
        ]

    # Queries and pipelines for both sets
    query_set1 = build_query(plant, inverter, mppt, string)
    query_set2 = build_query(plant1, inverter1, mppt1, string1)

    pipeline_set1 = build_pipeline(query_set1)
    pipeline_set2 = build_pipeline(query_set2)

    # Execute the aggregations
    results_set1 = list(device_hour_collection.aggregate(pipeline_set1))
    results_set2 = list(device_hour_collection.aggregate(pipeline_set2))

    # Determine the key dynamically based on input fields
    if plant and not (inverter or mppt or string):
        key1 = f"Plant 1 - {plant}"
        key2 = f"Plant 2 - {plant1}"
    elif plant and inverter and not (mppt or string):
        key1 = f"Inverter 1 - {inverter}"
        key2 = f"Inverter 2 - {inverter1}"
    elif plant and inverter and mppt and not string:
        key1 = f"MPPT 1 - {mppt}"
        key2 = f"MPPT 2 - {mppt1}"
    elif plant and inverter and mppt and string:
        key1 = f"String 1 - {string}"
        key2 = f"String 2 - {string1}"
    else:
        key1 = "Set 1"
        key2 = "Set 2"

    # Build the response
    response = {
        key1: [
            {
                "date": result["_id"],
                "hourly_values": result["hourly_values"]
            }
            for result in results_set1
        ],
        key2: [
            {
                "date": result["_id"],
                "hourly_values": result["hourly_values"]
            }
            for result in results_set2
        ]
    }

    return jsonify(response)


@app.route('/ridge_line_chart', methods=['POST'])
def ridge_line_chart():
    try:
        # Parse the request payload
        payload = request.json
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        plant = payload.get('plant')

        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        pipeline = [
            {
                "$match": {
                    "Day": {"$gte": start_date, "$lte": end_date},
                    "Plant": plant
                }
            },
            {
                "$project": {
                    "day": {
                        "$arrayElemAt": [{"$split": ["$Day_Hour", " "]}, 0]  # Get the date part
                    },
                    "weekday": {
                        "$dayOfWeek": {
                            "$dateFromString": {
                                "dateString": {"$arrayElemAt": [{"$split": ["$Day_Hour", " "]}, 0]}
                            }
                        }
                    },
                    "active_power": 1
                }
            },
            {
                "$group": {
                    "_id": {"day": "$day", "weekday": "$weekday"},
                    "total_active_power": {"$sum": "$active_power"}
                }
            },
            {
                "$sort": {"_id.day": 1, "_id.weekday": 1}
            }
        ]

        results = list(GM_Hourly.aggregate(pipeline))

        # Mapping weekday numbers (1 to 7) to weekday names
        weekday_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        # Prepare response data for chart
        chart_data = []
        for result in results:
            day = result["_id"]["day"]
            weekday = result["_id"]["weekday"]
            total_active_power = result["total_active_power"]
            
            # Only include data where active power is greater than 0
            if total_active_power > 0:
                chart_data.append({
                    "sport": weekday_names[weekday - 1],  # Mapping weekday to "sport"
                    "weight": round(total_active_power, 2),  # Active power as "weight"
                    "date": day  # Date as the "date"
                })

        return jsonify({"data": chart_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/grouped_data_efficency', methods=['POST'])
def grouped_data_efficency():
    try:
        # Parse payload
        payload = request.json
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        plant = payload.get('plant')
        inverter = payload.get('inverter')
        mppt = payload.get('mppt')
        string = payload.get('string')

        # Validate required fields
        if not (start_date and end_date and plant):
            return jsonify({"error": "start_date, end_date, and plant are required"}), 400

        # Convert date ranges
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)  # Include the full end date

        # Query filter
        query = {
            "Plant": plant,
            "Day_Hour": {"$gte": start_date.strftime('%Y-%m-%d'), "$lt": end_date.strftime('%Y-%m-%d')}
        }

        if inverter:
            query["sn"] = inverter
        if mppt:
            query["MPPT"] = mppt
        if string:
            query["Strings"] = string

        # Build aggregation pipeline
        group_fields = {
            "Day_Hour": "$Day_Hour",
            "Plant": "$Plant"
        }
        if inverter:
            group_fields["sn"] = "$sn"
        if mppt:
            group_fields["MPPT"] = "$MPPT"
        if string:
            group_fields["Strings"] = "$Strings"

        pipeline = [
            {"$match": query},
            {"$group": {"_id": group_fields, "P_abd_sum": {"$sum": "$P_abd"}}},
            {"$sort": {"_id.Day_Hour": 1}}  # Sort by Day_Hour
        ]

        results = list(device_hour_collection.aggregate(pipeline))

        # Pre-fetch helper results
        divisor_mapping = {}
        if inverter and not mppt and not string:
            helper_result = sum_watt_string(inverter)
            divisor_mapping[inverter] = helper_result["total_watt_string"]/1000
        elif inverter and mppt and not string:
            helper_result = sum_watt_mppt(inverter, mppt)
            divisor_mapping[f"{inverter}_{mppt}"] = helper_result["total_watt_string"]/1000
        elif inverter and mppt and string:
            helper_result = sum_watt_strings(inverter, string)
            divisor_mapping[f"{inverter}_{string}"] = helper_result["total_watt_string"]/1000

        # Build the output
        output = []
        for result in results:
            day_hour = result["_id"]["Day_Hour"]
            date_only = datetime.strptime(day_hour, '%Y-%m-%d %H').strftime('%Y-%m-%d') if day_hour else None
            hour = datetime.strptime(day_hour, '%Y-%m-%d %H').hour if day_hour else None
            p_abd_sum = result["P_abd_sum"]

            # Determine divisor
            if inverter and not mppt and not string:
                divisor = divisor_mapping.get(inverter, 2400)
            elif inverter and mppt and not string:
                divisor = divisor_mapping.get(f"{inverter}_{mppt}", 2400)
            elif inverter and mppt and string:
                divisor = divisor_mapping.get(f"{inverter}_{string}", 2400)
            else:
                divisor = 2400

            adjusted_p_abd_sum = p_abd_sum / divisor if divisor else 0

            # Build grouped data
            grouped_data = {
                "Day_Hour": date_only,
                "Hour": hour,
                "Plant": result["_id"]["Plant"],
                "P_abd_sum": adjusted_p_abd_sum * 100
            }
            if inverter:
                grouped_data["sn"] = result["_id"].get("sn")
            if mppt:
                grouped_data["MPPT"] = result["_id"].get("MPPT")
            if string:
                grouped_data["Strings"] = result["_id"].get("Strings")

            output.append(grouped_data)

        # Sort the output
        output.sort(key=lambda x: (x["Day_Hour"], x["Hour"]))

        return jsonify(output), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/radiation_intensity_inter', methods=['POST'])
# def radiation_intensity_inter():
#     try:
#         # Parse input data
#         data = request.json
#         date = data.get("date")  # Format: YYYY-MM-DD
#         station_code1 = data.get("stationCode1")  # First station code
#         station_code2 = data.get("stationCode2")  # Second station code
#         option = data.get("option")  # Option 1 or 2

#         # Validate input
#         if not date or not station_code1 or not station_code2 or option not in [1, 2]:
#             return jsonify({"error": "Missing or invalid parameters. Provide 'date', 'stationCode1', 'stationCode2', and 'option' (1 or 2)."}), 400

#         # Convert date to timestamp range
#         try:
#             start_datetime = datetime.strptime(f"{date} 00:00:00", "%Y-%m-%d %H:%M:%S")
#             end_datetime = datetime.strptime(f"{date} 23:59:59", "%Y-%m-%d %H:%M:%S")
#         except ValueError:
#             return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

#         # Convert to string format to match the database field
#         start_timestamp = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
#         end_timestamp = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

#         # Define capacity values for station codes
#         capacities = {
#             "NE=53278269": 2400,
#             "NE=49739688": 3000,
#             "NE=51173154": 5000
#         }

#         # Build MongoDB query for the single date and both station codes
#         query = {
#             "$and": [
#                 {"stationCode": {"$in": [station_code1, station_code2]}},
#                 {"timestamp": {"$gte": start_timestamp, "$lte": end_timestamp}}
#             ]
#         }

#         # Adjust the field projection based on the option
#         projection = {
#             "timestamp": 1,
#             "stationCode": 1,
#             "_id": 0
#         }

#         if option == 1:
#             projection["dataItemMap.radiation_intensity"] = 1
#         elif option == 2:
#             projection["dataItemMap.inverter_power"] = 1

#         # Fetch results from MongoDB
#         results = list(collection.find(query, projection))

#         # Initialize grouped data structure for both station codes
#         grouped_data = {
#             station_code1: [0] * 24,  # Default 24-hour values for stationCode1
#             station_code2: [0] * 24  # Default 24-hour values for stationCode2
#         }

#         # Process results based on the option
#         for result in results:
#             station_code = result["stationCode"]
#             collect_time = datetime.strptime(result["timestamp"], "%Y-%m-%d %H:%M:%S")
#             hour = collect_time.hour

#             if option == 1:
#                 # Option 1: Use radiation_intensity
#                 value = result["dataItemMap"].get("radiation_intensity", 0)
#             elif option == 2:
#                 # Option 2: Use inverter_power and divide by capacity
#                 inverter_power = result["dataItemMap"].get("inverter_power", 0) or 0
#                 capacity = capacities.get(station_code, 1)
#                 value = inverter_power / capacity if capacity else 0

#             if station_code in grouped_data:
#                 grouped_data[station_code][hour] = value

#         # Ensure both station codes are included, duplicating values if necessary
#         if station_code1 == station_code2:
#             grouped_data[station_code2] = grouped_data[station_code1].copy()

#         # Format output
#         output = []
#         for station_code, hourly_values in grouped_data.items():
#             output.append({
#                 "stationCode": station_code,
#                 "date": date,
#                 "hourly_values": [{"hour": hour, "value": value} for hour, value in enumerate(hourly_values)]
#             })

#         return jsonify(output), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/radiation_intensity_inter', methods=['POST'])
def radiation_intensity_inter():
    try:
        # Parse input data
        data = request.json
        date = data.get("date")  # Format: YYYY-MM-DD
        station_code1 = data.get("stationCode1")  # First station code
        station_code2 = data.get("stationCode2")  # Second station code
        option = data.get("option")  # Option 1 or 2

        # Validate input
        if not date or not station_code1 or not station_code2 or option not in [1, 2]:
            return jsonify({"error": "Missing or invalid parameters. Provide 'date', 'stationCode1', 'stationCode2', and 'option' (1 or 2)."}), 400

        # Map custom labels
        station_label_map = {
            station_code1: "Plant 1 - Coca Cola Faisalabad",
            station_code2: "Plant 2 - Coca Cola Faisalabad"
        }

        # Generate a default 24-hour data structure
        default_hourly_values = [{"hour": hour, "value": 0} for hour in range(24)]

        # Build MongoDB query
        query = {
            "$and": [
                {"stationCode": {"$in": [station_code1, station_code2]}},
                {"timestamp": {"$regex": f"^{date}"}}
            ]
        }

        # Projection based on the option
        projection = {
            "timestamp": 1,
            "stationCode": 1,
            "_id": 0
        }
        if option == 1:
            projection["dataItemMap.radiation_intensity"] = 1
        elif option == 2:
            projection["dataItemMap.inverter_power"] = 1

        # Fetch data from MongoDB
        results = list(collection.find(query, projection))

        # Prepare the grouped data for both station codes
        grouped_data = {station_code1: default_hourly_values.copy(), station_code2: default_hourly_values.copy()}

        for result in results:
            station_code = result["stationCode"]
            timestamp = datetime.strptime(result["timestamp"], "%Y-%m-%d %H:%M:%S")
            hour = timestamp.hour

            if option == 1:
                value = result["dataItemMap"].get("radiation_intensity", 0)
            elif option == 2:
                value = result["dataItemMap"].get("inverter_power", 0)

            if station_code in grouped_data:
                grouped_data[station_code][hour]["value"] = value

        # Prepare the final response with mapped labels
        output = [
            {
                "stationCode": station_label_map.get(station_code1, station_code1),
                "date": date,
                "hourly_values": grouped_data[station_code1]
            },
            {
                "stationCode": station_label_map.get(station_code2, station_code2),
                "date": date,
                "hourly_values": grouped_data[station_code2]
            }
        ]

        return jsonify(output), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def preprocess_and_predict(data):
    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

    # Feature engineering
    data['Date'] = data['timestamp'].dt.date
    data['Hour'] = data['timestamp'].dt.hour
    data['Day_of_Year'] = data['timestamp'].dt.dayofyear
    data['Hour_Sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_Cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    data['Day_Sin'] = np.sin(2 * np.pi * data['Day_of_Year'] / 365)
    data['Day_Cos'] = np.cos(2 * np.pi * data['Day_of_Year'] / 365)
    data['radiation_lag_1'] = data['radiation_intensity'].shift(1).fillna(0)
    data['radiation_lag_2'] = data['radiation_intensity'].shift(2).fillna(0)
    data['radiation_hour_interaction'] = data['radiation_intensity'] * data['Hour']
    data['radiation_day_interaction'] = data['radiation_intensity'] * data['Day_of_Year']
    data['rate_of_change'] = data['radiation_intensity'].diff().fillna(0)
    data['irradiance_rate'] = data['radiation_intensity'].pct_change().fillna(0) * 100
    data['rolling_mean_2h'] = data['radiation_intensity'].rolling(window=2).mean().fillna(0)
    data['rolling_std_2h'] = data['radiation_intensity'].rolling(window=2).std().fillna(0)
    data['rolling_mean_3h'] = data['radiation_intensity'].rolling(window=3).mean().fillna(0)
    data['rolling_std_3h'] = data['radiation_intensity'].rolling(window=3).std().fillna(0)

    # Daily accumulated irradiance
    daily_accumulated = data.groupby('Date')['radiation_intensity'].sum().reset_index()
    daily_accumulated.columns = ['Date', 'daily_accumulated_irradiance']
    data = pd.merge(data, daily_accumulated, how='left', on='Date')

    data['ema_2h'] = data['radiation_intensity'].ewm(span=2).mean().fillna(0)
    data['relative_radiation'] = data['radiation_intensity'] / data['radiation_intensity'].rolling(window=24).max().fillna(0)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # Feature selection
    features = [
        'Hour', 'Day_of_Year', 'radiation_intensity', 'Hour_Sin', 'Hour_Cos', 
        'Day_Sin', 'Day_Cos', 'radiation_lag_1', 'radiation_lag_2', 
        'radiation_hour_interaction', 'radiation_day_interaction', 
        'relative_radiation', 'ema_2h', 'rate_of_change', 'irradiance_rate', 
        'rolling_mean_2h', 'rolling_std_2h', 'rolling_mean_3h', 'rolling_std_3h', 
        'daily_accumulated_irradiance'
    ]
    filtered_df = data[features]
    #X=filtered_df [['Hour', 'radiation_intensity']]
    X=filtered_df [['Hour', 'Day_of_Year', 'radiation_intensity', 'Hour_Sin', 'Hour_Cos', 'Day_Sin','Day_Cos', 
                'radiation_lag_1', 'radiation_lag_2', 'radiation_hour_interaction','radiation_day_interaction','relative_radiation','ema_2h',
                'rate_of_change','irradiance_rate','rolling_mean_2h','rolling_std_2h','rolling_mean_3h','rolling_std_3h','daily_accumulated_irradiance']]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Predict power
    data['Predicted Power'] = model.predict(X_scaled)
    return data


@app.route('/calculate_suppression', methods=['POST'])
def calculate_suppression():
    try:
        payload = request.get_json()
        if not payload:
            raise ValueError("Payload is missing or invalid.")

        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        station_code = payload.get('stationCode') 
        tarrif = payload.get('tarrif')
        option = payload.get('option')
        if not start_date or not end_date:
            raise ValueError("'start_date' and 'end_date' must be provided in the payload.")
        if option not in [1, 2]:
            raise ValueError("'option' must be either 1 or 2.")
        # Convert start_date and end_date to datetime objects
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = pd.Timestamp(start_date) - pd.Timedelta(days=1)
        # Fetch data from MongoDB within the date range
        mongo_data = list(collection.find({
           "timestamp": {
                "$gte": start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                "$lt": end_date.strftime('%Y-%m-%dT%H:%M:%S')  # Include end_date
            },
            "stationCode": station_code
        }))
       
        if not mongo_data:
            raise ValueError("No data found in the specified date range.")

        # Convert MongoDB data to a Pandas DataFrame
        data = []
        for record in mongo_data:
            data.append({
                "timestamp": record["timestamp"],
                "radiation_intensity": round(record["dataItemMap"].get("radiation_intensity", 0) or 0, 2),
                "inverter_power": round(record["dataItemMap"].get("inverter_power", 0) or 0, 2)
            })
        df = pd.DataFrame(data)
        
        # Ensure proper datetime formatting
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].isna().any():
            raise ValueError("Invalid datetime entries found in the 'timestamp' column. Please check the data.")
        df['Date'] = df['timestamp'].dt.date

        # Feature engineering
        df['Hour'] = df['timestamp'].dt.hour
        df['Day_of_Year'] = df['timestamp'].dt.dayofyear
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Day_Sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
        df['Day_Cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)
        df['radiation_lag_1'] = df['radiation_intensity'].shift(1).fillna(0)
        df['radiation_lag_2'] = df['radiation_intensity'].shift(2).fillna(0)
        df['radiation_hour_interaction'] = df['radiation_intensity'] * df['Hour']
        df['radiation_day_interaction'] = df['radiation_intensity'] * df['Day_of_Year']
        df['rate_of_change'] = df['radiation_intensity'].diff().fillna(0)
        df['irradiance_rate'] = df['radiation_intensity'].pct_change().fillna(0) * 100
        df['rolling_mean_2h'] = df['radiation_intensity'].rolling(window=2).mean().fillna(0)
        df['rolling_std_2h'] = df['radiation_intensity'].rolling(window=2).std().fillna(0)
        df['rolling_mean_3h'] = df['radiation_intensity'].rolling(window=3).mean().fillna(0)
        df['rolling_std_3h'] = df['radiation_intensity'].rolling(window=3).std().fillna(0)
        # Daily accumulated radiation intensity
        daily_accumulated = df.groupby('Date')['radiation_intensity'].sum().reset_index()
        daily_accumulated.columns = ['Date', 'daily_accumulated_irradiance']
        df = pd.merge(df, daily_accumulated, how='left', on='Date')

        # Additional features
        df['ema_2h'] = df['radiation_intensity'].ewm(span=2).mean().fillna(0)
        df['relative_radiation'] = df['radiation_intensity'] / df['radiation_intensity'].rolling(window=24).max().fillna(0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # Filtering and scaling
        filtered_columns = ['Hour', 'Day_of_Year', 'radiation_intensity', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 
                            'radiation_lag_1', 'radiation_lag_2', 'radiation_hour_interaction', 'radiation_day_interaction', 
                            'relative_radiation', 'ema_2h', 'rate_of_change', 'irradiance_rate', 'rolling_mean_2h', 
                            'rolling_std_2h', 'rolling_mean_3h', 'rolling_std_3h', 'daily_accumulated_irradiance']
        X = df[filtered_columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Model prediction
        best_model_loaded = joblib.load('Best_RF_Model_CocaCola.pkl')
        df['Predicted Power'] = best_model_loaded.predict(X_scaled)
        # Ensure 'DateTime' column is in datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract the Date and Hour from the 'DateTime' column
        df['Hour'] = df['timestamp'].dt.hour
        df['Date'] = df['timestamp'].dt.date
       # Daily statistics
        df['Difference'] = df['Predicted Power'] - df['inverter_power']
        df_daily = df.groupby('Date').agg({
            'Difference': 'sum',
            'inverter_power': 'sum',
            'Predicted Power': 'sum',
            'radiation_intensity': 'sum'
        }).reset_index()

        df_daily['score'] = np.where(
            df_daily['radiation_intensity'] > 0,
            df_daily['inverter_power'] / (df_daily['radiation_intensity'] * 1000),
            np.nan
        )
        df_daily.dropna(subset=['score'], inplace=True)

        df_daily['Power Ratio (%)'] = (df_daily['inverter_power'] / df_daily['Predicted Power']) * 100
        df_daily['MAE'] = np.abs(df_daily['Predicted Power'] - df_daily['inverter_power'])
        df_daily['MAPE'] = (df_daily['MAE'] / df_daily['inverter_power'].replace(0, np.nan)) * 100
        df_daily['MAPE'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df_daily.dropna(subset=['MAPE'], inplace=True)

        # Anomaly detection using GMM with Optuna
        X_mae = df_daily['MAE'].values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled_mae = scaler.fit_transform(X_mae)

        def objective(trial):
            n_components = trial.suggest_int("n_components", 2, 2)
            covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
            gmm_model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
            gmm_model.fit(X_scaled_mae)
            return gmm_model.bic(X_scaled_mae)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=150)

        best_params = study.best_params
        gmm_model = GaussianMixture(n_components=best_params["n_components"], covariance_type=best_params["covariance_type"], random_state=42)
        gmm_model.fit(X_scaled_mae)

        log_likelihood = gmm_model.score_samples(X_scaled_mae)
        def optimize_threshold(log_likelihood):
            thresholds = np.linspace(np.min(log_likelihood), np.max(log_likelihood), 1000)
            best_threshold = thresholds[np.argmin(np.abs(thresholds - np.median(log_likelihood)))]  # Choose threshold around median
            return best_threshold
        best_threshold = optimize_threshold(log_likelihood)
        relaxation_factor = 1.0
        df_daily['mae anomaly'] = log_likelihood < (best_threshold * relaxation_factor)

        # Median MAE excluding anomalies
        median_mae = df_daily.loc[~df_daily['mae anomaly'], 'MAE'].median()
        df_daily['Suppression'] = (df_daily['Predicted Power'] - df_daily['inverter_power'] - median_mae).clip(lower=0)
        if tarrif is None:
            tarrif = 0  # Default to 0 if tarrif is not provided
        df_daily['Suppression Cost'] = round(df_daily['Suppression'],0) * tarrif
        # Flag significant suppression
        suppression_threshold = df_daily['Suppression'].quantile(0.95)
        df_daily['significant_suppression'] = df_daily['Suppression'] > suppression_threshold
        if option == 1:
            # Daily suppression response
            result =  df_daily = df_daily.round({
            'Suppression': 0,
            'Suppression Cost': 0,
            'Predicted Power': 0,
            'inverter_power': 0,
            'radiation_intensity': 0
        })
            result = df_daily[['Date', 'Suppression', 'Suppression Cost', 'significant_suppression']].to_dict(orient='records')
        elif option == 2:
            # Monthly suppression response
            df_daily['Date'] = pd.to_datetime(df_daily['Date'])
            
            # Filter rows to only include data within the start_date and end_date range
            df_filtered = df_daily[(df_daily['Date'] >= start_date) & (df_daily['Date'] <= end_date)]
            
            # Extract year and month for grouping
            df_filtered['YearMonth'] = df_filtered['Date'].dt.to_period('M')
            
            # Group by Year-Month and calculate suppression sum
            df_monthly_suppression = df_filtered.groupby('YearMonth')['Suppression'].sum().reset_index()
            
            # Format YearMonth into readable format
            df_monthly_suppression['YearMonth'] = df_monthly_suppression['YearMonth'].dt.strftime('%b %Y')
            
            # Create a complete date range of months between start_date and end_date
            # Ensure the range starts from the actual month of start_date
            all_months = pd.date_range(start=start_date.replace(day=1), end=end_date, freq='MS').to_period('M')
            all_months_df = pd.DataFrame({
                'YearMonth': all_months.strftime('%b %Y'),
                'Suppression': 0
            })
            
            # Merge with the actual data to ensure all months are included
            df_complete = pd.merge(all_months_df, df_monthly_suppression, on='YearMonth', how='left')
            
            # Fill missing suppression values with 0
            df_complete['Suppression'] = df_complete['Suppression_y'].fillna(0)
            df_complete['Cost'] = df_complete['Suppression'] * tarrif
            
            # Keep only relevant columns
            df_complete = df_complete[['YearMonth', 'Suppression','Cost']]
            
            # Convert to dictionary format
            result = df_complete.to_dict(orient='records')

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})



@app.route('/calculate_dash_suppression', methods=['POST'])
def calculate_dash_suppression():
    try:
        # Parse the payload
        payload = request.get_json()
        if not payload:
            raise ValueError("Payload is missing or invalid.")

        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        station_code = payload.get('stationCode')
        tarrif = payload.get('tarrif', 0)  # Default to 0 if not provided
        option = payload.get('option')

        if not start_date or not end_date:
            raise ValueError("'start_date' and 'end_date' must be provided in the payload.")
        if option not in [1, 2, 3]:
            raise ValueError("'option' must be either 1, 2, or 3.")

        # Convert start_date and end_date to datetime objects
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Fetch data from MongoDB (mocked here as a placeholder DataFrame)
        mongo_data = list(collection.find({
            "timestamp": {
                "$gte": start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                "$lt": end_date.strftime('%Y-%m-%dT%H:%M:%S')
            },
            "stationCode": station_code
        }))

        if not mongo_data:
            raise ValueError("No data found in the specified date range.")

        # Convert MongoDB data to a DataFrame
        data = [
            {
                "timestamp": record["timestamp"],
                "radiation_intensity": round(record["dataItemMap"].get("radiation_intensity", 0) or 0, 2),
                "inverter_power": round(record["dataItemMap"].get("inverter_power", 0) or 0, 2)
            } for record in mongo_data
        ]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['Date'] = df['timestamp'].dt.date

        df['Hour'] = df['timestamp'].dt.hour
        df['Day_of_Year'] = df['timestamp'].dt.dayofyear
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Day_Sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
        df['Day_Cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)
        df['radiation_lag_1'] = df['radiation_intensity'].shift(1).fillna(0)
        df['radiation_lag_2'] = df['radiation_intensity'].shift(2).fillna(0)
        df['radiation_hour_interaction'] = df['radiation_intensity'] * df['Hour']
        df['radiation_day_interaction'] = df['radiation_intensity'] * df['Day_of_Year']
        df['rate_of_change'] = df['radiation_intensity'].diff().fillna(0)
        df['irradiance_rate'] = df['radiation_intensity'].pct_change().fillna(0) * 100
        df['rolling_mean_2h'] = df['radiation_intensity'].rolling(window=2).mean().fillna(0)
        df['rolling_std_2h'] = df['radiation_intensity'].rolling(window=2).std().fillna(0)
        df['rolling_mean_3h'] = df['radiation_intensity'].rolling(window=3).mean().fillna(0)
        df['rolling_std_3h'] = df['radiation_intensity'].rolling(window=3).std().fillna(0)
        # Daily accumulated radiation intensity
        daily_accumulated = df.groupby('Date')['radiation_intensity'].sum().reset_index()
        daily_accumulated.columns = ['Date', 'daily_accumulated_irradiance']
        df = pd.merge(df, daily_accumulated, how='left', on='Date')

        # Additional features
        df['ema_2h'] = df['radiation_intensity'].ewm(span=2).mean().fillna(0)
        df['relative_radiation'] = df['radiation_intensity'] / df['radiation_intensity'].rolling(window=24).max().fillna(0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # Filtering and scaling
        filtered_columns = ['Hour', 'Day_of_Year', 'radiation_intensity', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 
                            'radiation_lag_1', 'radiation_lag_2', 'radiation_hour_interaction', 'radiation_day_interaction', 
                            'relative_radiation', 'ema_2h', 'rate_of_change', 'irradiance_rate', 'rolling_mean_2h', 
                            'rolling_std_2h', 'rolling_mean_3h', 'rolling_std_3h', 'daily_accumulated_irradiance']
        X = df[filtered_columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Model prediction
        best_model_loaded = joblib.load('Best_RF_Model_CocaCola.pkl')
        df['Predicted Power'] = best_model_loaded.predict(X_scaled)
        # Ensure 'DateTime' column is in datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract the Date and Hour from the 'DateTime' column
        df['Hour'] = df['timestamp'].dt.hour
        df['Date'] = df['timestamp'].dt.date
       # Daily statistics
        df['Difference'] = df['Predicted Power'] - df['inverter_power']
        df_daily = df.groupby('Date').agg({
            'Difference': 'sum',
            'inverter_power': 'sum',
            'Predicted Power': 'sum',
            'radiation_intensity': 'sum'
        }).reset_index()

        df_daily['score'] = np.where(
            df_daily['radiation_intensity'] > 0,
            df_daily['inverter_power'] / (df_daily['radiation_intensity'] * 1000),
            np.nan
        )
        df_daily.dropna(subset=['score'], inplace=True)

        df_daily['Power Ratio (%)'] = (df_daily['inverter_power'] / df_daily['Predicted Power']) * 100
        df_daily['MAE'] = np.abs(df_daily['Predicted Power'] - df_daily['inverter_power'])
        df_daily['MAPE'] = (df_daily['MAE'] / df_daily['inverter_power'].replace(0, np.nan)) * 100
        df_daily['MAPE'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df_daily.dropna(subset=['MAPE'], inplace=True)

        # Anomaly detection using GMM with Optuna
        X_mae = df_daily['MAE'].values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled_mae = scaler.fit_transform(X_mae)

        def objective(trial):
            n_components = trial.suggest_int("n_components", 2, 2)
            covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
            gmm_model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
            gmm_model.fit(X_scaled_mae)
            return gmm_model.bic(X_scaled_mae)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=150)

        best_params = study.best_params
        gmm_model = GaussianMixture(n_components=best_params["n_components"], covariance_type=best_params["covariance_type"], random_state=42)
        gmm_model.fit(X_scaled_mae)

        log_likelihood = gmm_model.score_samples(X_scaled_mae)
        def optimize_threshold(log_likelihood):
            thresholds = np.linspace(np.min(log_likelihood), np.max(log_likelihood), 1000)
            best_threshold = thresholds[np.argmin(np.abs(thresholds - np.median(log_likelihood)))]  # Choose threshold around median
            return best_threshold
        best_threshold = optimize_threshold(log_likelihood)
        relaxation_factor = 1.0
        df_daily['mae anomaly'] = log_likelihood < (best_threshold * relaxation_factor)

        # Median MAE excluding anomalies
        median_mae = df_daily.loc[~df_daily['mae anomaly'], 'MAE'].median()
        df_daily['Suppression'] = (df_daily['Predicted Power'] - df_daily['inverter_power'] - median_mae).clip(lower=0)
        if tarrif is None:
            tarrif = 0  # Default to 0 if tarrif is not provided
        df_daily['Suppression Cost'] = round(df_daily['Suppression'],0) * tarrif
        # Flag significant suppression
        suppression_threshold = df_daily['Suppression'].quantile(0.95)
        df_daily['significant_suppression'] = df_daily['Suppression'] > suppression_threshold
        if option == 1:
            yesterday = datetime.now() - timedelta(days=1)
            day_before_yesterday = datetime.now() - timedelta(days=2)

            response = df_daily[df_daily['Date'].isin([yesterday.date(), day_before_yesterday.date()])]
            response = response[['Date', 'Suppression', 'Suppression Cost']]
            response['Date'] = response['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))  # Format the date as YYYY-MM-DD
            response = response.to_dict(orient='records')

            # Ensure dates are unique and add missing dates if necessary
            dates_to_check = [yesterday.strftime('%Y-%m-%d'), day_before_yesterday.strftime('%Y-%m-%d')]
            for date in dates_to_check:
                if not any(entry['Date'] == date for entry in response):
                    response.append({"Date": date, "Suppression": 0, "Suppression Cost": 0})

            # Sort the response by Date to maintain chronological order
            response.sort(key=lambda x: x['Date'])

        elif option == 2:
            current_month = datetime.now().strftime('%Y-%m')
            last_month = (datetime.now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
            
            df_daily['YearMonth'] = df_daily['Date'].astype(str).str[:7]
            suppression_summary = df_daily.groupby('YearMonth')['Suppression'].sum().reset_index()

            current_month_value = suppression_summary[suppression_summary['YearMonth'] == current_month]['Suppression'].sum()
            last_month_value = suppression_summary[suppression_summary['YearMonth'] == last_month]['Suppression'].sum()

            response = [
                {"Month": datetime.strptime(last_month, '%Y-%m').strftime('%B %Y'), "Suppression": last_month_value if last_month_value else 0},
                {"Month": datetime.strptime(current_month, '%Y-%m').strftime('%B %Y'), "Suppression": current_month_value if current_month_value else 0}

            ]
        elif option == 3:
            current_year = datetime.now().year
            last_year = current_year - 1

            df_daily['Year'] = pd.to_datetime(df_daily['Date']).dt.year
            suppression_summary = df_daily.groupby('Year')['Suppression'].sum().reset_index()

            current_year_value = suppression_summary[suppression_summary['Year'] == current_year]['Suppression'].sum()
            last_year_value = suppression_summary[suppression_summary['Year'] == last_year]['Suppression'].sum()

            response = [
                {"Year": last_year, "Suppression": last_year_value if last_year_value else 0},
                {"Year": current_year, "Suppression": current_year_value if current_year_value else 0}

            ]
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/process', methods=['POST'])
def process_file():
    try:
        # Get the request payload
        payload = request.get_json()
        option = payload.get('option')
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        plant = payload.get('plant')  # Get the plant from payload

        # Validate the payload
        if not start_date or not end_date:
            return jsonify({'status': 'error', 'message': 'Start and end dates are required'})
        if not plant:
            return jsonify({'status': 'error', 'message': 'Plant is required'})

        # Validate and parse start and end dates
        end_date = pd.to_datetime(end_date)
        start_date = pd.Timestamp(start_date) - pd.Timedelta(days=1)
        # Fetch data from MongoDB within the date range and matching the plant
        cursor = collection.find(
            {
                'timestamp': {
                    '$gte': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    '$lte': end_date.strftime('%Y-%m-%dT%H:%M:%S')
                },
                'stationCode': plant  # Filter by Plant
            },
            {'timestamp': 1, 'dataItemMap': 1, 'stationCode': 1, '_id': 0}  # Include Plant field if needed
        )
        df = pd.json_normalize(list(cursor), sep='_')
        
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data found for the given date range and plant'})

        # Extract required fields from dataItemMap
        df['timestamp'] = df['timestamp']
        df['inverter_power'] = df['dataItemMap_inverter_power']
        df['radiation_intensity'] = df['dataItemMap_radiation_intensity']
        df['inverter_power'] = df['inverter_power'].round(2)  # Round to 2 decimal places
        df['radiation_intensity'] = df['radiation_intensity'].round(2)

        # Preprocess and predict based on the selected option
        if option == 1:
            processed_data = preprocess_and_predict(df)
            processed_data = processed_data.drop_duplicates(subset=['timestamp'])

            plot_data = processed_data[['timestamp', 'inverter_power', 'Predicted Power', 'radiation_intensity']].to_dict('records')
            return jsonify({'status': 'success', 'plot_data': plot_data})

        elif option == 2:
            processed_data = preprocess_and_predict(df)

            processed_data['Difference'] = processed_data['Predicted Power'] - processed_data['inverter_power']

            df_daily = processed_data.groupby('Date').agg(
                {
                    'Difference': 'sum',
                    'inverter_power': 'sum',
                    'Predicted Power': 'sum',
                    'radiation_intensity': 'sum'
                }
            ).reset_index()

            df_daily['Color Label'] = np.where(
                df_daily['Difference'] > 0, 
                'Predicted > Generated',
                np.where(df_daily['Difference'] < 0, 'Generated > Predicted', 'Equal')
            )

            chart_data = {
                'dates': df_daily['Date'].astype(str).tolist(),
                'difference': df_daily['Difference'].tolist(),
                'inverter_power': df_daily['inverter_power'].tolist(),
                'predicted_power': df_daily['Predicted Power'].tolist(),
                'radiation_intensity': df_daily['radiation_intensity'].tolist(),
                'color_label': df_daily['Color Label'].tolist()
            }

            return jsonify({'status': 'success', 'chart_data': chart_data})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid option provided'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/combined_dash_data', methods=['POST'])
def combined_dash_data():
    try:
        # Parse request payload
        payload = request.get_json()
        option = payload.get("option")

        if option not in [1, 2, 3]:
            return jsonify({"error": "Invalid option. Valid options are 1, 2, and 3."}), 400

        # Initialize response structure
        response = {}

        # Define date range based on the option
        today = pd.Timestamp.today()
        if option == 1:  # Yesterday
            target_day = today - pd.Timedelta(days=1)
            start_date = target_day.replace(hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
            end_date = target_day.replace(hour=23, minute=59, second=59, microsecond=999999).strftime('%Y-%m-%d %H:%M:%S')
        elif option == 2:  # This month
            start_date = today.replace(day=1).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
        elif option == 3:  # This year
            start_date = today.replace(month=1, day=1).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')

        # MongoDB Query for suppression data
        query = {'timestamp': {'$gte': start_date, '$lte': end_date}, "stationCode": "NE=53278269"}
        cursor = collection.find(query, {'timestamp': 1, 'dataItemMap': 1, 'stationCode': 1, '_id': 0})
        df = pd.json_normalize(list(cursor), sep='_')

        # Debug: Check data fetched
        print("Fetched Data:")
        print(df.head())

        # Validate the data
        if df.empty:
            return jsonify({"error": "No data found for the given query"}), 404

        # Data preprocessing
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['inverter_power'] = df.get('dataItemMap_inverter_power', np.nan).fillna(0)
        df['radiation_intensity'] = df.get('dataItemMap_radiation_intensity', np.nan).fillna(0)

        # Predict 'Predicted Power'
        processed_data = preprocess_and_predict(df)
        processed_data['Predicted Power'] = processed_data.get('Predicted Power', np.nan).fillna(0)

        # Calculate MAE
        processed_data['MAE'] = np.abs(processed_data['Predicted Power'] - processed_data['inverter_power'])

        # Calculate Median MAE
        median_mae = processed_data['MAE'].median()

        # Suppression Calculation
        processed_data['Suppression'] = (
            processed_data['Predicted Power'] -
            processed_data['inverter_power'] -
            median_mae
        ).clip(lower=0)

        # Debug: Check suppression values
        print("Processed Data with Suppression:")
        print(processed_data[['timestamp', 'Predicted Power', 'inverter_power', 'Suppression', 'MAE']].head())

        # Calculate total suppression
        total_suppression = processed_data['Suppression'].sum()
        response['suppression'] = total_suppression

        # Fetch stat data
        if option == 1:  # Yesterday
            yesterday = datetime.now() - timedelta(days=1)
            yesterday_str = yesterday.strftime('%Y-%m-%d')
            pipeline = [
                {"$match": {"Day_Hour": {"$regex": f"^{yesterday_str}"}}},
                {"$group": {"_id": None, "active_power_sum": {"$sum": "$P_abd"}}}
            ]
            results = list(device_hour_collection.aggregate(pipeline))
            kw = round(results[0]["active_power_sum"], 2) if results else 0
            capacity = 2400
        elif option == 2:  # This month
            now = datetime.now()
            first_day_current_month = datetime(now.year, now.month, 1)
            pipeline = [
                {"$match": {"Day": {"$gte": first_day_current_month.strftime('%Y-%m-%d')}}},
                {"$group": {
                    "_id": {"month": {"$substr": ["$Day", 0, 7]}},
                    "active_power_sum": {"$sum": "$P_abd"},
                    "unique_days": {"$addToSet": "$Day"}
                }}
            ]
            results = list(device_day_collection.aggregate(pipeline))
            kw = round(results[0]["active_power_sum"], 2) if results else 0
            unique_days_count = len(results[0]["unique_days"]) if results else 0
            capacity = unique_days_count * 2400
        elif option == 3:  # This year
            current_year = datetime.now().year
            pipeline = [
                {"$match": {"Day": {"$regex": f"^{current_year}"}}},
                {"$group": {
                    "_id": {"year": {"$substr": ["$Day", 0, 4]}},
                    "active_power_sum": {"$sum": "$P_abd"},
                    "unique_days": {"$addToSet": "$Day"}
                }}
            ]
            results = list(device_day_collection.aggregate(pipeline))
            kw = round(results[0]["active_power_sum"], 2) if results else 0
            unique_days_count = len(results[0]["unique_days"]) if results else 0
            capacity = unique_days_count * 2400

        predictedco2 = round(capacity * 0.56, 2)
        actualco2 = round(kw * 0.56, 2)

        # Add stat data to response
        response.update({
            "kw": round(kw),
            "capacity": round(capacity),
            "predictedco2": round(predictedco2),
            "actualco2": round(actualco2)
        })

        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/dash_suppression', methods=['POST'])
def dash_suppression():
    try:
        # Get the request payload
        payload = request.get_json()
        option = payload.get('option')

        if option not in [1, 2, 3]:
            return jsonify({'status': 'error', 'message': 'Invalid option. Valid options are 1, 2, or 3.'})

        # Determine the date range based on the option
        today = pd.Timestamp.today()
        if option == 1:  # Yesterday
            # yesterday = today - pd.Timedelta(days=17)  # Subtract 1 day
            today = datetime.today()  # Current date and time
            target_day = today - timedelta(days=1)  # 25 days ago
            # Set start_date and end_date to cover the entire 25th day
            start_date = target_day.replace(hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
            end_date = target_day.replace(hour=23, minute=59, second=59, microsecond=999999).strftime('%Y-%m-%d %H:%M:%S')
        elif option == 2:  # This month
            start_date = today.replace(day=1).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
        elif option == 3:  # This year
            start_date = today.replace(month=1, day=1).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')

        # Fetch data from MongoDB
        query = {'timestamp': {'$gte': start_date, '$lte': end_date}}
        cursor = collection.find(query, {'timestamp': 1, 'dataItemMap': 1, '_id': 0})
        df = pd.json_normalize(list(cursor), sep='_')

        # Extract required fields from dataItemMap
        df['timestamp'] = df['timestamp']
        df['inverter_power'] = df['dataItemMap_inverter_power']
        df['radiation_intensity'] = df['dataItemMap_radiation_intensity']

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # Preprocess and predict to add 'Predicted Power' column
        processed_data = preprocess_and_predict(df)

        # Calculate Suppression and related fields
        processed_data['Difference'] = processed_data['Predicted Power'] - processed_data['inverter_power']
        processed_data['score'] = np.where(
            processed_data['radiation_intensity'] > 0,
            processed_data['inverter_power'] / (processed_data['radiation_intensity'] * 1000),
            np.nan
        )
        processed_data = processed_data.dropna(subset=['score'])

        # Calculate rolling average score
        processed_data['rolling_avg_score'] = processed_data['score'].rolling(window=14, min_periods=1).median()
        processed_data['rolling_avg_score'] = processed_data['rolling_avg_score'].bfill()

        # Calculate Total Suppression
        median_score = processed_data['rolling_avg_score'].median()
        processed_data['Loss_Dust & Misc Factors'] = (
            processed_data['Predicted Power'] - 
            (processed_data['Predicted Power'] * (processed_data['rolling_avg_score'] / median_score))
        ).clip(lower=0)
        processed_data['Suppression'] = (
            processed_data['Predicted Power'] - processed_data['inverter_power'] -
            processed_data['Loss_Dust & Misc Factors']
        ).clip(lower=0)
        processed_data['Total Suppression'] = (
            processed_data['Suppression'] + processed_data['Loss_Dust & Misc Factors']
        )

        # Calculate the sum of suppression
        total_suppression = processed_data['Total Suppression'].sum()

        # Return the response
        return jsonify({'status': 'success', 'suppression': total_suppression})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/power_ratio', methods=['POST'])
def power_ratio():
    try:
        # Get the request payload
        payload = request.get_json()
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        plant = payload.get('plant')
        end_date = datetime.strptime(payload.get('end_date'), "%Y-%m-%d")
        end_date = end_date + timedelta(days=1)
        end_date = end_date.strftime("%Y-%m-%d")
        # Fetch data from MongoDB
        query = {'timestamp': {'$gte': start_date, '$lte': end_date},"stationCode":plant}
        cursor = collection.find(query, {'timestamp': 1, 'dataItemMap': 1,'stationCode':1, '_id': 0})
        df = pd.json_normalize(list(cursor), sep='_')

        # Extract required fields from dataItemMap
        df['timestamp'] = df['timestamp']
        df['inverter_power'] = df['dataItemMap_inverter_power']
        df['radiation_intensity'] = df['dataItemMap_radiation_intensity']
        df['inverter_power'] = df['inverter_power'].round(2)  # Round to 2 decimal places
        df['radiation_intensity'] = df['radiation_intensity'].round(2)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # Preprocess and predict to add 'Predicted Power' column
        processed_data = preprocess_and_predict(df)

        # Aggregate daily data
        df_daily_inverter_power = processed_data.groupby('Date')['inverter_power'].sum().reset_index()
        df_daily_predicted_power = processed_data.groupby('Date')['Predicted Power'].sum().reset_index()

        df_daily = df_daily_inverter_power.merge(df_daily_predicted_power, on='Date', how='left')
        df_daily['Power Ratio (%)'] = (df_daily['inverter_power'] / df_daily['Predicted Power']) * 100

        # Prepare output data
        response_data = {
            'dates': df_daily['Date'].astype(str).tolist(),
            'power_ratios': df_daily['Power Ratio (%)'].tolist()
        }

        return jsonify({'status': 'success', 'data': response_data})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/mae_mape', methods=['POST'])
def mae_mape():
    try:
        # Get the request payload
        payload = request.get_json()
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')

        # Fetch data from MongoDB
        query = {'timestamp': {'$gte': start_date, '$lte': end_date}}
        cursor = collection.find(query, {'timestamp': 1, 'dataItemMap': 1, '_id': 0})
        df = pd.json_normalize(list(cursor), sep='_')

        # Extract required fields from dataItemMap
        df['timestamp'] = df['timestamp']
        df['inverter_power'] = df['dataItemMap_inverter_power']
        df['radiation_intensity'] = df['dataItemMap_radiation_intensity']
        df['inverter_power'] = df['inverter_power'].round(2)  # Round to 2 decimal places
        df['radiation_intensity'] = df['radiation_intensity'].round(2)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # Preprocess and predict to add 'Predicted Power' column
        processed_data = preprocess_and_predict(df)

        # Aggregate daily data
        df_daily = processed_data.groupby('Date').agg({
            'inverter_power': 'sum',
            'Predicted Power': 'sum'
        }).reset_index()

        df_daily['MAE'] = abs(df_daily['Predicted Power'] - df_daily['inverter_power'])
        df_daily['MAPE'] = (abs(df_daily['Predicted Power'] - df_daily['inverter_power']) / df_daily['inverter_power']) * 100

        # Prepare output data
        response_data = {
            'dates': df_daily['Date'].astype(str).tolist(),
            'mae': df_daily['MAE'].tolist(),
            'mape': df_daily['MAPE'].tolist()
        }

        return jsonify({'status': 'success', 'data': response_data})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/score_vs_suppression', methods=['POST'])
def score_vs_suppression():
    try:
        # Get the request payload
        payload = request.get_json()
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        plant = payload.get('plant')
        tarrif = payload.get('tarrif')
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M:%S')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d %H:%M:%S')

        # Fetch data from MongoDB
        query = {'timestamp': {'$gte': start_date, '$lte': end_date}, "stationCode": plant}
        cursor = collection.find(query, {'timestamp': 1, 'dataItemMap': 1, 'stationCode': 1, '_id': 0})
        df = pd.json_normalize(list(cursor), sep='_')

        # Validate fetched dataddd
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data found for the given query'})

        # Extract required fields
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['inverter_power'] = df['dataItemMap_inverter_power']
        df['radiation_intensity'] = df['dataItemMap_radiation_intensity']
        df['inverter_power'] = df['inverter_power'].round(2)  # Round to 2 decimal places
        df['radiation_intensity'] = df['radiation_intensity'].round(2)

        # Preprocess and predict 'Predicted Power'
        processed_data = preprocess_and_predict(df)
        processed_data['MAE'] = np.abs(processed_data['Predicted Power'] - processed_data['inverter_power'])

        # Normalize MAE values for GMM
        X = processed_data['MAE'].values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Optimize GMM with Optuna
        def objective(trial, X_scaled):
            n_components = trial.suggest_int("n_components", 2, 2)
            covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
            gmm_model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
            gmm_model.fit(X_scaled)
            return gmm_model.bic(X_scaled)

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, X_scaled), n_trials=150)

        # Fit the GMM with best parameters
        best_params = study.best_params
        gmm_model = GaussianMixture(n_components=best_params["n_components"],
                                    covariance_type=best_params["covariance_type"],
                                    random_state=42)
        gmm_model.fit(X_scaled)

        # Detect anomalies using log-likelihood
        log_likelihood = gmm_model.score_samples(X_scaled)
        def optimize_threshold(log_likelihood):
            thresholds = np.linspace(np.min(log_likelihood), np.max(log_likelihood), 1000)
            best_threshold = thresholds[np.argmin(np.abs(thresholds - np.median(log_likelihood)))]  # Choose threshold around median
            return best_threshold
        best_threshold = optimize_threshold(log_likelihood)
        relaxation_factor = 1.0
        processed_data['is_anomaly'] = log_likelihood < (best_threshold * relaxation_factor)
        filter_dates = processed_data.loc[processed_data['is_anomaly'] == True, 'Date'].tolist()
        filter_dates = pd.to_datetime(filter_dates)                             # Ensure filter_dates is a datetime64[ns] series
        processed_data['Date'] = pd.to_datetime(processed_data['Date'])                     # Ensure df_daily['Date'] is in datetime64[ns] format
        df_new = processed_data[~processed_data['Date'].isin(filter_dates)] 
        # filtered_data = processed_data[~processed_data['Date'].isin(filter_dates)]

        # Calculate median MAE
        median_mae = df_new['MAE'].median()
        processed_data['Median MAE']= median_mae
        processed_data['Suppression'] = (processed_data['Predicted Power'] - processed_data['inverter_power'] - median_mae).clip(lower=0)

        # Aggregate daily data
        df_daily = processed_data.groupby(processed_data['timestamp'].dt.date).agg({
            'Suppression': 'sum',
            'MAE': 'mean'
        }).reset_index()
        df_daily.rename(columns={'timestamp': 'Date'}, inplace=True)
        df_daily['tarrif'] = tarrif
        # Prepare chart data in the desired format
        chart_data = {
            "dates": df_daily['Date'].astype(str).tolist(),
            "total_suppression": df_daily['Suppression'].fillna(0).tolist(),
            "loss": (df_daily['tarrif'] * df_daily['Suppression'].fillna(0)).tolist(),  # Convert Series to list
        }


        return jsonify({'status': 'success', 'chart_data': chart_data})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/chart_data', methods=['POST'])
def get_chart_data():
    try:
        # Get the request payload
        payload = request.get_json()
        datelist = payload.get('date')  # Input date as string (e.g., '2024-12-15')
        top_n = payload.get('top_n', 3)  # Number of similar dates to retrieve (default is 3)

        # Convert input date to proper datetime.date format
        input_date = pd.to_datetime(datelist).date()

        # Fetch data from MongoDB
        cursor = collection.find({"stationCode":"NE=53278269"}, {'timestamp': 1, 'dataItemMap': 1, '_id': 0,'stationCode':1})
        df = pd.json_normalize(list(cursor), sep='_')

        # Extract required fields
        df['timestamp'] = df['timestamp']
        df['inverter_power'] = df['dataItemMap_inverter_power']
        df['radiation_intensity'] = df['dataItemMap_radiation_intensity']
        df['inverter_power'] = df['inverter_power'].round(2)  # Round to 2 decimal places
        df['radiation_intensity'] = df['radiation_intensity'].round(2)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # Preprocess and predict to add 'Predicted Power'
        processed_data = preprocess_and_predict(df)

        # Group data by date
        grouped = processed_data.groupby('Date')

        # Check if input_date is in the data
        if input_date not in processed_data['Date'].unique():
            return jsonify({'status': 'error', 'message': f"The provided date {datelist} is not in the dataset."})

        # Extract radiation intensity for each date
        radiation_series = {
            date: group.sort_values('Hour')['radiation_intensity'].values for date, group in grouped
        }

        # Use DTW to find similar days (hour-on-hour basis)
        target_series = radiation_series[input_date]
        distances = {
            date: dtw.distance(target_series, series)
            for date, series in radiation_series.items() if date != input_date
        }

        # Find top similar dates
        similar_dates = sorted(distances, key=distances.get)[:top_n]
        dtw_scores = {date: distances[date] for date in similar_dates}

        # Prepare data for charts
        chart_data = {
            'radiation_intensity': [],
            'generated_power': [],
            'predicted_power': []
        }

        # Populate chart data for the input date and similar dates
        for date in [input_date] + similar_dates:
            daily_data = grouped.get_group(date).sort_values('Hour')
            chart_data['radiation_intensity'].append({
                'date': str(date),
                'score': dtw_scores.get(date, 0),
                'hours': daily_data['Hour'].tolist(),
                'values': daily_data['radiation_intensity'].tolist()
            })
            chart_data['generated_power'].append({
                'date': str(date),
                'score': dtw_scores.get(date, 0),
                'hours': daily_data['Hour'].tolist(),
                'values': daily_data['inverter_power'].tolist()
            })
            chart_data['predicted_power'].append({
                'date': str(date),
                'score': dtw_scores.get(date, 0),
                'hours': daily_data['Hour'].tolist(),
                'values': daily_data['Predicted Power'].tolist()
            })

        # Return chart data
        return jsonify({'status': 'success', 'chart_data': chart_data})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/get_dash_active_stat_data', methods=['POST'])
def get_dash_active_stat_data():
    payload = request.json
    option = payload.get("option")

    if option not in [1, 2, 3]:
        return jsonify({"error": "Invalid option. Only options 1, 2, and 3 are supported."}), 400

    if option == 1:
        # Option 1: Get Active Power Sum for Yesterday Only
        yesterday = datetime.now() - timedelta(days=1)
        print(yesterday)
        yesterday_str = yesterday.strftime('%Y-%m-%d')

        pipeline = [
            # Match documents where Day equals yesterday_str
            {"$match": {"Day": yesterday_str}},
            
            # Group by a constant value (e.g., "date") and sum active_power
            {
                "$group": {
                    "_id": None,  # We don't need a specific grouping key since it's for a single day
                    "active_power_sum": {"$sum": "$active_power"}
                }
            }
        ]

        results = list(GM_Day.aggregate(pipeline))

        # Use yesterday_str as the key for the response
        response = {
            yesterday_str: round(results[0]["active_power_sum"]) if results else 0
        }

        return jsonify(response)

    elif option == 2:
        # Option 2: Group by month and remove Last Month
        now = datetime.now()
        first_day_current_month = datetime(now.year, now.month, 1)

        pipeline = [
            {
                "$match": {
                    "Day": {"$gte": first_day_current_month.strftime('%Y-%m-%d')}
                }
            },
            {
                "$addFields": {
                    "month": {"$substr": ["$Day", 0, 7]}
                }
            },
            {
                "$group": {
                    "_id": {"month": "$month"},
                    "active_power_sum": {"$sum": "$active_power"}
                }
            },
            {"$sort": {"_id.month": 1}}
        ]

        results = list(GM_Day.aggregate(pipeline))
        response = {
            record["_id"]["month"]: round(record["active_power_sum"]) for record in results
        }

        return jsonify(response)

    elif option == 3:
        # Option 3: Group by year and remove Last Year
        current_year = datetime.now().year

        pipeline = [
            {
                "$match": {
                    "Day": {"$regex": f"^{current_year}"}
                }
            },
            {
                "$addFields": {
                    "year": {"$substr": ["$Day", 0, 4]}
                }
            },
            {
                "$group": {
                    "_id": {"year": "$year"},
                    "active_power_sum": {"$sum": "$active_power"}
                }
            },
            {"$sort": {"_id.year": 1}}
        ]

        results = list(GM_Day.aggregate(pipeline))
        response = {
            record["_id"]["year"]: round(record["active_power_sum"]) for record in results
        }

        return jsonify(response)

@app.route('/get_dash_stat_data', methods=['POST'])
def get_dash_stat_data():
    payload = request.json
    option = payload.get("option")

    if option not in [1, 2, 3]:
        return jsonify({"error": "Invalid option. Only options 1, 2, and 3 are supported."}), 400

    if option == 1:
        # Option 1: Get data for Yesterday
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_str = yesterday.strftime('%Y-%m-%d')

        pipeline = [
            {"$match": {"Day_Hour": {"$regex": f"^{yesterday_str}"}}},
            {
                "$group": {
                    "_id": None,  # Single group since it's for a single day
                    "active_power_sum": {"$sum": "$P_abd"}
                }
            }
        ]

        results = list(device_hour_collection.aggregate(pipeline))
        kw = round(results[0]["active_power_sum"]) if results else 0
        capacity = 2400  # Fixed capacity for option 1
        predictedco2 = round(capacity * 0.56)
        actualco2 = round(kw * 0.56)

        response = {
            "kw": kw,
            "capacity": capacity,
            "predictedco2": predictedco2,
            "actualco2": actualco2
        }

        return jsonify(response)

    elif option == 2:
        # Option 2: Get data for the current month and calculate capacity based on unique days
        now = datetime.now()
        first_day_current_month = datetime(now.year, now.month, 1)

        pipeline = [
            {"$match": {"Day": {"$gte": first_day_current_month.strftime('%Y-%m-%d')}}},
            {
                "$addFields": {
                    "month": {"$substr": ["$Day", 0, 7]}
                }
            },
            {
                "$group": {
                    "_id": {"month": "$month"},
                    "active_power_sum": {"$sum": "$P_abd"},
                    "unique_days": {"$addToSet": "$Day"}  # Collect unique days
                }
            },
            {"$sort": {"_id.month": 1}}
        ]

        results = list(device_day_collection.aggregate(pipeline))
        kw = round(results[0]["active_power_sum"]) if results else 0
        unique_days_count = len(results[0]["unique_days"]) if results else 0  # Count unique days
        capacity = unique_days_count * 2400
        predictedco2 = round(capacity * 0.56)
        actualco2 = round(kw * 0.56)

        response = {
            "kw": kw,
            "capacity": capacity,
            "predictedco2": predictedco2,
            "actualco2": actualco2
        }

        return jsonify(response)

    elif option == 3:
        # Option 3: Get data for the current year and calculate capacity based on unique days
        current_year = datetime.now().year

        pipeline = [
            {"$match": {"Day": {"$regex": f"^{current_year}"}}},
            {
                "$addFields": {
                    "year": {"$substr": ["$Day", 0, 4]}
                }
            },
            {
                "$group": {
                    "_id": {"year": "$year"},
                    "active_power_sum": {"$sum": "$P_abd"},
                    "unique_days": {"$addToSet": "$Day"}  # Collect unique days
                }
            },
            {"$sort": {"_id.year": 1}}
        ]

        results = list(device_day_collection.aggregate(pipeline))
        kw = round(results[0]["active_power_sum"]) if results else 0
        unique_days_count = len(results[0]["unique_days"]) if results else 0  # Count unique days
        capacity = unique_days_count * 2400
        predictedco2 = round(capacity * 0.56)
        actualco2 = round(kw * 0.56)

        response = {
            "kw": kw,
            "capacity": capacity,
            "predictedco2": predictedco2,
            "actualco2": actualco2
        }

        return jsonify(response)

# @app.route('/get_dash_stat_data', methods=['POST'])
# def get_dash_stat_data():
#     payload = request.json
#     option = payload.get("option")

#     if option not in [1, 2, 3]:
#         return jsonify({"error": "Invalid option. Only options 1, 2, and 3 are supported."}), 400

#     # Fixed year 2024
#     fixed_year = 2024

#     if option == 1:
#         # Option 1: Get data for Yesterday (from 2024)
#         yesterday = datetime(fixed_year, datetime.now().month, datetime.now().day) - timedelta(days=1)
#         yesterday_str = yesterday.strftime('%Y-%m-%d')

#         pipeline = [
#             {"$match": {"Day_Hour": {"$regex": f"^{yesterday_str}"}}},  # Match the date for yesterday in 2024
#             {
#                 "$group": {
#                     "_id": None,  # Single group since it's for a single day
#                     "active_power_sum": {"$sum": "$P_abd"}
#                 }
#             }
#         ]

#         results = list(device_hour_collection.aggregate(pipeline))
#         kw = round(results[0]["active_power_sum"]) if results else 0
#         capacity = 2400  # Fixed capacity for option 1
#         predictedco2 = round(capacity * 0.56)
#         actualco2 = round(kw * 0.56)

#         response = {
#             "kw": kw,
#             "capacity": capacity,
#             "predictedco2": predictedco2,
#             "actualco2": actualco2
#         }

#         return jsonify(response)

#     elif option == 2:
#         # Option 2: Get data for the current month (from 2024) and calculate capacity based on unique days
#         first_day_current_month = datetime(fixed_year, datetime.now().month, 1)

#         pipeline = [
#             {"$match": {"Day": {"$gte": first_day_current_month.strftime('%Y-%m-%d')}}},  # Match data from 2024
#             {
#                 "$addFields": {
#                     "month": {"$substr": ["$Day", 0, 7]}
#                 }
#             },
#             {
#                 "$group": {
#                     "_id": {"month": "$month"},
#                     "active_power_sum": {"$sum": "$P_abd"},
#                     "unique_days": {"$addToSet": "$Day"}  # Collect unique days
#                 }
#             },
#             {"$sort": {"_id.month": 1}}
#         ]

#         results = list(device_day_collection.aggregate(pipeline))
#         kw = round(results[0]["active_power_sum"]) if results else 0
#         unique_days_count = len(results[0]["unique_days"]) if results else 0  # Count unique days
#         capacity = unique_days_count * 2400
#         predictedco2 = round(capacity * 0.56)
#         actualco2 = round(kw * 0.56)

#         response = {
#             "kw": kw,
#             "capacity": capacity,
#             "predictedco2": predictedco2,
#             "actualco2": actualco2
#         }

#         return jsonify(response)

#     elif option == 3:
#         # Option 3: Get data for the current year (2024) and calculate capacity based on unique days
#         pipeline = [
#             {"$match": {"Day": {"$regex": f"^{fixed_year}"}}},  # Match data for year 2024
#             {
#                 "$addFields": {
#                     "year": {"$substr": ["$Day", 0, 4]}
#                 }
#             },
#             {
#                 "$group": {
#                     "_id": {"year": "$year"},
#                     "active_power_sum": {"$sum": "$P_abd"},
#                     "unique_days": {"$addToSet": "$Day"}  # Collect unique days
#                 }
#             },
#             {"$sort": {"_id.year": 1}}
#         ]

#         results = list(device_day_collection.aggregate(pipeline))
#         kw = round(results[0]["active_power_sum"]) if results else 0
#         unique_days_count = len(results[0]["unique_days"]) if results else 0  # Count unique days
#         capacity = unique_days_count * 2400
#         predictedco2 = round(capacity * 0.56)
#         actualco2 = round(kw * 0.56)

#         response = {
#             "kw": kw,
#             "capacity": capacity,
#             "predictedco2": predictedco2,
#             "actualco2": actualco2
#         }

#         return jsonify(response)


@app.route('/get_dash_cost_data', methods=['POST'])
def get_dash_cost_data():
    payload = request.json
    option = payload.get("option")

    if option not in [1, 2, 3]:
        return jsonify({"error": "Invalid option. Only options 1, 2, and 3 are supported."}), 400

    if option == 1:
        # Option 1: Group by date for Yesterday and Day Before Yesterday
        yesterday = datetime.now() - timedelta(days=1)
        day_before_yesterday = datetime.now() - timedelta(days=2)

        yesterday_str = yesterday.strftime('%Y-%m-%d')
        day_before_yesterday_str = day_before_yesterday.strftime('%Y-%m-%d')

        queries = [
            {"Day_Hour": {"$regex": f"^{yesterday_str}"}},
            {"Day_Hour": {"$regex": f"^{day_before_yesterday_str}"}}
        ]

        pipeline = [
            {"$match": {"$or": queries}},
            {
                "$group": {
                    "_id": {"date": {"$substr": ["$Day_Hour", 0, 10]}},
                    "active_power_sum": {"$sum": "$P_abd"}
                }
            },
            {"$sort": {"_id.date": 1}}
        ]

        results = list(device_hour_collection.aggregate(pipeline))
        data_by_date = {record["_id"]["date"]: round(record["active_power_sum"]) for record in results}

        labels = [yesterday_str, day_before_yesterday_str]
        data = [data_by_date.get(date, 0) for date in labels]

        response = {
            "labels": labels,
            "datasets": [
                {
                    "label": "Active Power",
                    "data": data,
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    # "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": True,
                    "tension": 0.4,
                    "borderWidth": 3
                }
            ]
        }

        return jsonify(response)

    elif option == 2:
        # Option 2: Group by month for This Month and Last Month
        now = datetime.now()
        first_day_current_month = datetime(now.year, now.month, 1)
        last_day_last_month = first_day_current_month - timedelta(days=1)
        first_day_last_month = datetime(last_day_last_month.year, last_day_last_month.month, 1)

        pipeline = [
            {
                "$match": {
                    "Day": {
                        "$gte": first_day_last_month.strftime('%Y-%m-%d'),
                        "$lte": now.strftime('%Y-%m-%d')
                    }
                }
            },
            {
                "$addFields": {
                    "month": {"$substr": ["$Day", 0, 7]}
                }
            },
            {
                "$group": {
                    "_id": {"month": "$month"},
                    "active_power_sum": {"$sum": "$P_abd"}
                }
            },
            {"$sort": {"_id.month": 1}}
        ]

        results = list(device_day_collection.aggregate(pipeline))
        data_by_month = {record["_id"]["month"]: round(record["active_power_sum"]) for record in results}

        current_month_str = first_day_current_month.strftime('%Y-%m')
        last_month_str = first_day_last_month.strftime('%Y-%m')

        labels = [
            first_day_last_month.strftime('%B %Y'),
            first_day_current_month.strftime('%B %Y')
        ]

        data = [data_by_month.get(last_month_str, 0), data_by_month.get(current_month_str, 0)]

        response = {
            "labels": labels,
            "datasets": [
                {
                    "label": "Active Power",
                    "data": data,
                    "backgroundColor": "rgba(75, 192, 192, 0.4)",
                    # "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": True,
                    "tension": 0.4
                }
            ]
        }

        return jsonify(response)

    elif option == 3:
        # Option 3: Group by year for This Year and Last Year
        current_year = datetime.now().year
        last_year = current_year - 1

        pipeline = [
            {
                "$match": {
                    "Day": {"$exists": True, "$ne": None}
                }
            },
            {
                "$addFields": {
                    "year": {"$substr": ["$Day", 0, 4]}
                }
            },
            {
                "$group": {
                    "_id": {"year": "$year"},
                    "active_power_sum": {"$sum": "$P_abd"}
                }
            },
            {"$sort": {"_id.year": 1}}
        ]

        results = list(device_day_collection.aggregate(pipeline))
        data_by_year = {record["_id"]["year"]: round(record["active_power_sum"]) for record in results}

        labels = [str(last_year), str(current_year)]
        data = [data_by_year.get(str(last_year), 0), data_by_year.get(str(current_year), 0)]

        response = {
            "labels": labels,
            "datasets": [
                {
                    "label": "Active Power",
                    "data": data,
                    "backgroundColor": "rgba(75, 192, 192, 0.4)",
                    # "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": True,
                    "tension": 0.4
                }
            ]
        }

        return jsonify(response)

@app.route('/get_dash_column_data', methods=['POST'])
def get_dash_column_data():
    payload = request.json
    option = payload.get("option")

    if option not in [1, 2, 3]:
        return jsonify({"error": "Invalid option. Only options 1, 2, and 3 are supported."}), 400

    if option == 1:
        # Option 1: Yesterday
        yesterday = datetime.now() - timedelta(days=1)
        print('----',yesterday)
        yesterday_str = yesterday.strftime('%Y-%m-%d')

        pipeline = [
            {"$match": {"Day_Hour": {"$regex": f"^{yesterday_str}"}}},
            {
                "$group": {
                    "_id": {
                        "plant": "$Plant",
                        "day_hour": "$Day_Hour"
                    },
                    "active_power_sum": {"$sum": "$P_abd"}
                }
            },
            {"$sort": {"_id.day_hour": 1}}
        ]

        results = list(device_hour_collection.aggregate(pipeline))
        yesterday_data = []
        all_hours = set()

        for record in results:
            plant = record["_id"]["plant"]
            day_hour = record["_id"]["day_hour"]
            active_power_sum = record["active_power_sum"]

            hour = int(day_hour.split(" ")[1])
            all_hours.add(hour)

            yesterday_data.append({"hour": hour, "value": active_power_sum, "plant": plant})

        sorted_hours = sorted(all_hours)
        labels = [f"Hour {hour}" for hour in sorted_hours]
        generation_data = [d["value"] for d in sorted(yesterday_data, key=lambda x: x["hour"])]
        cost_data = [value * 60 for value in generation_data]  # Calculate cost as generation * 60

        response = {
            "labels": labels,
            "datasets": [
                {
                    "type": "bar",
                    "label": "Power (KW)",
                    "data": generation_data,
                    "backgroundColor": "rgba(75, 192, 192, 0.4)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": True,
                    "tension": 0.4
                },
                {
                    "type": "line",
                    "label": "Cost (PKR)",
                    "data": cost_data,
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "fill": False,
                    "tension": 0.4,
                    "yAxisID": "y-axis-cost"
                }
            ]
        }

        return jsonify(response)

    elif option == 2:
        # Option 2: Group Current Month by Weeks 1 to 4
        now = datetime.now()
        current_month = now.month
        current_year = now.year

        pipeline = [
            {
                "$match": {
                    "Day": {"$exists": True, "$ne": None}
                }
            },
            {
                "$addFields": {
                    "parsedDate": {
                        "$dateFromString": {
                            "dateString": "$Day",
                            "format": "%Y-%m-%d"
                        }
                    }
                }
            },
            {
                "$match": {  # Match current month and year
                    "$expr": {
                        "$and": [
                            {"$eq": [{"$month": "$parsedDate"}, current_month]},
                            {"$eq": [{"$year": "$parsedDate"}, current_year]}
                        ]
                    }
                }
            },
            {
                "$addFields": {
                    "week": {
                        "$ceil": {"$divide": [{"$dayOfMonth": "$parsedDate"}, 7]}
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "week": "$week"
                    },
                    "active_power_sum": {"$sum": "$P_abd"}
                }
            },
            {
                "$sort": {"_id.week": 1}
            }
        ]

        results = list(device_day_collection.aggregate(pipeline))

        # Process results into datasets
        current_month_data = [0, 0, 0, 0]  # Initialize for weeks 1 to 4

        for record in results:
            week = int(record["_id"]["week"])  # Explicitly convert to integer
            active_power_sum = record["active_power_sum"]
            if 1 <= week <= 4:  # Ensure valid week numbers
                current_month_data[week - 1] += active_power_sum

        cost_data = [value * 60 for value in current_month_data]  # Calculate cost as generation * 60

        # Prepare response
        labels = ["Week 1", "Week 2", "Week 3", "Week 4"]

        response = {
            "labels": labels,
            "datasets": [
                {
                    "type": "bar",
                    "label": "Power (KW)",
                    "data": current_month_data,
                    "backgroundColor": "rgba(75, 192, 192, 0.4)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": True,
                    "tension": 0.4
                },
                {
                    "type": "line",
                    "label": "Cost (PKR)",
                    "data": cost_data,
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "fill": False,
                    "tension": 0.4,
                    "yAxisID": "y-axis-cost"
                }
            ]
        }

        return jsonify(response)

    elif option == 3:
        # Option 3: Group Current Year by Quarters
        current_year = datetime.now().year

        pipeline = [
            {
                "$match": {
                    "Day": {"$exists": True, "$ne": None}
                }
            },
            {
                "$addFields": {
                    "parsedDate": {
                        "$dateFromString": {
                            "dateString": "$Day",
                            "format": "%Y-%m-%d"
                        }
                    }
                }
            },
            {
                "$match": {  # Match current year
                    "$expr": {
                        "$eq": [{"$year": "$parsedDate"}, current_year]
                    }
                }
            },
            {
                "$addFields": {
                    "quarter": {
                        "$switch": {
                            "branches": [
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 3]}, "then": 1},
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 6]}, "then": 2},
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 9]}, "then": 3},
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 12]}, "then": 4}
                            ],
                            "default": 4
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "quarter": "$quarter"
                    },
                    "active_power_sum": {"$sum": "$P_abd"}
                }
            },
            {
                "$sort": {"_id.quarter": 1}
            }
        ]

        results = list(device_day_collection.aggregate(pipeline))

        # Initialize quarter data for the current year
        quarter_data_current_year = [0, 0, 0, 0]  # For Q1 to Q4

        for record in results:
            quarter = record["_id"]["quarter"]
            active_power_sum = record["active_power_sum"]
            quarter_data_current_year[quarter - 1] += active_power_sum

        cost_data = [value * 60 for value in quarter_data_current_year]  # Calculate cost as generation * 60

        # Labels for quarters
        labels = ["Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4"]

        # Prepare response
        response = {
            "labels": labels,
            "datasets": [
                {
                    "type": "bar",
                    "label": "Power (KW)",
                    "data": quarter_data_current_year,
                    "backgroundColor": "rgba(75, 192, 192, 0.4)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": True,
                    "tension": 0.4
                },
                {
                    "type": "line",
                    "label": "Cost (PKR)",
                    "data": cost_data,
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "fill": False,
                    "tension": 0.4,
                    "yAxisID": "y-axis-cost"
                }
            ]
        }

        return jsonify(response)

        # Option 3: Group Current Year by Quarters
        current_year = datetime.now().year

        pipeline = [
            {
                "$match": {
                    "Day": {"$exists": True, "$ne": None}
                }
            },
            {
                "$addFields": {
                    "parsedDate": {
                        "$dateFromString": {
                            "dateString": "$Day",
                            "format": "%Y-%m-%d"
                        }
                    }
                }
            },
            {
                "$addFields": {
                    "quarter": {
                        "$switch": {
                            "branches": [
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 3]}, "then": 1},
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 6]}, "then": 2},
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 9]}, "then": 3},
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 12]}, "then": 4}
                            ],
                            "default": 4
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "quarter": "$quarter"
                    },
                    "active_power_sum": {"$sum": "$P_abd"}
                }
            },
            {
                "$sort": {"_id.quarter": 1}
            }
        ]

        results = list(device_day_collection.aggregate(pipeline))
        print(results)

        # Initialize quarter data for the current year
        quarter_data_current_year = [0, 0, 0, 0]  # For Q1 to Q4

        for record in results:
            quarter = record["_id"]["quarter"]
            active_power_sum = record["active_power_sum"]
            quarter_data_current_year[quarter - 1] += active_power_sum

        cost_data = [value * 60 for value in quarter_data_current_year]  # Calculate cost as generation * 60

        # Labels for quarters
        labels = ["Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4"]

        # Prepare response
        response = {
            "labels": labels,
            "datasets": [
                {
                    "type": "bar",
                    "label": "Power (KW)",
                    "data": quarter_data_current_year,
                    "backgroundColor": "rgba(75, 192, 192, 0.4)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": True,
                    "tension": 0.4
                },
                {
                    "type": "line",
                    "label": "Cost (PKR)",
                    "data": cost_data,
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "fill": False,
                    "tension": 0.4,
                    "yAxisID": "y-axis-cost"
                }
            ]
        }

        return jsonify(response)

# @app.route('/get_dash_column_data', methods=['POST'])
# def get_dash_column_data():
#     payload = request.json
#     option = payload.get("option")

#     if option not in [1, 2, 3]:
#         return jsonify({"error": "Invalid option. Only options 1, 2, and 3 are supported."}), 400

#     if option == 1:
#         # Option 1: Yesterday
#         yesterday = datetime.now() - timedelta(days=1)
#         print('----',yesterday)
#         yesterday_str = yesterday.strftime('%Y-%m-%d')

#         pipeline = [
#             {"$match": {"Day_Hour": {"$regex": f"^{yesterday_str}"}}},
#             {
#                 "$group": {
#                     "_id": {
#                         "plant": "$Plant",
#                         "day_hour": "$Day_Hour"
#                     },
#                     "active_power_sum": {"$sum": "$P_abd"}
#                 }
#             },
#             {"$sort": {"_id.day_hour": 1}}
#         ]

#         results = list(device_hour_collection.aggregate(pipeline))
#         yesterday_data = []
#         all_hours = set()

#         for record in results:
#             plant = record["_id"]["plant"]
#             day_hour = record["_id"]["day_hour"]
#             active_power_sum = record["active_power_sum"]

#             hour = int(day_hour.split(" ")[1])
#             all_hours.add(hour)

#             yesterday_data.append({"hour": hour, "value": active_power_sum, "plant": plant})

#         sorted_hours = sorted(all_hours)
#         labels = [f"Hour {hour}" for hour in sorted_hours]
#         generation_data = [d["value"] for d in sorted(yesterday_data, key=lambda x: x["hour"])]
#         cost_data = [value * 60 for value in generation_data]  # Calculate cost as generation * 60

#         response = {
#             "labels": labels,
#             "datasets": [
#                 {
#                     "type": "bar",
#                     "label": "Power (KW)",
#                     "data": generation_data,
#                     "backgroundColor": "rgba(75, 192, 192, 0.4)",
#                     "borderColor": "rgba(75, 192, 192, 1)",
#                     "fill": True,
#                     "tension": 0.4
#                 },
#                 {
#                     "type": "line",
#                     "label": "Cost (PKR)",
#                     "data": cost_data,
#                     "borderColor": "rgba(255, 99, 132, 1)",
#                     "backgroundColor": "rgba(255, 99, 132, 0.2)",
#                     "fill": False,
#                     "tension": 0.4,
#                     "yAxisID": "y-axis-cost"
#                 }
#             ]
#         }

#         return jsonify(response)

#     elif option == 2:
#         # Option 2: Fixed labels and data
#         labels = ["Week 1", "Week 2", "Week 3", "Week 4"]
#         generation_data = [42660, 7184, 14291, 27675]  # Bar chart data (Power in KW)
#         cost_data = [value * 60 for value in generation_data]  # Line chart data (Cost in PKR)

#         response = {
#             "labels": labels,
#             "datasets": [
#                 {
#                     "type": "bar",
#                     "label": "Power (KW)",
#                     "data": generation_data,
#                     "backgroundColor": "rgba(75, 192, 192, 0.4)",
#                     "borderColor": "rgba(75, 192, 192, 1)",
#                     "fill": True,
#                     "tension": 0.4
#                 },
#                 {
#                     "type": "line",
#                     "label": "Cost (PKR)",
#                     "data": cost_data,
#                     "borderColor": "rgba(255, 99, 132, 1)",
#                     "backgroundColor": "rgba(255, 99, 132, 0.2)",
#                     "fill": False,
#                     "tension": 0.4,
#                     "yAxisID": "y-axis-cost"
#                 }
#             ]
#         }

#         return jsonify(response)

#     elif option == 3:
#         # Option 3: Group Current Year by Quarters
#         current_year = datetime.now().year

#         pipeline = [
#             {
#                 "$match": {
#                     "Day": {"$exists": True, "$ne": None}
#                 }
#             },
#             {
#                 "$addFields": {
#                     "parsedDate": {
#                         "$dateFromString": {
#                             "dateString": "$Day",
#                             "format": "%Y-%m-%d"
#                         }
#                     }
#                 }
#             },
#             {
#                 "$addFields": {
#                     "quarter": {
#                         "$switch": {
#                             "branches": [
#                                 {"case": {"$lte": [{"$month": "$parsedDate"}, 3]}, "then": 1},
#                                 {"case": {"$lte": [{"$month": "$parsedDate"}, 6]}, "then": 2},
#                                 {"case": {"$lte": [{"$month": "$parsedDate"}, 9]}, "then": 3},
#                                 {"case": {"$lte": [{"$month": "$parsedDate"}, 12]}, "then": 4}
#                             ],
#                             "default": 4
#                         }
#                     }
#                 }
#             },
#             {
#                 "$group": {
#                     "_id": {
#                         "quarter": "$quarter"
#                     },
#                     "active_power_sum": {"$sum": "$P_abd"}
#                 }
#             },
#             {
#                 "$sort": {"_id.quarter": 1}
#             }
#         ]

#         results = list(device_day_collection.aggregate(pipeline))
#         print(results)

#         # Initialize quarter data for the current year
#         quarter_data_current_year = [0, 0, 0, 0]  # For Q1 to Q4

#         for record in results:
#             quarter = record["_id"]["quarter"]
#             active_power_sum = record["active_power_sum"]
#             quarter_data_current_year[quarter - 1] += active_power_sum

#         cost_data = [value * 60 for value in quarter_data_current_year]  # Calculate cost as generation * 60

#         # Labels for quarters
#         labels = ["Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4"]

#         # Prepare response
#         response = {
#             "labels": labels,
#             "datasets": [
#                 {
#                     "type": "bar",
#                     "label": "Power (KW)",
#                     "data": quarter_data_current_year,
#                     "backgroundColor": "rgba(75, 192, 192, 0.4)",
#                     "borderColor": "rgba(75, 192, 192, 1)",
#                     "fill": True,
#                     "tension": 0.4
#                 },
#                 {
#                     "type": "line",
#                     "label": "Cost (PKR)",
#                     "data": cost_data,
#                     "borderColor": "rgba(255, 99, 132, 1)",
#                     "backgroundColor": "rgba(255, 99, 132, 0.2)",
#                     "fill": False,
#                     "tension": 0.4,
#                     "yAxisID": "y-axis-cost"
#                 }
#             ]
#         }

#         return jsonify(response)



@app.route('/get_dash_data', methods=['POST'])
def get_dash_data():
    payload = request.json
    option = payload.get("option")

    if option not in [1, 2, 3]:
        return jsonify({"error": "Invalid option. Only options 1, 2, and 3 are supported."}), 400

    if option == 1:
        # Option 1: Yesterday and Day Before Yesterday
        yesterday = datetime.now() - timedelta(days=1)
        day_before_yesterday = datetime.now() - timedelta(days=2)

        yesterday_str = yesterday.strftime('%Y-%m-%d')
        day_before_yesterday_str = day_before_yesterday.strftime('%Y-%m-%d')

        queries = [
            {"Day_Hour": {"$regex": f"^{yesterday_str}"}},
            {"Day_Hour": {"$regex": f"^{day_before_yesterday_str}"}}
        ]

        pipeline = [
            {"$match": {"$or": queries}},
            {
                "$group": {
                    "_id": {
                        "plant": "$Plant",
                        "day_hour": "$Day_Hour"
                    },
                    "active_power_sum": {"$sum": "$P_abd"}
                }
            },
            {"$sort": {"_id.day_hour": 1}}
        ]

        results = list(device_hour_collection.aggregate(pipeline))
        yesterday_data = []
        day_before_yesterday_data = []
        all_hours = set()

        for record in results:
            plant = record["_id"]["plant"]
            day_hour = record["_id"]["day_hour"]
            active_power_sum = record["active_power_sum"]

            hour = int(day_hour.split(" ")[1])
            all_hours.add(hour)

            if day_hour.startswith(yesterday_str):
                yesterday_data.append({"hour": hour, "value": active_power_sum, "plant": plant})
            elif day_hour.startswith(day_before_yesterday_str):
                day_before_yesterday_data.append({"hour": hour, "value": active_power_sum, "plant": plant})

        sorted_hours = sorted(all_hours)
        labels = [f"Hour {hour}" for hour in sorted_hours]

        response = {
            "labels": labels,
            "datasets": [
                {
                    "label": yesterday_str,
                    "data": [d["value"] for d in sorted(yesterday_data, key=lambda x: x["hour"])],
                    "backgroundColor": "rgba(75, 192, 192, 0.4)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": True,
                    "tension": 0.4
                },
                {
                    "label": day_before_yesterday_str,
                    "data": [d["value"] for d in sorted(day_before_yesterday_data, key=lambda x: x["hour"])],
                    "backgroundColor": "rgba(255, 99, 132, 0.4)",
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "fill": True,
                    "tension": 0.4
                }
            ]
        }

        return jsonify(response)

    elif option == 2:
        # Option 2: Group Current and Last Month by Weeks 1 to 4
        now = datetime.now()
        first_day_current_month = datetime(now.year, now.month, 1)
        last_day_last_month = first_day_current_month - timedelta(days=1)
        first_day_last_month = datetime(last_day_last_month.year, last_day_last_month.month, 1)
        queries = [
            {"Day": {"$gte": first_day_last_month.strftime('%Y-%m-%d'), "$lte": last_day_last_month.strftime('%Y-%m-%d')}},
            {"Day": {"$gte": first_day_current_month.strftime('%Y-%m-%d'), "$lte": now.strftime('%Y-%m-%d')}}
        ]

        pipeline = [
            {
                "$match": {
                    "Day": { "$exists": True, "$ne": None }
                }
            },
            {
                "$addFields": {
                    "parsedDate": {
                        "$dateFromString": {
                            "dateString": "$Day",
                            "format": "%Y-%m-%d"
                        }
                    }
                }
            },
            {
                "$addFields": {
                    "week": {
                        "$switch": {
                            "branches": [
                                { "case": { "$lte": [{ "$dayOfMonth": "$parsedDate" }, 7] }, "then": 1 },
                                { "case": { "$lte": [{ "$dayOfMonth": "$parsedDate" }, 14] }, "then": 2 },
                                { "case": { "$lte": [{ "$dayOfMonth": "$parsedDate" }, 21] }, "then": 3 },
                                { "case": { "$lte": [{ "$dayOfMonth": "$parsedDate" }, 28] }, "then": 4 }
                            ],
                            "default": 4
                        }
                    },
                    "month": { "$month": "$parsedDate" }
                }
            },
            {
                "$group": {
                    "_id": {
                        "month": "$month",
                        "week": "$week"
                    },
                    "active_power_sum": { "$sum": "$P_abd" }
                }
            },
            {
                "$sort": { "_id.week": 1 }
            }
        ]

        results = list(device_day_collection.aggregate(pipeline))

        # Process results into datasets
        current_month_data = [0, 0, 0, 0]  # For weeks 1 to 4
        last_month_data = [0, 0, 0, 0]     # For weeks 1 to 4

        for record in results:
            month = record["_id"]["month"]
            week = record["_id"]["week"]
            active_power_sum = record["active_power_sum"]

            if month == first_day_current_month.month:
                current_month_data[week - 1] += active_power_sum
            elif month == first_day_last_month.month:
                last_month_data[week - 1] += active_power_sum
            currentm=first_day_last_month.strftime('%B'),
            prevm=first_day_current_month.strftime('%B')
        # Prepare response
        labels = ["Week 1", "Week 2", "Week 3", "Week 4"]

        response = {
            "labels": labels,
            "datasets": [
                {
                    "label": currentm,
                    "data": current_month_data,
                    "backgroundColor": "rgba(75, 192, 192, 0.4)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": True,
                    "tension": 0.4
                },
                {
                    "label": prevm,
                    "data": last_month_data,
                    "backgroundColor": "rgba(255, 99, 132, 0.4)",
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "fill": True,
                    "tension": 0.4
                }
            ]
        }

        return jsonify(response)

    elif option == 3:
        # Option 3: Group Current Year and Last Year by Quarters
        current_year = datetime.now().year
        last_year = current_year - 1

        pipeline = [
            {
                "$match": {
                    "Day": {"$exists": True, "$ne": None}
                }
            },
            {
                "$addFields": {
                    "parsedDate": {
                        "$dateFromString": {
                            "dateString": "$Day",
                            "format": "%Y-%m-%d"
                        }
                    }
                }
            },
            {
                "$addFields": {
                    "quarter": {
                        "$switch": {
                            "branches": [
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 3]}, "then": 1},
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 6]}, "then": 2},
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 9]}, "then": 3},
                                {"case": {"$lte": [{"$month": "$parsedDate"}, 12]}, "then": 4}
                            ],
                            "default": 4
                        }
                    },
                    "year": {"$year": "$parsedDate"}
                }
            },
            {
                "$group": {
                    "_id": {
                        "year": "$year",
                        "quarter": "$quarter"
                    },
                    "active_power_sum": {"$sum": "$P_abd"}
                }
            },
            {
                "$sort": {"_id.quarter": 1, "_id.year": 1}
            }
        ]

        results = list(device_day_collection.aggregate(pipeline))
        print(results)

        # Initialize quarter data for both years
        quarter_data_current_year = [0, 0, 0, 0]  # For Q1 to Q4 (current year)
        quarter_data_last_year = [0, 0, 0, 0]     # For Q1 to Q4 (last year)

        for record in results:
            year = record["_id"]["year"]
            quarter = record["_id"]["quarter"]
            active_power_sum = record["active_power_sum"]

            if year == current_year:
                quarter_data_current_year[quarter - 1] += active_power_sum
            elif year == last_year:
                quarter_data_last_year[quarter - 1] += active_power_sum

        # Labels for quarters
        labels = ["Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4"]

        # Prepare response
        response = {
            "labels": labels,
            "datasets": [
                {
                    "label": f"{last_year}",
                    "data": quarter_data_last_year,
                    "backgroundColor": "rgba(75, 192, 192, 0.4)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": True,
                    "tension": 0.4
                },
                {
                    "label": f"{current_year}",
                    "data": quarter_data_current_year,
                    "backgroundColor": "rgba(255, 99, 132, 0.4)",
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "fill": True,
                    "tension": 0.4
                }
            ]
        }

        return jsonify(response)

@app.route('/radiation_intensity', methods=['POST'])
def get_radiation_intensity():
    try:
        # Parse input data
        data = request.json
        start_date = data.get("start_date")  # Format: YYYY-MM-DD
        end_date = data.get("end_date")      # Format: YYYY-MM-DD
        station_code = data.get("stationCode")

        if not start_date or not end_date or not station_code:
            return jsonify({"error": "Missing required parameters."}), 400

        # Convert dates to timestamps
        try:
            start_timestamp = datetime.strptime(start_date, "%Y-%m-%d")
            end_timestamp = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)  # End of the day
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        if start_timestamp > end_timestamp:
            return jsonify({"error": "start_date must be earlier than or equal to end_date."}), 400

        # Build MongoDB query for the two specific dates
        query = {
            "$and": [
                {"stationCode": station_code},
                {"$or": [
                    {"timestamp": {"$gte": start_timestamp.strftime("%Y-%m-%d %H:%M:%S"), "$lt": (start_timestamp + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")}},
                    {"timestamp": {"$gte": (end_timestamp - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), "$lte": end_timestamp.strftime("%Y-%m-%d %H:%M:%S")}}
                ]}
            ]
        }

        projection = {
            "timestamp": 1,
            "dataItemMap.radiation_intensity": 1,
            "_id": 0
        }

        results = list(collection.find(query, projection))

        # Process results
        grouped_data = {}
        for result in results:
            collect_time = result["timestamp"]
            collect_time_dt = datetime.strptime(collect_time, "%Y-%m-%d %H:%M:%S")
            date = collect_time_dt.strftime("%Y-%m-%d")
            hour = collect_time_dt.hour
            value = result["dataItemMap"].get("radiation_intensity", 0)

            if date not in grouped_data:
                grouped_data[date] = [0] * 24

            grouped_data[date][hour] = value

        # Format output
        output = []
        for date, hourly_values in grouped_data.items():
            output.append({
                "date": date,
                "hourly_values": [
                    {"hour": hour, "value": value} for hour, value in enumerate(hourly_values)
                ]
            })

        return jsonify(output), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/temperature", methods=["POST"])
def temperature_api():
    data = request.get_json()

    # Extracting payload values
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    option = data.get("option")
    plant = data.get("plant")
    tag = data.get("tag")

    if not all([start_date, end_date, option, plant, tag]):
        return jsonify({"error": "Missing required parameters"}), 400

    # Validate tag
    if tag not in [1, 2]:
        return jsonify({"error": "Invalid tag value. Use 1 for temperature or 2 for efficiency"}), 400

    # Converting dates to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Querying the database
    query = {
        "Plant": plant,
        "Day_Hour": {
            "$gte": start_date.strftime("%Y-%m-%d"),
            "$lte": end_date.strftime("%Y-%m-%d")
        }
    }

    cursor = GT_Hourly.find(query)

    # Select the appropriate field based on the tag
    field = "temperature" if tag == 1 else "efficiency"

    # Processing the data
    grouped_data = {}

    for record in cursor:
        day_hour = record.get("Day_Hour")
        value = record.get(field)
        sn = record.get("sn")

        if not (day_hour and value and sn):
            continue

        date_part, hour_part = day_hour.split(" ")

        if option == 1:
            # Group by Day_Hour and sn
            grouped_data.setdefault(sn, {}).setdefault(day_hour, value)

        elif option == 2:
            # Group by date and sn, accumulate values for averaging
            grouped_data.setdefault(sn, {}).setdefault(date_part, []).append(value)

    results = {}

    if option == 2:
        # Calculate average for each date
        for sn, date_data in grouped_data.items():
            for date, values in date_data.items():
                grouped_data[sn][date] = sum(values) / len(values)

    # Adjusting response format
    formatted_results = []
    for sn, data in grouped_data.items():
        formatted_sn_data = []
        for category, value in data.items():
            formatted_sn_data.append({
                "category": category,  # X-axis values: Day_Hour or date
                "value": value         # Y-axis value
            })
        formatted_results.append({
            "sn": sn,
            "data": formatted_sn_data
        })

    return jsonify(formatted_results)

@app.route("/temperature1", methods=["POST"])
def temperature_api1():
    data = request.get_json()

    # Extracting payload values
    week_numbers = data.get("week_numbers")  # List of week numbers, e.g., [28, 41, 46]
    year = data.get("year")                 # e.g., 2024
    option = data.get("option")
    plant = data.get("plant")
    tag = data.get("tag")

    if not all([week_numbers, year, option, plant, tag]):
        return jsonify({"error": "Missing required parameters"}), 400

    # Validate tag
    if tag not in [1, 2]:
        return jsonify({"error": "Invalid tag value. Use 1 for temperature or 2 for efficiency"}), 400

    # Convert week numbers to date ranges
    try:
        year = int(year)
        date_ranges = []
        for week in week_numbers:
            week = int(week)
            start_date = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
            end_date = start_date + timedelta(days=6)
            date_ranges.append((start_date, end_date))
    except ValueError:
        return jsonify({"error": "Invalid week or year format"}), 400

    # Querying the database for each week range
    query_results = []
    for start_date, end_date in date_ranges:
        query = {
            "Plant": plant,
            "Day_Hour": {
                "$gte": start_date.strftime("%Y-%m-%d"),
                "$lte": end_date.strftime("%Y-%m-%d")
            }
        }
        query_results.extend(GT_Hourly.find(query))

    # Select the appropriate field based on the tag
    field = "temperature" if tag == 1 else "efficiency"

    # Processing the data
    grouped_data = {}

    for record in query_results:
        day_hour = record.get("Day_Hour")
        value = record.get(field)
        sn = record.get("sn")

        if not (day_hour and value and sn):
            continue

        date_part, hour_part = day_hour.split(" ")

        if option == 1:
            # Group by Day_Hour and sn
            grouped_data.setdefault(sn, {}).setdefault(day_hour, value)

        elif option == 2:
            # Group by date and sn, accumulate values for averaging
            grouped_data.setdefault(sn, {}).setdefault(date_part, []).append(value)

    if option == 2:
        # Calculate average for each date
        for sn, date_data in grouped_data.items():
            for date, values in date_data.items():
                grouped_data[sn][date] = sum(values) / len(values)

    # Adjusting response format
    formatted_results = []
    for sn, data in grouped_data.items():
        formatted_sn_data = []
        for category, value in data.items():
            formatted_sn_data.append({
                "category": category,  # X-axis values: Day_Hour or date
                "value": value         # Y-axis value
            })
        formatted_results.append({
            "sn": sn,
            "data": formatted_sn_data
        })

    return jsonify(formatted_results)


def process_and_cluster_data(start_date, end_date,plant,max_clusters=10):
    try:
        # Fetch data from MongoDB collection
        query = {
            "timestamp": {
                "$gte": start_date,
                "$lte": end_date
            },
            "dataItemMap.Plant":plant
        }
        cursor = overall_data.find(query)
        data = list(cursor)

        if not data:
            return {"message": "No data found for the given date range"}, 404

        # Convert data to a DataFrame
        df = pd.json_normalize(data)

        # Extract necessary fields from the `dataItemMap` column
        
        df['P_abd'] = df['dataItemMap.P_abd']
        df['Plant'] = df['dataItemMap.Plant']
        df['sn'] = df['dataItemMap.sn']
        df['MPPT'] = df['dataItemMap.MPPT']
        df['Strings'] = df['dataItemMap.Strings']
        df['Watt/String'] = df['dataItemMap.Watt/String']
        df['Day'] = df['timestamp']
        df['dischargeCap'] = df['dataItemMap.dischargeCap']
        df['radiation_intensity'] = df['dataItemMap.radiation_intensity']

        # Convert timestamp string to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Filter data by date range
        df = df[(df['timestamp'] >= pd.to_datetime(start_date)) & (df['timestamp'] <= pd.to_datetime(end_date))]

        if df.empty:
            return {"message": "No data found for the given date range"}, 404

        # Calculate Perf_score
        df['Perf_score'] = df['P_abd'] / (df['Watt/String'] / 1000) / df['radiation_intensity']
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Perf_score'])

        # Normalize Date and Create Key
        df['Date'] = df['timestamp'].dt.normalize().astype(str)
        df['Key'] = df['Plant'] + '-' + df['sn'] + '-' + df['MPPT'] + '-' + df['Strings']

        # Create a complete date range
        date_range = pd.date_range(start=start_date, end=end_date).normalize()
        date_range_str = date_range.astype(str)

        # Pivot table for clustering
        pivot_df = df.pivot_table(index='Key', columns='Date', values='Perf_score').fillna(0)

        # Ensure all dates are included by reindexing
        pivot_df = pivot_df.reindex(columns=date_range_str, fill_value=0)

        # Determine optimal clusters
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(pivot_df)
            silhouette_scores.append(silhouette_score(pivot_df, model.labels_))
        optimal_clusters = range(2, max_clusters + 1)[np.argmax(silhouette_scores)]

        # Perform KMeans clustering
        model = KMeans(n_clusters=optimal_clusters, random_state=42)
        pivot_df['Cluster'] = model.fit_predict(pivot_df)

        # Calculate the mean Perf_score for each cluster
        # Compute mean Perf_score for each cluster per day
        cluster_means = (
            pivot_df.drop(columns=['Cluster'])  # Remove the Cluster column for calculations
            .groupby(pivot_df['Cluster'])       # Group by Cluster
            .mean()                             # Calculate mean for each date
            .to_dict(orient='index')            # Convert to a dictionary
        )


        # Prepare cluster-wise data points
        cluster_data = pivot_df.reset_index().melt(
            id_vars=['Key', 'Cluster'], var_name='Date', value_name='Perf_score'
        )

        clustered_datapoints = cluster_data.groupby('Cluster')[['Key', 'Date', 'Perf_score']].apply(
            lambda x: x.to_dict(orient='records')
        ).to_dict()

        # Prepare response
        response = {
            'optimal_clusters': optimal_clusters,
            'cluster_counts': pivot_df['Cluster'].value_counts().to_dict(),
            'clustered_datapoints': clustered_datapoints,
            'cluster_means': cluster_means  # Add the cluster means to the response
        }
        return response, 200

    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/cluster', methods=['POST'])
def cluster_api():
    try:
        # Parse JSON payload
        payload = request.get_json()
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        plant = payload.get('plant')
        # Input validation
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        # Process and return data
        response, status = process_and_cluster_data(start_date, end_date,plant)
        return jsonify(response), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        # Parse start_date, end_date, and additional filters from the request payload
        payload = request.json
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        plant = payload.get("plant")
        inverter = payload.get("inverter")  # Matches with "sn" in the data
        mppt = payload.get("mppt")
        string = payload.get("string")

        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        # Convert start_date and end_date to string in ISO format
        start_date = str(pd.to_datetime(start_date).date())
        end_date = str(pd.to_datetime(end_date).date())

        # Step 1: Build the query dynamically based on the payload
        query = {"timestamp": {"$gte": start_date, "$lte": end_date}}
        if plant:
            query["dataItemMap.Plant"] = plant
        if inverter:
            query["dataItemMap.sn"] = inverter
        if mppt:
            query["dataItemMap.MPPT"] = mppt
        if string:
            query["dataItemMap.Strings"] = string

        # Projection to limit the fields returned
        projection = {
            "timestamp": 1,
            "dataItemMap.Plant": 1,
            "dataItemMap.MPPT": 1,
            "dataItemMap.Strings": 1,
            "dataItemMap.sn": 1,
            "dataItemMap.Watt/String": 1,
            "dataItemMap.P_abd": 1,
            "dataItemMap.i": 1,
            "dataItemMap.u": 1,
            "dataItemMap.radiation_intensity": 1,
        }

        # Execute the query
        cursor = overall_data.find(query, projection)
        data = list(cursor)

        # Step 2: Convert to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            return jsonify({"error": "No data found for the given filters"}), 404

        if 'dataItemMap' in df.columns:
            data_map = pd.json_normalize(df['dataItemMap'])
            df = pd.concat([df.drop(columns=['dataItemMap']), data_map], axis=1)

        # Print columns to debug
        print("Columns in DataFrame:", df.columns)

        # Check for missing columns and adapt dynamically
        required_columns = [
            "timestamp", "Plant", "MPPT", "Strings", "sn", "Watt/String", "P_abd", 
            "i", "u", "radiation_intensity"
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing columns in data: {missing_columns}"}), 400

        # Step 3: Clean and Transform Data
        df.rename(columns={
            "timestamp": "Date",
            "Plant": "Plant",
            "MPPT": "MPPT",
            "Strings": "String",
            "sn": "sn",
            "Watt/String": "Watts/String",
            "P_abd": "Power",
            "i": "Current",
            "u": "Voltage",
            "radiation_intensity": "radiation_intensity"
        }, inplace=True)

        # Select required columns
        selected_columns = [
            "Date", "Plant", "MPPT", "String", "sn", "Watts/String", "Power", 
            "Current", "Voltage", "radiation_intensity"
        ]
        df = df[selected_columns]

        # Handle missing or invalid data
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        # Convert data types
        numeric_columns = ['radiation_intensity', 'Watts/String', 'Power', 'Current', 'Voltage']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with any missing numeric values
        df = df.dropna(subset=numeric_columns)

        # Remove rows where radiation_intensity is zero to avoid division by zero
        df = df[df['radiation_intensity'] > 0]

        # Create new columns
        df['Key'] = df['Plant'] + '-' + df['sn'] + '-' + df['MPPT'] + '-' + df['String']
        df['Perf_score'] = df['Power'] / (df['Watts/String'] / 1000) / df['radiation_intensity']

        # Remove infinite or NaN performance scores
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Perf_score'])

        # Step 4: Create Pivot Tables for Perf_score, Power, and Watts/String
        perf_pivot = df.pivot_table(index='Key', columns='Date', values='Perf_score')
        power_pivot = df.pivot_table(index='Key', columns='Date', values='Power')
        watts_pivot = df.pivot_table(index='Key', columns='Date', values='Watts/String')

        # Format column names as dates
        perf_pivot.columns = perf_pivot.columns.strftime('%m/%d/%Y')
        power_pivot.columns = power_pivot.columns.strftime('%m/%d/%Y')
        watts_pivot.columns = watts_pivot.columns.strftime('%m/%d/%Y')

        # Remove columns with NaN values entirely from all pivot tables
        valid_columns = perf_pivot.columns.intersection(power_pivot.columns).intersection(watts_pivot.columns)
        valid_columns = valid_columns[~perf_pivot[valid_columns].isnull().any(axis=0)]
        valid_columns = valid_columns[~power_pivot[valid_columns].isnull().any(axis=0)]
        valid_columns = valid_columns[~watts_pivot[valid_columns].isnull().any(axis=0)]

        # Filter pivot tables
        perf_pivot = perf_pivot[valid_columns]
        power_pivot = power_pivot[valid_columns]
        watts_pivot = watts_pivot[valid_columns]

        # Step 5: Calculate absolute z-scores for Perf_score and assign colors
        zscore_df = (perf_pivot - perf_pivot.mean()) / perf_pivot.std()
        color_df = zscore_df.applymap(lambda z: "red" if z < -3 else ("orange" if z < -2 else "transparent"))

        # Combine data into a single response
        result = []
        for key in perf_pivot.index:
            row = {"Key": key}
            for date in perf_pivot.columns:
                perf_score = perf_pivot.loc[key, date]
                power = power_pivot.loc[key, date]
                watts = watts_pivot.loc[key, date]
                z_score = zscore_df.loc[key, date]
                color = color_df.loc[key, date]
                row[date] = {
                    "value": perf_score,
                    "power": round(power, 2),
                    "watts_per_string": round(watts, 2),
                    "z_score": z_score,
                    "color": color
                }
            result.append(row)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/chart-water-data', methods=['POST'])
def chartwaterdata():
    # Get query parameters
    payload = request.json
    start_date = payload.get('start_date')
    end_date = payload.get('end_date')
    plant = payload.get('plant')
    inverter = payload.get('inverter', None)
    mppt = payload.get('mppt', None)
    string = payload.get('string', None)

    # Parse dates
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Build the aggregation pipeline
    pipeline = [
        {"$match": {
            "Day": {"$gte": start_date.strftime("%Y-%m-%d"), "$lte": end_date.strftime("%Y-%m-%d")},
            "Plant": plant
        }},
        # Add additional filters based on optional query parameters
        {"$match": {**({"sn": inverter} if inverter else {}),
                    **({"MPPT": mppt} if mppt else {}),
                    **({"Strings": string} if string else {})}},
        {"$group": {
            "_id": {"month": {"$substr": ["$Day", 0, 7]}},  # Extract month-year from "Day"
            "total_P_abd": {"$sum": "$P_abd"}
        }},
        {"$sort": {"_id.month": 1}}  # Sort by month
    ]

    # Execute the aggregation query
    records = device_day_collection.aggregate(pipeline)

    # Prepare result data
    result = []
    previous_value = 0
    for record in records:
        month_year = record["_id"]["month"]
        value = record["total_P_abd"]
        result.append({
            "category": month_year,
            "value": round(value,2)-round(previous_value,2),
            "open": round(value,2)-round(previous_value,2),
            "stepValue": round(value,2),
            "displayValue": round(value,2)-round(previous_value,2)
        })
        previous_value = value

    return jsonify(result)

@app.route('/grouped_data', methods=['POST'])
def grouped_data():
    try:
        # Parse payload
        payload = request.json
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        plant = payload.get('plant')
        inverter = payload.get('inverter')
        mppt = payload.get('mppt')
        string = payload.get('string')
        if not (start_date and end_date and plant):
            return jsonify({"error": "start_date, end_date, and plant are required"}), 400
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        end_date = end_date + timedelta(days=1)
        end_date_str = end_date.strftime('%Y-%m-%d')
        query = {
            "Plant": plant,
            "Day_Hour": {"$gte": start_date.strftime('%Y-%m-%d'), "$lte": end_date_str}
        }

        if inverter:
            query["sn"] = inverter
        if mppt:
            query["MPPT"] = mppt
        if string:
            query["Strings"] = string

        # Group data and calculate sum of P_abd
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": {
                        "Day_Hour": "$Day_Hour",
                        "Plant": "$Plant",
                        "sn": "$sn" if inverter else None,
                        "MPPT": "$MPPT" if mppt else None,
                        "Strings": "$Strings" if string else None,
                    },
                    "P_abd_sum": {"$sum": "$P_abd"}
                }
            },
            {"$sort": {"_id.Day_Hour": 1}}  # Sort by Day_Hour
        ]

        # Execute aggregation
        results = list(device_hour_collection.aggregate(pipeline))

        # Format the output
        output = []
        for result in results:
            day_hour = result["_id"]["Day_Hour"]
            date_only = datetime.strptime(day_hour, '%Y-%m-%d %H').strftime('%Y-%m-%d') if day_hour else None
            hour = datetime.strptime(day_hour, '%Y-%m-%d %H').hour if day_hour else None
            grouped_data = {
                "Day_Hour": date_only,
                "Hour": hour,
                "Plant": result["_id"]["Plant"],
                "P_abd_sum": result["P_abd_sum"]
            }
            if inverter:
                grouped_data["sn"] = result["_id"].get("sn")
            if mppt:
                grouped_data["MPPT"] = result["_id"].get("MPPT")
            if string:
                grouped_data["Strings"] = result["_id"].get("Strings")
            output.append(grouped_data)

        return jsonify(output), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_data', methods=['POST'])
def get_data():
    # Parse request JSON
    data = request.json
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    plant = data.get('plant')
    inverter = data.get('inverter')
    mppt = data.get('mppt')
    string = data.get('string')
    option = data.get('option')
    ph = data.get('ph')

    # Convert start_date and end_date to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    match_stage = {
        "$match": {
            "Day": {"$gte": start_date.strftime("%Y-%m-%d"), "$lte": end_date.strftime("%Y-%m-%d")},
            "Plant": plant
        }
    }

    # Add inverter, mppt, and string filtering based on payload values
    if inverter:
        match_stage["$match"]["sn"] = inverter
    if mppt:
        match_stage["$match"]["MPPT"] = mppt
    if string:
        match_stage["$match"]["Strings"] = string

    # Convert 'Day' from string to Date using $dateFromString and perform grouping based on option
    if option == 1:  # Group by Date
        group_stage = {
            "$group": {
                "_id": {
                    "date": {"$dateFromString": {"dateString": "$Day"}},
                    "plant": "$Plant",
                    "inverter": "$sn" if inverter else None,
                    "mppt": "$MPPT" if mppt else None,
                    "string": "$Strings" if string else None,
                },
                "total_P_abd": {"$sum": "$P_abd"},
                "P_abd": {"$first": "$P_abd"}  # Include the first P_abd value for response
            }
        }

    elif option == 2:  # Group by Week Number
        group_stage = {
            "$group": {
                "_id": {
                    "week": {"$isoWeek": {"$dateFromString": {"dateString": "$Day"}}},
                    "plant": "$Plant",
                    "inverter": "$sn" if inverter else None,
                    "mppt": "$MPPT" if mppt else None,
                    "string": "$Strings" if string else None,
                },
                "total_P_abd": {"$sum": "$P_abd"},
                "P_abd": {"$first": "$P_abd"}  # Include the first P_abd value for response
            }
        }

    elif option == 3:  # Group by Month
        group_stage = {
            "$group": {
                "_id": {
                    "month": {"$month": {"$dateFromString": {"dateString": "$Day"}}},
                    "year": {"$year": {"$dateFromString": {"dateString": "$Day"}}},
                    "plant": "$Plant",
                    "inverter": "$sn" if inverter else None,
                    "mppt": "$MPPT" if mppt else None,
                    "string": "$Strings" if string else None,
                },
                "total_P_abd": {"$sum": "$P_abd"},
                "P_abd": {"$first": "$P_abd"}  # Include the first P_abd value for response
            }
        }

    # Multiply the result by PH
    project_stage = {
        "$project": {
            "_id": 0,
            "date": {
                "$cond": {
                    "if": {"$eq": [option, 1]},  # Option 1: Group by Date
                    "then": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",  # Format date as "year-month-day"
                            "date": "$_id.date"
                        }
                    },
                    "else": {
                        "$cond": {
                            "if": {"$eq": [option, 2]},  # Option 2: Group by Week
                            "then": {
                                "$concat": [
                                    "Week ",
                                    {"$toString": "$_id.week"}  # Prefix week number with "week: "
                                ]
                            },
                            "else": {  # Option 3: Group by Month
                                "$concat": [
                                    "Month ",
                                    {"$toString": "$_id.month"}  # Prefix month number with "month: "
                                ]
                            }
                        }
                    }
                }
            },
            "year": {"$cond": {"if": {"$eq": [option, 3]}, "then": "$_id.year", "else": None}},
            "plant": "$_id.plant",
            "inverter": "$_id.inverter",
            "mppt": "$_id.mppt",
            "string": "$_id.string",
            "sum_abd": {"$multiply": ["$total_P_abd", ph]},  # Rename and multiply by ph
            "P_abd": "$total_P_abd"  # Include P_abd value
        }
    }

    # Sorting stage to arrange results by date, week, or month
    sort_stage = {
        "$sort": {}
    }

    if option == 1:  # Sort by Date
        sort_stage["$sort"]["_id.date"] = 1
    elif option == 2:  # Sort by Week
        sort_stage["$sort"]["_id.week"] = 1
    elif option == 3:  # Sort by Year and Month
        sort_stage["$sort"]["_id.year"] = 1
        sort_stage["$sort"]["_id.month"] = 1

    # Execute the aggregation pipeline
    pipeline = [match_stage, group_stage, project_stage, sort_stage]
    result = list(device_day_collection.aggregate(pipeline))

    # Return the result as JSON
    return jsonify(result)

@app.route('/get_solar_power_values', methods=['POST'])
def get_solar_power_values():
    # Parse payload
    payload = request.json
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    plant = payload.get("plant")
    inverter = payload.get("inverter")
    mppt = payload.get("mppt")
    string = payload.get("string")
    option = payload.get("option", 1)  # Option 1 means group by date, 0 means group by hour

    # Convert dates to strings for filtering
    start_date = datetime.strptime(start_date, '%Y-%m-%d').isoformat()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').isoformat()

    # Build the query
    query = {
        "Day_Hour": {"$gte": start_date, "$lte": end_date}  # Filter between start and end date
    }
    if plant:
        query["Plant"] = plant
    if inverter:
        query["sn"] = inverter
    if mppt:
        query["MPPT"] = mppt
    if string:
        query["Strings"] = string

    # Determine grouping conditions based on option
    if option == 1:  # Group by date as well as existing conditions
        grouping_conditions = {
            "date": {"$substr": ["$Day_Hour", 0, 10]},  # Extract YYYY-MM-DD from Day_Hour
            "hour": {"$toInt": {"$substr": ["$Day_Hour", 11, 2]}}  # Extract HH from Day_Hour and convert to integer
        }
    else:  # Default behavior is to group by hour only
        grouping_conditions = {
            "hour": {"$toInt": {"$substr": ["$Day_Hour", 11, 2]}}  # Extract HH from Day_Hour and convert to integer
        }

    if plant:
        grouping_conditions["plant"] = "$Plant"
    if inverter:
        grouping_conditions["inverter"] = "$sn"
    if mppt:
        grouping_conditions["mppt"] = "$MPPT"
    if string:
        grouping_conditions["string"] = "$Strings"

    # MongoDB aggregation pipeline
    pipeline = [
        {"$match": query},  # Match documents based on the query
        {
            "$group": {
                "_id": grouping_conditions,
                "avg_u": {"$sum": "$P_abd"}  # Calculate the sum of the field 'u'
            }
        }
    ]
    
    # If option is 1, group by date as well and format the response accordingly
    if option == 1:
        pipeline.append(
            {
                "$group": {
                    "_id": "$_id.date",  # Group by date
                    "hourly_values": {
                        "$push": {
                            "hour": "$_id.hour",
                            "value": "$avg_u"
                        }
                    }
                }
            }
        )
    
    pipeline.append({"$sort": {"_id": 1}})  # Sort by date (or hour depending on the option)

    # Execute the aggregation
    results = list(device_hour_collection.aggregate(pipeline))

    # Format the response
    response = [
        {
            "date": result["_id"],
            "hourly_values": result["hourly_values"]
        }
        for result in results
    ]

    return jsonify(response)


@app.route('/active_peak_power', methods=['POST'])
def active_peak_power():
    try:
        # Parse the request payload
        payload = request.json
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        plant = payload.get('plant')
        # Validate the input dates
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        # Query the data from MongoDB
        pipeline = [
    {
        "$match": {
            "Day": {"$gte": start_date, "$lte": end_date},
            "Plant":plant
        }
    },
    {
        "$addFields": {
            "hour": {
                "$cond": [
                    {"$ifNull": ["$Day_Hour", False]},  # Check if Day_Hour exists
                    {"$arrayElemAt": [{"$split": ["$Day_Hour", " "]}, 1]},  # Extract the hour
                    None  # Set to None if Day_Hour is missing
                ]
            }
        }
    },
    {
        "$sort": {"active_power": -1}  # Sort by active_power descending
    },
    {
        "$group": {
            "_id": "$Day",  # Group by day
            "max_active_power": {"$first": "$active_power"},  # Take the highest active_power
            "hour_with_max_power": {"$first": "$hour"}  # Take the corresponding hour
        }
    },
    {
        "$sort": {"_id": 1}  # Sort by day
    }
]

        results = list(GM_Hourly.aggregate(pipeline))
        print(results)
        # Prepare the response
        response_data = []
        for result in results:
            day = result.get("_id")
            max_active_power = result.get("max_active_power")
            max_power_hour = result.get("hour_with_max_power")

            # Ensure hour is not None before applying zfill
            max_power_hour = max_power_hour.zfill(2) if max_power_hour else "--"

            # Append to response data
            response_data.append({
                "date": day,
                "hour": max_power_hour,  # Ensure hour is two digits or placeholder
                "max_active_power": round(max_active_power, 2) if max_active_power else 0
            })


        # Return the response
        return jsonify({
            "data": response_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/active_power_weekday_values', methods=['POST'])
def active_power_weekday_values():
    # Parse payload
    payload = request.json
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    weekday_name = payload.get("weekday")  # Get the weekday from the payload
    plant = payload.get("plant")
    # Map weekday names to integers (Monday = 0, Sunday = 6)
    weekday_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    if weekday_name not in weekday_map:
        return jsonify({"error": "Invalid weekday specified"}), 400

    weekday = weekday_map[weekday_name]

    # Convert dates to strings for filtering
    start_date = datetime.strptime(start_date, '%Y-%m-%d').isoformat()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').isoformat()

    # Build the query
    query = {
        "Day_Hour": {
            "$gte": f"{start_date[:10]} 00",
            "$lte": f"{end_date[:10]} 23"
        },
        "Plant":plant
    }

    # Determine grouping conditions
    grouping_conditions = {
        "date": {"$substr": ["$Day_Hour", 0, 10]},  # Extract YYYY-MM-DD from Day_Hour
        "hour": {"$toInt": {"$substr": ["$Day_Hour", 11, 2]}}  # Extract HH from Day_Hour and convert to integer
    }

    # MongoDB aggregation pipeline
    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": grouping_conditions,
                "avg_u": {"$sum": "$active_power"}  # Calculate the sum of active_power
            }
        },
        {
            "$group": {
                "_id": "$_id.date",
                "hourly_values": {
                    "$push": {
                        "hour": "$_id.hour",
                        "value": "$avg_u"
                    }
                },
                "total_power": {"$sum": "$avg_u"}  # Calculate total power for each date
            }
        },
        {"$sort": {"_id": 1}}  # Sort by date
    ]

    # Execute the aggregation
    results = list(GM_Hourly.aggregate(pipeline))

    # Filter results for the specified weekday
    weekday_results = [
        {
            "date": result["_id"],
            "hourly_values": result["hourly_values"],
            "total_power": result["total_power"]
        }
        for result in results
        if datetime.strptime(result["_id"], '%Y-%m-%d').weekday() == weekday
    ]

    # Format the response with all the specified weekdays
    response = [
        {
            "description": f"{weekday_name} ({entry['date']})",
            "date": entry["date"],
            "hourly_values": entry["hourly_values"]
        }
        for entry in weekday_results
    ]

    return jsonify(response)


@app.route('/active_power_monday_values', methods=['POST'])
def active_power_monday_values():
    # Parse payload
    payload = request.json
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    weekday_name = payload.get("weekday")  # Get the weekday from the payload
    plant = payload.get("plant")
    # Map weekday names to integers (Monday = 0, Sunday = 6)
    weekday_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    if weekday_name not in weekday_map:
        return jsonify({"error": "Invalid weekday specified"}), 400

    weekday = weekday_map[weekday_name]

    # Convert dates to strings for filtering
    start_date = datetime.strptime(start_date, '%Y-%m-%d').isoformat()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').isoformat()

    # Build the query
    query = {
        "Day_Hour": {
            "$gte": f"{start_date[:10]} 00",
            "$lte": f"{end_date[:10]} 23"
        },
        "Plant":plant
    }

    # Determine grouping conditions
    grouping_conditions = {
        "date": {"$substr": ["$Day_Hour", 0, 10]},  # Extract YYYY-MM-DD from Day_Hour
        "hour": {"$toInt": {"$substr": ["$Day_Hour", 11, 2]}}  # Extract HH from Day_Hour and convert to integer
    }

    # MongoDB aggregation pipeline
    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": grouping_conditions,
                "avg_u": {"$sum": "$active_power"}  # Calculate the sum of active_power
            }
        },
        {
            "$group": {
                "_id": "$_id.date",
                "hourly_values": {
                    "$push": {
                        "hour": "$_id.hour",
                        "value": "$avg_u"
                    }
                },
                "total_power": {"$sum": "$avg_u"}  # Calculate total power for each date
            }
        },
        {"$sort": {"_id": 1}}  # Sort by date
    ]

    # Execute the aggregation
    results = list(GM_Hourly.aggregate(pipeline))

    # Filter results for the specified weekday and find the last and busiest day
    weekday_results = [
        {
            "date": result["_id"],
            "hourly_values": result["hourly_values"],
            "total_power": result["total_power"]
        }
        for result in results
        if datetime.strptime(result["_id"], '%Y-%m-%d').weekday() == weekday
    ]

    # Find the last specified weekday and the busiest specified weekday
    if weekday_results:
        last_weekday = max(weekday_results, key=lambda x: x["date"])
        busiest_weekday = max(weekday_results, key=lambda x: x["total_power"])

        # Format the response with only the required weekdays
        response = [
            {
                "description": f"Last {weekday_name} ({last_weekday['date']})",
                "date": last_weekday["date"],
                "hourly_values": last_weekday["hourly_values"]
            },
            {
                "description": f"Busy {weekday_name} ({busiest_weekday['date']})",
                "date": busiest_weekday["date"],
                "hourly_values": busiest_weekday["hourly_values"]
            }
        ]
    else:
        response = []

    return jsonify(response)

@app.route('/active_power_hourly_values', methods=['POST'])
def active_power_hourly_values():
    # Parse payload
    payload = request.json
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    plant = payload.get("plant")
    # Validate input
    if not start_date or not end_date:
        return jsonify({"error": "start_date and end_date are required"}), 400

    # Convert dates to ISO format for filtering
    start_date = datetime.strptime(start_date, '%Y-%m-%d').isoformat()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').isoformat()

    # Build the query to fetch data only for start_date and end_date
    query = {
        "$or": [
            {"Day_Hour": {"$regex": f"^{start_date[:10]}"}},
            {"Day_Hour": {"$regex": f"^{end_date[:10]}"}}
        ],
        "Plant":plant
    }

    # Determine grouping conditions
    grouping_conditions = {
        "date": {"$substr": ["$Day_Hour", 0, 10]},  # Extract YYYY-MM-DD from Day_Hour
        "hour": {"$toInt": {"$substr": ["$Day_Hour", 11, 2]}}  # Extract HH from Day_Hour and convert to integer
    }

    # MongoDB aggregation pipeline
    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": grouping_conditions,
                "avg_u": {"$sum": "$active_power"}  # Calculate the sum of 'active_power'
            }
        },
        {
            "$group": {
                "_id": "$_id.date",
                "hourly_values": {
                    "$push": {
                        "hour": "$_id.hour",
                        "value": "$avg_u"
                    }
                }
            }
        },
        {"$sort": {"_id": 1}}
    ]

    # Execute the aggregation
    results = list(GM_Hourly.aggregate(pipeline))

    # Format the response
    response = [
        {
            "date": result["_id"],
            "hourly_values": result["hourly_values"]
        }
        for result in results
    ]

    return jsonify(response)


@app.route('/active_power_hour_week', methods=['POST'])
def calculate_active_power_hour_week():
    try:
        # Parse the request payload
        payload = request.json
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        plant = payload.get('plant')
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        pipeline = [
    {
        "$match": {
            "Day_Hour": {
                "$regex": r"^\d{4}-\d{2}-\d{2} \d{1,2}$"  # Match single- and double-digit hours
            },
            "Day": {"$gte": start_date, "$lte": end_date},
            "Plant":plant
        }
    },
    {
        "$project": {
            "hour": {
                "$let": {
                    "vars": {"hour_raw": {"$arrayElemAt": [{"$split": ["$Day_Hour", " "]}, 1]}},
                    "in": {
                        "$cond": {
                            "if": {"$lt": [{"$toInt": "$$hour_raw"}, 10]},
                            "then": {"$concat": ["0", "$$hour_raw"]},
                            "else": "$$hour_raw"
                        }
                    }
                }
            },
            "weekday": {
                "$dayOfWeek": {
                    "$dateFromString": {
                        "dateString": {"$arrayElemAt": [{"$split": ["$Day_Hour", " "]}, 0]}
                    }
                }
            },
            "active_power": 1
        }
    },
    {
        "$group": {
            "_id": {"hour": "$hour", "weekday": "$weekday"},
            "total_active_power": {"$sum": "$active_power"}
        }
    },
    {
        "$sort": {"_id.hour": 1, "_id.weekday": 1}
    }
]


        results = list(GM_Hourly.aggregate(pipeline))
        # Prepare hour-wise data for weekdays
        hour_data = {str(i).zfill(2): {j: 0 for j in range(1, 8)} for i in range(24)}

        for result in results:
            hour = result["_id"]["hour"]
            weekday = result["_id"]["weekday"]
            total_active_power = result["total_active_power"]
            hour_data[hour][weekday] = round(total_active_power, 2)

        weekday_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        response_data = []
        for hour, weekdays in hour_data.items():
            response_data.append({
                "hour": hour,
                "weekdays": {
                    weekday_names[weekday - 1]: value for weekday, value in weekdays.items()
                }
            })

        return jsonify({"data": response_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/active_power_hour_week1', methods=['POST'])
def calculate_active_power_hour_week1():
    try:
        # Parse the request payload
        payload = request.json
        week_numbers = payload.get('week_number')
        year = payload.get('year')
        plant = payload.get('plant')
        option = int(payload.get('option'))

        if not week_numbers or not year or not option:
            return jsonify({"error": "week_number, year, and option are required"}), 400

        if not isinstance(week_numbers, list):
            return jsonify({"error": "week_number must be a list"}), 400

        # Convert the week numbers to start and end dates
        from datetime import datetime, timedelta

        date_ranges = []
        for week_number in week_numbers:
            start_date = datetime.strptime(f'{year}-W{int(week_number)}-1', "%Y-W%U-%w").date()
            end_date = start_date + timedelta(days=6)
            date_ranges.append((str(start_date), str(end_date)))

        # MongoDB aggregation pipeline
        pipeline = [
            {
                "$match": {
                    "$or": [
                        {"Day": {"$gte": start, "$lte": end}} for start, end in date_ranges
                    ],
                    "Day_Hour": {
                        "$regex": r"^\d{4}-\d{2}-\d{2} \d{1,2}$"  # Match single- and double-digit hours
                    },
                    "Plant": plant
                }
            },
            {
                "$project": {
                    "hour": {
                        "$let": {
                            "vars": {"hour_raw": {"$arrayElemAt": [{"$split": ["$Day_Hour", " "]}, 1]}},
                            "in": {
                                "$cond": {
                                    "if": {"$lt": [{"$toInt": "$$hour_raw"}, 10]},
                                    "then": {"$concat": ["0", "$$hour_raw"]},
                                    "else": "$$hour_raw"
                                }
                            }
                        }
                    },
                    "weekday": {
                        "$dayOfWeek": {
                            "$dateFromString": {
                                "dateString": {"$arrayElemAt": [{"$split": ["$Day_Hour", " "]}, 0]}
                            }
                        }
                    },
                    "active_power": 1
                }
            },
            {
                "$group": {
                    "_id": {"hour": "$hour", "weekday": "$weekday"},
                    "total_active_power": {
                        "$sum" if option == 1 else "$avg": "$active_power"
                    }
                }
            },
            {
                "$sort": {"_id.hour": 1, "_id.weekday": 1}
            }
        ]

        # Execute the aggregation pipeline
        results = list(GM_Hourly.aggregate(pipeline))

        # Prepare hour-wise data for weekdays
        hour_data = {str(i).zfill(2): {j: 0 for j in range(1, 8)} for i in range(24)}

        for result in results:
            hour = result["_id"]["hour"]
            weekday = result["_id"]["weekday"]
            total_active_power = result["total_active_power"]
            hour_data[hour][weekday] = round(total_active_power, 2)

        weekday_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        response_data = []
        for hour, weekdays in hour_data.items():
            response_data.append({
                "hour": hour,
                "weekdays": {
                    weekday_names[weekday - 1]: value for weekday, value in weekdays.items()
                }
            })

        return jsonify({"data": response_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/active_power_week', methods=['POST'])
def calculate_active_power_week():
    try:
        # Parse the request payload
        payload = request.json
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        plant = payload.get('plant')
        # Validate the input dates
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        # Convert start and end dates to datetime objects
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Query the data from MongoDB
        pipeline = [
            {
                "$match": {
                    "Day": {"$gte": start_date.strftime("%Y-%m-%d"), "$lte": end_date.strftime("%Y-%m-%d")},
                    "Plant":plant 
                }
            },
            {
                "$project": {
                    "day": {"$dateToString": {"format": "%Y-%m-%d", "date": {"$dateFromString": {"dateString": "$Day"}}}},  # Convert string to date
                    "weekday": {"$dayOfWeek": {"$dateFromString": {"dateString": "$Day"}}},  # Get weekday (1=Sunday, 7=Saturday)
                    "week_number": {"$isoWeek": {"$dateFromString": {"dateString": "$Day"}}},  # Get ISO week number
                    "active_power": 1  # Include the active_power field
                }
            },
            {
                "$group": {
                    "_id": {"day": "$day", "weekday": "$weekday", "week_number": "$week_number"},
                    "total_P_abd": {"$sum": "$active_power"}  # Sum the active_power for each day
                }
            },
            {
                "$sort": {"_id.week_number": 1, "_id.weekday": 1}  # Sort by week number and weekday
            }
        ]

        results = list(GM_Hourly.aggregate(pipeline))

        # Prepare the response data
        week_data = {i: {} for i in range(1, 8)}  # Create a dictionary for each weekday (1=Sunday to 7=Saturday)
        
        for result in results:
            week_number = result["_id"]["week_number"]
            weekday = result["_id"]["weekday"]
            total_P_abd = result["total_P_abd"]

            # Get weekday name
            weekday_name = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][weekday - 1]

            # Add data to the corresponding weekday's week_data
            if week_number not in week_data[weekday]:
                week_data[weekday][f"week{week_number}"] = round(total_P_abd, 2)
            else:
                week_data[weekday][f"week{week_number}"] = round(total_P_abd, 2)

        # Prepare the final response, ordered from Monday to Sunday
        response_data = [
            {"weekday": "Monday", "week_data": [week_data[2]]},
            {"weekday": "Tuesday", "week_data": [week_data[3]]},
            {"weekday": "Wednesday", "week_data": [week_data[4]]},
            {"weekday": "Thursday", "week_data": [week_data[5]]},
            {"weekday": "Friday", "week_data": [week_data[6]]},
            {"weekday": "Saturday", "week_data": [week_data[7]]},
            {"weekday": "Sunday", "week_data": [week_data[1]]},
        ]

        # Return the response
        return jsonify({
            "data": response_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/active_power_week1', methods=['POST'])
def calculate_active_power_week1():
    try:
        # Parse the request payload
        payload = request.json
        week_numbers = payload.get('week_numbers')  # Expecting a list of week numbers
        year = payload.get('year')  # Expecting a specific year
        plant = payload.get('plant')

        # Validate the input
        if not week_numbers or not isinstance(week_numbers, list):
            return jsonify({"error": "week_numbers must be a list of integers"}), 400
        if not year or not isinstance(year, int):
            return jsonify({"error": "year must be provided as an integer"}), 400

        # Query the data from MongoDB
        pipeline = [
            {
                "$match": {
                    "Plant": plant
                }
            },
            {
                "$project": {
                    "day": {"$dateToString": {"format": "%Y-%m-%d", "date": {"$dateFromString": {"dateString": "$Day"}}}},
                    "weekday": {"$dayOfWeek": {"$dateFromString": {"dateString": "$Day"}}},
                    "week_number": {"$isoWeek": {"$dateFromString": {"dateString": "$Day"}}},  # This is supported
                    "year": {"$year": {"$dateFromString": {"dateString": "$Day"}}},  # Fallback year calculation
                    "active_power": 1
                }
            },
            {
                "$match": {
                    "week_number": {"$in": week_numbers},  # Filter by week numbers
                    "year": year  # Filter by the specified year
                }
            },
            {
                "$group": {
                    "_id": {"day": "$day", "weekday": "$weekday", "week_number": "$week_number"},
                    "total_P_abd": {"$sum": "$active_power"}
                }
            },
            {
                "$sort": {"_id.week_number": 1, "_id.weekday": 1}
            }
        ]

        results = list(GM_Hourly.aggregate(pipeline))

        # Prepare the response data
        week_data = {i: {} for i in range(1, 8)}  # Create a dictionary for each weekday (1=Sunday to 7=Saturday)

        for result in results:
            week_number = result["_id"]["week_number"]
            weekday = result["_id"]["weekday"]
            total_P_abd = result["total_P_abd"]

            # Get weekday name
            weekday_name = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][weekday - 1]

            # Add data to the corresponding weekday's week_data
            if week_number not in week_data[weekday]:
                week_data[weekday][f"week{week_number}"] = round(total_P_abd, 2)
            else:
                week_data[weekday][f"week{week_number}"] = round(total_P_abd, 2)

        # Prepare the final response, ordered from Monday to Sunday
        response_data = [
            {"weekday": "Monday", "week_data": [week_data[2]]},
            {"weekday": "Tuesday", "week_data": [week_data[3]]},
            {"weekday": "Wednesday", "week_data": [week_data[4]]},
            {"weekday": "Thursday", "week_data": [week_data[5]]},
            {"weekday": "Friday", "week_data": [week_data[6]]},
            {"weekday": "Saturday", "week_data": [week_data[7]]},
            {"weekday": "Sunday", "week_data": [week_data[1]]},
        ]

        # Return the response
        return jsonify({
            "data": response_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/active_power_day', methods=['POST'])
def calculate_active_power_day():
    try:
        # Parse the request payload
        payload = request.json
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        tarrif = payload.get('tarrif')
        option = payload.get('option', 1)  # Default to option 1 if not provided
        # Validate the input
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400
        if option not in [1, 2, 3]:
            return jsonify({"error": "Invalid option. Must be 1, 2, or 3"}), 400

        # Common match stage
        match_stage = {
            "$match": {
                "Day": {"$gte": start_date, "$lte": end_date}
            }
        }

        # Determine group stage and field based on option
        if option == 1:
            group_stage = {
                "$group": {
                    "_id": "$Day",  # Group by Day
                    "total_active_power": {"$sum": "$active_power"}  # Sum active power
                }
            }
            sort_stage = {"$sort": {"_id": 1}}  # Sort by Day
        elif option == 2:
            group_stage = {
                "$group": {
                    "_id": {"$week": {"$dateFromString": {"dateString": "$Day"}}},  # Group by Week
                    "total_active_power": {"$sum": "$active_power"}  # Sum active power
                }
            }
            sort_stage = {"$sort": {"_id": 1}}  # Sort by Week
        elif option == 3:
            group_stage = {
                "$group": {
                    "_id": {"$month": {"$dateFromString": {"dateString": "$Day"}}},  # Group by Month
                    "total_active_power": {"$sum": "$active_power"}  # Sum active power
                }
            }
            sort_stage = {"$sort": {"_id": 1}}  # Sort by Month

        # Query the data
        pipeline = [match_stage, group_stage, sort_stage]
        results = list(GM_Hourly.aggregate(pipeline))

        # Prepare the response
        response_data = []
        for result in results:
            group_key = result["_id"]
            total_power = result["total_active_power"]

            # Cost calculation (same as total_power in this example)
            cost = total_power * tarrif

            # Format the group key based on the option
            if option == 1:
                group_label = group_key  # Day in YYYY-MM-DD format
            elif option == 2:
                group_label = f"Week {group_key}"  # Week number
            elif option == 3:
                group_label = f"Month {group_key}"  # Month number

            # Append to response data
            response_data.append({
                "group": group_label,
                "value": round(total_power, 2),
                "cost": round(cost, 2),
            })

        # Return the response
        return jsonify({
            "data": response_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/active_power_weekday', methods=['POST'])
def active_power_weekday():
    try:
        # Parse the request payload
        payload = request.json
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        aggregation = int(payload.get("aggregation"))
        plant = payload.get("plant")
        
        # Validate the input dates
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        # Query the data from MongoDB
        pipeline = [
            {
                "$match": {
                    "Day": {"$gte": start_date, "$lte": end_date},
                    "Plant": plant
                }
            },
            {
                "$project": {
                    "weekday": {"$isoDayOfWeek": {"$dateFromString": {"dateString": "$Day"}}},  # Extract weekday (ISO format)
                    "active_power": 1,
                    "aggregation_type": {
                        "$cond": {
                            "if": {"$eq": [aggregation, 1]},  # 1 for sum
                            "then": "sum",
                            "else": "avg"
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": "$weekday",  # Group by weekday
                    "total_active_power": {
                        "$sum": {
                            "$cond": {
                                "if": {"$eq": ["$aggregation_type", "sum"]},  # Check if aggregation is "sum"
                                "then": "$active_power",
                                "else": 0
                            }
                        }
                    },
                    "average_active_power": {
                        "$avg": {
                            "$cond": {
                                "if": {"$eq": ["$aggregation_type", "avg"]},  # Check if aggregation is "avg"
                                "then": "$active_power",
                                "else": 0
                            }
                        }
                    }
                }
            },
            {
                "$project": {
                    "total_active_power": {
                        "$cond": {
                            "if": {"$eq": [aggregation, 1]},  # Choose sum or avg based on aggregation
                            "then": "$total_active_power",
                            "else": "$average_active_power"
                        }
                    },
                    "_id": 0,
                    "weekday": "$_id"
                }
            },
            {
                "$sort": {"weekday": 1}  # Sort by weekday (Monday = 1, Sunday = 7)
            }
        ]

        results = list(GM_Hourly.aggregate(pipeline))

        # Map weekday numbers to names
        weekday_map = {
            1: "Monday", 2: "Tuesday", 3: "Wednesday",
            4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"
        }

        # Prepare the response
        response_data = []
        for result in results:
            weekday = weekday_map[result["weekday"]]  # Map weekday number to name
            total_power = result["total_active_power"]

            # Append to response data
            response_data.append({
                "weekday": weekday,
                "value": round(total_power, 2)
            })

        # Return the response
        return jsonify({
            "data": response_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/active_power_hourgroup', methods=['POST'])
def active_power_hourgroup():
    try:
        # Parse the request payload
        payload = request.json
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        plant = payload.get('plant')
        peakhour = payload.get('peakhour', 1.0)
        nonpeakhour = payload.get('nonpeakhour', 1.0)

        # Validate the input dates
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        # Query the data from MongoDB
        pipeline = [
            {
                "$match": {
                    "Day": {"$gte": start_date, "$lte": end_date},
                    "Plant":plant
                }
            },
            {
                "$project": {
                    "hour": {"$toInt": {"$arrayElemAt": [{"$split": ["$Day_Hour", " "]}, 1]}},  # Extract hour as integer
                    "active_power": 1
                }
            },
            {
                "$project": {
                    "hour_range": {
                        "$switch": {
                            "branches": [
                                {"case": {"$lt": ["$hour", 6]}, "then": "0-6"},
                                {"case": {"$lt": ["$hour", 12]}, "then": "6-12"},
                                {"case": {"$lt": ["$hour", 18]}, "then": "12-18"},
                                {"case": {"$lt": ["$hour", 24]}, "then": "18-24"}
                            ],
                            "default": "Unknown"
                        }
                    },
                    "active_power": 1
                }
            },
            {
                "$group": {
                    "_id": "$hour_range",  # Group by hour range
                    "total_active_power": {"$sum": "$active_power"}  # Sum active power
                }
            },
            {
                "$sort": {"_id": 1}  # Sort by hour range
            }
        ]

        results = list(GM_Hourly.aggregate(pipeline))

        # Prepare the response
        response_data = []
        for result in results:
            hour_range = result["_id"]  # Extract grouped hour range
            total_power = result["total_active_power"]

            # Determine the cost multiplier
            if hour_range in ["18-24"]:
                cost_multiplier = peakhour
            else:
                cost_multiplier = nonpeakhour

            cost = total_power * cost_multiplier

            # Append to response data
            response_data.append({
                "hour_range": hour_range,
                "value": round(total_power, 2),
                "cost": round(cost, 2),
            })

        # Return the response
        return jsonify({
            "data": response_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/active_power', methods=['POST'])
def calculate_active_power():
    try:
        # Parse the request payload
        payload = request.json
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        peakhour = payload.get('peakhour', 1.0)
        nonpeakhour = payload.get('nonpeakhour', 1.0)
        plant = payload.get('plant')

        # Validate the input dates
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        # Query the data from MongoDB
        pipeline = [
            {
                "$match": {
                    "Day": {"$gte": start_date, "$lte": end_date},
                    "Plant":plant
                }
            },
            {
                "$project": {
                    "hour": {"$arrayElemAt": [{"$split": ["$Day_Hour", " "]}, 1]},  # Extract hour
                    "active_power": 1
                }
            },
            {
                "$group": {
                    "_id": "$hour",  # Group by hour
                    "average_active_power": {"$sum": "$active_power"}  # Calculate average
                }
            },
            {
                "$sort": {"_id": 1}  # Sort by hour
            }
        ]

        results = list(GM_Hourly.aggregate(pipeline))

        # Prepare the response
        response_data = []
        for result in results:
            hour = result["_id"]  # Extract grouped hour
            average_power = result["average_active_power"]

            # Determine the cost multiplier
            hour_int = int(hour)
            if 0 <= hour_int <= 18 or hour_int == 23:
                cost_multiplier = nonpeakhour
            elif 19 <= hour_int <= 22:
                cost_multiplier = peakhour
            else:
                cost_multiplier = 0  # Should not happen, but just in case

            cost = average_power * cost_multiplier

            # Append to response data
            response_data.append({
                "hour": hour.zfill(2),  # Ensure hour is two digits
                "value": round(average_power, 2),
                "cost": round(cost, 2),
            })

        # Sort the response_data by hour
        response_data = sorted(response_data, key=lambda x: int(x["hour"]))

        # Return the response
        return jsonify({
            "data": response_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def sum_watt_string(sn):
    try:
        target_date = "2024-11-18"

        # Query MongoDB for the specified 'sn' and exact date
        query = {
            "dataItemMap.sn": sn,
            "dataItemMap.Day": target_date
        }

        # Fetch data from MongoDB
        records = overall_data.find(query)

        # Debugging: Print the records to ensure their structure
        for rec in records:
            print("DEBUG - Record fetched:", rec)

        # Reset the cursor for iteration (if needed)
        records = overall_data.find(query)

        # Calculate the sum of Watt/String
        total_watt_string = sum(
            rec.get("dataItemMap", {}).get("Watt/String", 0) if isinstance(rec, dict) else 0
            for rec in records
        )

        return {"sn": sn, "date": target_date, "total_watt_string": total_watt_string}

    except Exception as e:
        raise Exception(f"Error calculating sum for sn {sn}: {str(e)}")

def sum_watt_strings(sn, Strings):
    try:
        target_date = "2024-11-18"

        # Query MongoDB for the specified 'sn', 'Strings', and exact date
        query = {
            "dataItemMap.sn": sn,
            "dataItemMap.Strings": Strings,
            "dataItemMap.Day": target_date
        }

        # Fetch data from MongoDB
        records = overall_data.find(query)

        # Debugging: Print the records to ensure their structure
        for rec in records:
            print("DEBUG - Record fetched:", rec)

        # Reset the cursor for iteration (if needed)
        records = overall_data.find(query)

        # Calculate the sum of Watt/String
        total_watt_string = sum(
            rec.get("dataItemMap", {}).get("Watt/String", 0) if isinstance(rec, dict) else 0
            for rec in records
        )

        return {"sn": sn, "Strings": Strings, "date": target_date, "total_watt_string": total_watt_string}

    except Exception as e:
        raise Exception(f"Error calculating sum for sn {sn} and Strings {Strings}: {str(e)}")

def sum_watt_mppt(sn, mppt):
    try:
        target_date = "2024-11-18"

        # Query MongoDB for the specified 'sn', 'Strings', and exact date
        query = {
            "dataItemMap.sn": sn,
            "dataItemMap.MPPT": mppt,
            "dataItemMap.Day": target_date
        }

        # Fetch data from MongoDB
        records = overall_data.find(query)

        # Debugging: Print the records to ensure their structure
        for rec in records:
            print("DEBUG - Record fetched:", rec)

        # Reset the cursor for iteration (if needed)
        records = overall_data.find(query)

        # Calculate the sum of Watt/String
        total_watt_string = sum(
            rec.get("dataItemMap", {}).get("Watt/String", 0) if isinstance(rec, dict) else 0
            for rec in records
        )

        return {"sn": sn, "date": target_date, "total_watt_string": total_watt_string}

    except Exception as e:
        raise Exception(f"Error calculating sum for sn {sn} and Strings: {str(e)}")

@app.route('/sankey-data-mppts', methods=['POST'])
def sankey_data_mppts():
    try:
        payload = request.json
        plant_name = payload.get("Plant")
        dev_id = payload.get("devId")
        start_date = payload.get("startDate")
        end_date = payload.get("endDate")
        if not plant_name or not dev_id or not start_date or not end_date:
            return jsonify({"error": "Plant, devId, startDate, and EndDate are required"}), 400
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        records = overall_data.find({
            "dataItemMap.Plant": plant_name,
            "dataItemMap.sn": dev_id,
            "timestamp": {"$gte": start_date.strftime("%Y-%m-%d"), "$lte": end_date.strftime("%Y-%m-%d")}
        })
        total_value = 0
        mppt_values = {}
        string_values = {}
        for record in records:
            data_map = record.get("dataItemMap", {})
            P_abd = data_map.get("P_abd", 0)
            mppt = data_map.get("MPPT", "Unknown")
            string = data_map.get("Strings", "Unknown")
            sn = data_map.get("sn", "Unknown")
            if string == "General Tags":
                continue
            total_value += P_abd
            if mppt not in mppt_values:
                mppt_values[mppt] = 0
            mppt_values[mppt] += P_abd
            key = (sn, mppt, string)  # Include sn in the key
            if key not in string_values:
                string_values[key] = 0
            string_values[key] += P_abd
        sankey_data = []
        for mppt, value in mppt_values.items():
            sankey_data.append({
                "source": f"[bold]Device {dev_id}\n{round(total_value)} KW",
                "target": f"[bold]{mppt}\n{round(value)} KW",
                "value": round(value)
            })
        for (sn, mppt, string), value in string_values.items():
            sankey_data.append({
                "source": f"[bold]{mppt}\n{round(mppt_values[mppt])} KW",
                "target": f"[bold]{string}\n{round(value)} KW",
                "value": round(value)
            })
        def natural_key(item):
            target = item["target"]
            if "pv" in target:
                num = "".join(filter(str.isdigit, target))
                return (0, int(num)) if num.isdigit() else (1, target)
            elif "MPPT" in target:
                num = "".join(filter(str.isdigit, target))
                return (0, int(num)) if num.isdigit() else (1, target)
            return (1, target)
        sankey_data = sorted(sankey_data, key=natural_key)
        return jsonify(sankey_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/sankey-data', methods=['POST'])
def generate_sankey_data():
    try:
        # Parse the payload
        payload = request.json
        plant_name = payload.get("Plant")
        start_date = payload.get("startDate")
        end_date = payload.get("endDate")
        
        # Validate payload
        if not plant_name or not start_date or not end_date:
            return jsonify({"error": "Plant, startDate, and endDate are required"}), 400

        # Convert startDate and endDate to datetime objects
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Query MongoDB for the specified plant and date range
        plant_data = overall_data.find({
            "dataItemMap.Plant": plant_name,
            "timestamp": {
                "$gte": start_date.strftime("%Y-%m-%d"),
                "$lte": end_date.strftime("%Y-%m-%d")
            }
        })
        # First Level: Aggregate data for the Plant
        total_value = 0
        for record in plant_data:
            P_abd = record.get("dataItemMap", {}).get("P_abd", 0)
            total_value += P_abd  # Multiply u and i tags and sum them

        # Prepare first and second levels
        sankey_data = [
            {
                "source": f"[bold]{plant_name}\n{round(total_value)} KW",
                "target": f"[bold]Sub Plant\n{round(total_value)} KW",
                "value": round(total_value)
            }
        ]

        # Reset the cursor for reprocessing plant_data
        plant_data.rewind()

        # Third Level: Aggregate data by devId
        devId_values = {}
        for record in plant_data:
            dev_id = record.get("dataItemMap", {}).get("sn")
            P_abd = record.get("dataItemMap", {}).get("P_abd", 0)
            devId_values[dev_id] = devId_values.get(dev_id, 0) + (P_abd)

        # Add devId entries to the sankey_data
        for dev_id, value in devId_values.items():
            sankey_data.append({
                "source": f"[bold]Sub Plant\n{round(total_value)} KW",
                "target": f"[bold]{dev_id}\n{round(value)} KW",
                "value": round(value)
            })

        return jsonify(sankey_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_filtered_data(start_date, end_date):
    # Add time components to cover the entire day
    start_date_str = start_date.strftime("%Y-%m-%d")  # Convert datetime to string
    end_date_str = end_date.strftime("%Y-%m-%d") 
    
    return Plant_Day.find({
        "timestamp": {
            "$gte": start_date_str,
            "$lte": end_date_str
        }
    })

def calculate_data(data, group_field, calculation_field=None, operation="multiply"):
    result = {}

    for record in data:
        # Access nested fields in `dataItemMap`
        data_item_map = record.get("dataItemMap", {})
        key = data_item_map.get(group_field)  # Use `group_field` dynamically from `dataItemMap`

        if not key:  # Skip records where the `group_field` is missing
            continue

        if key in result:  # Skip if we already have the first value for this group
            continue

        if operation == "multiply":
            u = data_item_map.get("u", 0) or 0
            i = data_item_map.get("i", 0) or 0
            result[key] = u * i
        elif operation == "first" and calculation_field:
            value = data_item_map.get(calculation_field, None)
            if value is not None:  # Take the first valid value
                result[key] = value

    return result

@app.route('/sankey', methods=['POST'])
def generate_sankey():
    payload = request.json
    options = payload.get("options", [])
    start_date = datetime.strptime(payload.get("start_date"), "%Y-%m-%d")
    end_date = datetime.strptime(payload.get("end_date"), "%Y-%m-%d")
    
    if not options or not all(1 <= opt <= 5 for opt in options):
        return jsonify({"error": "Invalid options. Each must be between 1 and 5."}), 400
    
    # Fetch data based on date range
    data = list(get_filtered_data(start_date, end_date))
    # print(data)
    if not data:
        return jsonify({"error": "No data found for the specified date range."}), 404
    
    # Inline calculations
    province_totals = {}
    city_totals = {}
    plant_totals = {}
    plant_capacities = {}

    for record in data:
        data_item_map = record.get("dataItemMap", {})
        province = record.get("Province")
        city = record.get("City")
        plant = record.get("Plant")
        watt_string = record.get("Plant Capacity")
        P_abd = data_item_map.get("inverter_power", 0)
        power = P_abd  # Multiply voltage and current
        
        # Province-level aggregation
        if province:
            province_totals[province] = province_totals.get(province, 0) + power

            # City-level aggregation under each province
            if city:
                if province not in city_totals:
                    city_totals[province] = {}
                city_totals[province][city] = city_totals[province].get(city, 0) + power
                
                # Plant-level aggregation under each city
                if plant:
                    if province not in plant_totals:
                        plant_totals[province] = {}
                    if city not in plant_totals[province]:
                        plant_totals[province][city] = {}
                    plant_totals[province][city][plant] = plant_totals[province][city].get(plant, 0) + power

                    # Plant capacity aggregation (use first valid watt_string)
                    if watt_string and plant not in plant_capacities:
                        plant_capacities[plant] = watt_string
    
    # Construct Sankey data
    sankey_data = []
    overall_total = sum(province_totals.values())

    if 1 in options and 2 in options:
        # Overall -> Provinces
        for province, value in province_totals.items():
            sankey_data.append({
                "source": f"[bold]Nationwide\n{round(overall_total)} KW",
                "target": f"[bold]{province}\n{round(value)} KW",
                "value": value
            })
    if 1 in options and 3 in options and 2 not in options:
        # Provinces -> Cities
        for province, cities in city_totals.items():
            total_city_values = sum(cities.values())
            for city, value in cities.items():
                sankey_data.append({
                    "source": f"[bold]Nationwide\n{round(overall_total)} KW",
                    "target": f"[bold]{city}\n{round(value)} KW",
                    "value": value
                })


    if 2 in options and 3 in options:
        # Provinces -> Cities
        for province, cities in city_totals.items():
            total_city_values = sum(cities.values())
            for city, value in cities.items():
                sankey_data.append({
                    "source": f"[bold]{province}\n{round(total_city_values)} KW",
                    "target": f"[bold]{city}\n{round(value)} KW",
                    "value": value
                })

    if 2 in options and 4 in options:
        # Provinces -> Plantcapacity
        for province, cities in plant_totals.items():
            province_total = province_totals[province]
            for city, plants in cities.items():
                for plant_name, value in plants.items():
                    plant_capacity = plant_capacities.get(plant_name, 0)
                    sankey_data.append({
                        "source": f"[bold]{province}\n{round(province_total)} KW",
                        "target": f"[bold]{plant_name}\n{round(plant_capacity)} KW",
                        "value": plant_capacity
                    })
                    
    if 2 in options and 5 in options:
        # Provinces -> Plants
        for province, cities in plant_totals.items():
            province_total = province_totals[province]
            for city, plants in cities.items():
                for plant_name, value in plants.items():
                    sankey_data.append({
                        "source": f"[bold]{province}\n{round(province_total)} KW",
                        "target":f"[bold]{plant_name}\n{round(value)} KW",
                        "value": value
                    })
    if 3 in options and 4 in options:
        # Cities → Plants
        for province, cities in plant_totals.items():
            for city, plants in cities.items():
                for plant_name, value in plants.items():
                    plant_capacity = plant_capacities.get(plant_name, 0)
                    # Ensure City → Plant connection
                    sankey_data.append({
                        "source": f"[bold]{city}\n{round(city_totals[province][city])} KW",
                        "target": f"[bold]{plant_name}\n{round(plant_capacity)} KW",
                        "value": plant_capacity
                    })

    if 4 in options and 5 in options:
        # Plants → Plant Capacities
        for province, cities in plant_totals.items():
            for city, plants in cities.items():
                for plant_name, plant_value in plants.items():
                    plant_capacity = plant_capacities.get(plant_name, 0)
                    # Skip self-referential nodes
                    sankey_data.append({
                        "source": f"[bold]{plant_name}\n{round(plant_capacity)} KW",
                        "target": f"[bold]{plant_name}\n{round(city_totals[province][city])} KW",
                        "value": plant_value
                    })

    if 3 in options and 5 in options and 4 not in options:
        # Cities -> Plants
        for province, cities in plant_totals.items():
            for city, plants in cities.items():
                for plant_name, value in plants.items():
                    sankey_data.append({
                        "source": f"[bold]{city}\n{round(city_totals[province][city])} KW",
                        "target": f"[bold]{plant_name}\n{round(value)} KW",
                        "value": value
                    })
    if 4 in options and len(options) == 1:
        # Plants only
        for province, cities in plant_totals.items():
            for city, plants in cities.items():
                for plant_name, value in plants.items():
                    sankey_data.append({
                        "source": "[bold]Plant",
                        "target": f"[bold]{plant_name}\n{round(value, 2)} KW",
                        "value": value
                    })

 

    return jsonify(sankey_data)

@app.route("/fetch-data", methods=["POST"])
def fetch_data():
    try:
        # Parse payload
        payload = request.json
        resolution_option = payload.get("resolution_option")
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        plant = payload.get("plant")
        inverter = payload.get("inverter")
        string = payload.get("string")

        if resolution_option != 1:
            return jsonify({"error": "Invalid resolution_option"}), 400

        # Convert dates to datetime objects
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Build MongoDB aggregation pipeline
        pipeline = []

        # Add field to parse timestamp if it's stored as a string
        pipeline.append({
            "$addFields": {
                "parsed_timestamp": {
                    "$dateFromString": {
                        "dateString": "$timestamp",
                        "format": "%Y-%m-%d %H:%M:%S"
                    }
                }
            }
        })

        # Match stage to filter data within the given date range
        match_stage = {
            "$match": {
                "parsed_timestamp": {"$gte": start_date, "$lte": end_date}
            }
        }

        if plant and not inverter and not string:
            match_stage["$match"]["Plant"] = plant

        pipeline.append(match_stage)

        # Project stage to compute hourly power (u * i)
        pipeline.append({
            "$project": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$parsed_timestamp"}},
                "hour": "$Hour",
                "power": {"$multiply": [{"$ifNull": ["$u", 0]}, {"$ifNull": ["$i", 0]}]}
            }
        })

        # Group stage to sum power by date and hour
        pipeline.append({
            "$group": {
                "_id": {"date": "$date", "hour": "$hour"},
                "total_power": {"$sum": "$power"}
            }
        })

        # Sort stage to ensure data is ordered by date and hour
        pipeline.append({
            "$sort": {
                "_id.date": 1,
                "_id.hour": 1
            }
        })

        # Run the aggregation pipeline
        aggregated_data = device_hour_collection.aggregate(pipeline)

        # Process pipeline result into the desired output format
        result = {}
        for record in aggregated_data:
            date = record["_id"]["date"]
            hour = record["_id"]["hour"]
            power = record["total_power"]

            if date not in result:
                result[date] = [0] * 24  # Initialize 24 hours

            result[date][hour] = power

        # Format the output for the frontend
        output = [
            {
                "date": date,
                "hourly_values": [{"hour": hour, "value": value} for hour, value in enumerate(hourly_values)]
            }
            for date, hourly_values in result.items()
        ]

        return jsonify(output), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_hourly_values', methods=['POST'])
def get_hourly_values():
    # Parse payload
    payload = request.json
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    plant = payload.get("plant")
    inverter = payload.get("inverter")
    mppt = payload.get("mppt")
    string = payload.get("string")

    # Convert dates to strings for filtering
    start_date = datetime.strptime(start_date, '%Y-%m-%d').isoformat()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').isoformat()

    # Build the query
    query = {
        "$or": [
            {"Day_Hour": {"$regex": f"^{start_date[:10]}"}},
            {"Day_Hour": {"$regex": f"^{end_date[:10]}"}}
        ]
    }
    if plant:
        query["Plant"] = plant
    if inverter:
        query["sn"] = inverter
    if mppt:
        query["MPPT"] = mppt
    if string:
        query["Strings"] = string

    # Determine grouping conditions
    grouping_conditions = {
        "date": {"$substr": ["$Day_Hour", 0, 10]},  # Extract YYYY-MM-DD from Day_Hour
        "hour": {"$toInt": {"$substr": ["$Day_Hour", 11, 2]}}  # Extract HH from Day_Hour and convert to integer
    }
    if plant:
        grouping_conditions["plant"] = "$Plant"
    if inverter:
        grouping_conditions["inverter"] = "$sn"
    if mppt:
        grouping_conditions["mppt"] = "$MPPT"
    if string:
        grouping_conditions["string"] = "$Strings"

    # MongoDB aggregation pipeline
    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": grouping_conditions,
                "avg_u": {"$sum": "$P_abd"}  # Calculate the average of field 'u'
            }
        },
        {
            "$group": {
                "_id": "$_id.date",
                "hourly_values": {
                    "$push": {
                        "hour": "$_id.hour",
                        "value": "$avg_u"
                    }
                }
            }
        },
        {"$sort": {"_id": 1}}
    ]

    # Execute the aggregation
    results = list(device_hour_collection.aggregate(pipeline))

    # Format the response
    response = [
        {
            "date": result["_id"],
            "hourly_values": result["hourly_values"]
        }
        for result in results
    ]

    return jsonify(response)

@app.route("/aggregate-data-single", methods=["POST"])
def aggregate_data_single():
    try:
        # Get the payload
        payload = request.json

        # Parse and validate dates
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        # Convert dates to datetime objects
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = datetime.combine(start_date, datetime.min.time())  # Add 00:00:00
        end_date = datetime.combine(end_date, datetime.max.time()) 

        # Build the match query
        query = {
            "timestamp": {
                "$gte": start_date.strftime("%Y-%m-%d %H:%M:%S"),
                "$lte": end_date.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        if payload.get("plant"):
            query["Plant"] = payload["plant"]
        if payload.get("inverter"):
            query["sn"] = payload["inverter"]
        if payload.get("mppt"):
            query["MPPT"] = payload["mppt"]
        if payload.get("string"):
            query["Strings"] = payload["string"]

        # Build the grouping key
        group_id = {
            "timestamp": {
                "$dateToString": {"format": "%Y-%m-%dT%H:%M:%S", "date": "$timestamp"}
            }
        }
        if payload.get("plant"):
            group_id["Plant"] = "$Plant"
        if payload.get("inverter"):
            group_id["sn"] = "$sn"
        if payload.get("mppt"):
            group_id["MPPT"] = "$MPPT"
        if payload.get("string"):
            group_id["Strings"] = "$Strings"

        # Aggregation pipeline
        pipeline = [
        {"$match": query},
        # Step 1: Convert timestamp string to date
        {
            "$addFields": {
                "timestamp_date": {
                    "$dateFromString": {
                        "dateString": "$timestamp",
                        "format": "%Y-%m-%d %H:%M:%S"
                    }
                }
            }
        },
        # Step 2: Group by converted timestamp and other fields
        {
            "$group": {
                "_id": {
                    "Plant": "$Plant" if payload.get("plant") else None,
                    "sn": "$sn" if payload.get("inverter") else None,
                    "MPPT": "$MPPT" if payload.get("mppt") else None,
                    "Strings": "$Strings" if payload.get("string") else None,
                    "timestamp": {
                        "$dateToString": {
                            "format": "%Y-%m-%dT%H:%M:%S",
                            "date": "$timestamp_date"
                        }
                    }
                },
                "weighted_avg": {
                    "$avg": {"$multiply": ["$u", "$i"]}
                }
            }
        },
        # Step 3: Sort results by timestamp
        {"$sort": {"_id.timestamp": 1}}
    ]

        # Execute the aggregation
        results = list(collection2.aggregate(pipeline))

        # Format the response to match the desired structure
        output = [
            {
                "date": datetime.strptime(result["_id"]["timestamp"], "%Y-%m-%dT%H:%M:%S"),
                "value1": result["weighted_avg"]
            }
            for result in results
        ]

        return jsonify(output), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/aggregate-data1", methods=["POST"])
def aggregate_data1():
    try:
        payload = request.json
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        if not start_date or not end_date:
            return jsonify({"error": "start_date and end_date are required"}), 400

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())

        query = {
            "timestamp": {
                "$gte": start_date.strftime("%Y-%m-%d %H:%M:%S"),
                "$lte": end_date.strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        if payload.get("plant"):
            query["Plant"] = payload["plant"]
        if payload.get("inverter"):
            query["sn"] = payload["inverter"]
        if payload.get("mppt"):
            query["MPPT"] = payload["mppt"]

        pipeline = []
        mppt_filter = {
            "$expr": {
                "$and": [
                    {"$ne": ["$MPPT", None]},
                    {"$ne": ["$MPPT", float("nan")]}
                ]
            }
        }

        if payload.get("plant") and payload.get("inverter") and payload.get("mppt"):
            pipeline = [
                {"$match": {**query, **mppt_filter}},
                {
                    "$group": {
                        "_id": {
                            "timestamp": "$timestamp",
                            "Strings": "$Strings",
                            "Plant": payload["plant"],
                            "sn": payload["inverter"],
                            "mppt": payload["mppt"]
                        },
                        "value1": {"$avg": {"$multiply": ["$u", "$i"]}}
                    }
                },
                {
                    "$group": {
                        "_id": "$_id.timestamp",
                        "values": {
                            "$push": {
                                "Strings": "$_id.Strings",
                                "value1": "$value1",
                                "Plant": "$_id.Plant",
                                "sn": "$_id.sn",
                                "mppt": "$_id.mppt"
                            }
                        }
                    }
                },
                {"$sort": {"_id": 1}}
            ]
        elif payload.get("mppt") is None and payload.get("plant") and payload.get("inverter"):
            pipeline = [
                {"$match": {**query, **mppt_filter}},
                {
                    "$group": {
                        "_id": {
                            "timestamp": "$timestamp",
                            "mppt": "$MPPT",
                            "plant": payload["plant"],
                            "sn": payload["inverter"]
                        },
                        "value1": {"$avg": {"$multiply": ["$u", "$i"]}}
                    }
                },
                {
                    "$group": {
                        "_id": "$_id.timestamp",
                        "values": {
                            "$push": {
                                "sn": "$_id.sn",
                                "value1": "$value1",
                                "mppt": "$_id.mppt",
                                "plant": "$_id.plant"
                            }
                        }
                    }
                },
                {"$sort": {"_id": 1}}
            ]
        else:
            pipeline = [
                {"$match": {**query, **mppt_filter}},
                {
                    "$group": {
                        "_id": {
                            "timestamp": "$timestamp",
                            "sn": "$sn"
                        },
                        "value1": {"$avg": {"$multiply": ["$u", "$i"]}}
                    }
                },
                {
                    "$group": {
                        "_id": "$_id.timestamp",
                        "values": {
                            "$push": {
                                "sn": "$_id.sn",
                                "value1": "$value1"
                            }
                        }
                    }
                },
                {"$sort": {"_id": 1}}
            ]

        results = list(collection2.aggregate(pipeline))
        output = []

        if results:
            for result in results:
                timestamp = result["_id"]
                values = result.get("values", [])
                formatted_values = []

                for value in values:
                    sn = None
                    if payload.get("plant") and payload.get("inverter") and payload.get("mppt"):
                        sn = value["Strings"]
                    elif payload.get("plant") and payload.get("inverter"):
                        sn = value["mppt"]
                    elif payload.get("plant"):
                        sn = value["sn"]
                    formatted_value = {
                        "sn": sn,
                        "value1": value["value1"],
                        "Strings": value.get("Strings", ""),
                        "Plant": value.get("Plant", ""),
                        "mppt": value.get("mppt", "")
                    }
                    formatted_values.append(formatted_value)

                output.append({
                    "timestamp": timestamp,
                    "values": formatted_values
                })
        else:
            output.append({
                "timestamp": start_date.strftime("%Y-%m-%d %H:%M:%S"),
                "values": [{"sn": "Unknown", "value1": 0.0, "Strings": "", "Plant": "", "mppt": ""}]
            })

        return jsonify(output), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-devices', methods=['POST'])
def get_devices():
    try:
        payload = request.json
        station = payload.get('station')
        if not station:
            return jsonify({"error": "Station is required"}), 400
        pipeline = [
            {"$match": {"dataItemMap.Plant": station}},
            {"$group": {"_id": "$dataItemMap.sn"}},
            {"$project": {"_id": 0, "value": "$_id", "label": "$_id"}}
        ]
        results = list(overall_data.aggregate(pipeline))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-mppt', methods=['POST'])
def get_mppt():
    try:
        payload = request.json
        dev_id = payload.get('devId')
        if not dev_id:
            return jsonify({"error": "devId is required"}), 400

        pipeline = [
            {"$match": {"dataItemMap.sn": dev_id}},  # Filter by devId
            {"$match": {"dataItemMap.MPPT": {"$exists": True, "$ne": None}}},  # Exclude null and missing MPPT
            {"$match": {"dataItemMap.MPPT": {"$type": "string"}}},  # Include only string MPPT values
            {"$group": {"_id": "$dataItemMap.MPPT"}},  # Group by MPPT
            {"$project": {"_id": 0, "value": "$_id", "label": "$_id"}}  # Format the output
        ]

        results = list(overall_data.aggregate(pipeline))
        return jsonify(results)  # Send the result as JSON response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-strings', methods=['POST'])
def get_strings():
    try:
        payload = request.json
        dev_id = payload.get('devId')
        mppt = payload.get('mppt')

        if not dev_id or not mppt:
            return jsonify({"error": "devId and mppt are required"}), 400

        pipeline = [
            {"$match": {"dataItemMap.sn": dev_id, "dataItemMap.MPPT": mppt}},  # Correct $match syntax
            {"$group": {"_id": "$dataItemMap.Strings"}},  # Group by Strings
            {
                "$project": {
                    "_id": 0,
                    "value": "$_id",
                    "label": "$_id"
                }
            }
        ]

        results = list(overall_data.aggregate(pipeline))
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)
