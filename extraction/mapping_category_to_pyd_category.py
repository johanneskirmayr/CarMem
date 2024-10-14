def category_to_pyd_category_sub_extra(input_string, category):
    # Define the mapping of strings to numbers
    if category == "main_category":
        mapping = {
            "Points of Interest": "PointsOfInterest",
            "Navigation and Routing": "NavigationAndRouting",
            "Vehicle Settings and Comfort": "VehicleSettingsAndComfort",
            "Entertainment and Media": "EntertainmentAndMedia",
        }
    elif category == "subcategory":
        mapping = {
            "Restaurant": "Restaurant",
            "Gas Station": "GasStation",
            "Charging Station(in public)": "ChargingStation",
            "Grocery Shopping": "GroceryShopping",
            "Routing": "Routing",
            "Traffic and Conditions": "TrafficAndConditions",
            "Parking": "Parking",
            "Climate Control": "ClimateControl",
            "Lighting and Ambience": "LightingAndAmbience",
            "Music": "Music",
            "Radio and Podcasts": "RadioAndPodcast",
        }
    elif category == "detail_category":
        mapping = {
            "Favorite Cuisine": "favourite_cuisine",
            "Preferred Restaurant Type": "preferred_restaurant_type",
            "Fast Food Preference": "fast_food_preference",
            "Desired Price Range": "desired_price_range",
            "Dietary Preferences": "dietary_preference",
            "Preferred Payment method": "preferred_payment_method",
            "Preferred Gas Station": "preferred_gas_station",
            "Willingness to Pay Extra for Green Fuel": "willingness_to_pay_extra_for_green_fuel",
            "Price Sensitivity for Fuel": "price_sensitivity_for_fuel",
            "Preferred Charging Network": "preferred_charging_network",
            "Preferred type of Charging while traveling": "preferred_type_of_charging_while_traveling",
            "Preferred type of Charging when being at everyday points (f.e. work, grocery, restaurant)": "preferred_type_of_charging_at_everyday_points",
            "Charging Station Amenities": "charging_station_onsite_amenities",
            "Preferred Supermarket Chains": "preferred_supermarket_chain",
            "Preference for Local Markets/Farms or Supermarket": "preference_for_local_markets_farms_or_supermarket",
            "Avoidance of Specific Road Types": "avoidance_of_specific_road_types",
            "Priority for Shortest Time or Shortest Distance": "priority_for_shortest_time_or_shortest_distance",
            "Tolerance for Traffic": "tolerance_for_traffic",
            "Traffic Information Source Preferences": "traffic_information_source_preferences",
            "Willingness to Take Longer Route to Avoid Traffic": "willingness_to_take_longer_route_to_avoid_traffic",
            "Preferred Parking Type": "preferred_parking_type",
            "Price Sensitivity for Paid Parking": "price_sensitivity_for_paid_parking",
            "Distance Willing to Walk from Parking to Destination": "distance_willing_to_walk_from_parking_to_destination",
            "Preference for Covered Parking": "preference_for_covered_parking",
            "Need for Handicapped Accessible Parking": "need_for_handicapped_accessible_parking",
            "Preference for Parking with Security": "preference_for_parking_with_security",
            "Preferred Temperature": "preferred_temperature",
            "Fan Speed Preferences": "fan_speed_preferences",
            "Airflow Direction Preferences": "airflow_direction_preferences",
            "Seat Heating Preferences": "seat_heating_preferences",
            "Interior Lighting Brightness Preferences": "interior_lighting_brightness_preferences",
            "Interior Lighting Ambient Preferences": "interior_lighting_ambient_preferences",
            "Interior Lightning Color Preferences": "interior_lighting_color_preferences",
            "Favorite Genres": "favorite_genres",
            "Favorite Artists/Bands": "favorite_artists_or_bands",
            "Favorite Songs": "favorite_songs",
            "Preferred Music Streaming Service": "preferred_music_streaming_service",
            "Preferred Radio Station": "preferred_radio_station",
            "Favorite Podcast Genres": "favorite_podcast_genres",
            "Favorite Podcast Shows": "favorite_podcast_shows",
            "General News Source": "general_news_source",
        }
    else:
        mapping = {}

    # Check if the input string is in the mapping
    if input_string in mapping:
        mapping_result = mapping[input_string]
        return mapping_result
    else:
        # Raise an exception if the string is not found
        raise ValueError(f"No mapping found for string: {input_string}")


def category_to_pyd_category(input_string, category):
    # Define the mapping of strings to numbers
    if category == "main_category":
        mapping = {
            "Points of Interest": "points_of_interest",
            "Navigation and Routing": "navigation_and_routing",
            "Vehicle Settings and Comfort": "vehicle_settings_and_comfort",
            "Entertainment and Media": "entertainment_and_media",
        }
    elif category == "subcategory":
        mapping = {
            "Restaurant": "restaurant",
            "Gas Station": "gas_station",
            "Charging Station(in public)": "charging_station",
            "Grocery Shopping": "grocery_shopping",
            "Routing": "routing",
            "Traffic and Conditions": "traffic_and_conditions",
            "Parking": "parking",
            "Climate Control": "climate_control",
            "Lighting and Ambience": "lighting_and_ambience",
            "Music": "music",
            "Radio and Podcasts": "radio_and_podcast",
        }
    elif category == "detail_category":
        mapping = {
            "Favorite Cuisine": "favourite_cuisine",
            "Preferred Restaurant Type": "preferred_restaurant_type",
            "Fast Food Preference": "fast_food_preference",
            "Desired Price Range": "desired_price_range",
            "Dietary Preferences": "dietary_preference",
            "Preferred Payment method": "preferred_payment_method",
            "Preferred Gas Station": "preferred_gas_station",
            "Willingness to Pay Extra for Green Fuel": "willingness_to_pay_extra_for_green_fuel",
            "Price Sensitivity for Fuel": "price_sensitivity_for_fuel",
            "Preferred Charging Network": "preferred_charging_network",
            "Preferred type of Charging while traveling": "preferred_type_of_charging_while_traveling",
            "Preferred type of Charging when being at everyday points (f.e. work, grocery, restaurant)": "preferred_type_of_charging_at_everyday_points",
            "Charging Station Amenities": "charging_station_onsite_amenities",
            "Preferred Supermarket Chains": "preferred_supermarket_chain",
            "Preference for Local Markets/Farms or Supermarket": "preference_for_local_markets_farms_or_supermarket",
            "Avoidance of Specific Road Types": "avoidance_of_specific_road_types",
            "Priority for Shortest Time or Shortest Distance": "priority_for_shortest_time_or_shortest_distance",
            "Tolerance for Traffic": "tolerance_for_traffic",
            "Traffic Information Source Preferences": "traffic_information_source_preferences",
            "Willingness to Take Longer Route to Avoid Traffic": "willingness_to_take_longer_route_to_avoid_traffic",
            "Preferred Parking Type": "preferred_parking_type",
            "Price Sensitivity for Paid Parking": "price_sensitivity_for_paid_parking",
            "Distance Willing to Walk from Parking to Destination": "distance_willing_to_walk_from_parking_to_destination",
            "Preference for Covered Parking": "preference_for_covered_parking",
            "Need for Handicapped Accessible Parking": "need_for_handicapped_accessible_parking",
            "Preference for Parking with Security": "preference_for_parking_with_security",
            "Preferred Temperature": "preferred_temperature",
            "Fan Speed Preferences": "fan_speed_preferences",
            "Airflow Direction Preferences": "airflow_direction_preferences",
            "Seat Heating Preferences": "seat_heating_preferences",
            "Interior Lighting Brightness Preferences": "interior_lighting_brightness_preferences",
            "Interior Lighting Ambient Preferences": "interior_lighting_ambient_preferences",
            "Interior Lightning Color Preferences": "interior_lighting_color_preferences",
            "Favorite Genres": "favorite_genres",
            "Favorite Artists/Bands": "favorite_artists_or_bands",
            "Favorite Songs": "favorite_songs",
            "Preferred Music Streaming Service": "preferred_music_streaming_service",
            "Preferred Radio Station": "preferred_radio_station",
            "Favorite Podcast Genres": "favorite_podcast_genres",
            "Favorite Podcast Shows": "favorite_podcast_shows",
            "General News Source": "general_news_source",
        }
    else:
        mapping = {}

    # Check if the input string is in the mapping
    if input_string in mapping:
        mapping_result = mapping[input_string]
        return mapping_result
    else:
        # Raise an exception if the string is not found
        raise ValueError(f"No mapping found for string: {input_string}")
