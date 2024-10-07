import numpy as np

def string_to_number(input_string, category) -> int:
   # Define the mapping of strings to numbers
    if category=="main_category":
        mapping = {
            "Points of Interest": 0,
            "points_of_interest": 0,
            "Navigation and Routing": 1,
            "navigation_and_routing": 1,
            "Vehicle Settings and Comfort": 2,
            "vehicle_settings_and_comfort": 2,
            "Entertainment and Media": 3,
            "entertainment_and_media": 3,
            "No Main Category": 4,
        }
    elif category=="subcategory":
        mapping = {
            "Restaurant": 4,
            "restaurant": 4,
            "Gas Station": 5,
            "gas_station": 5,
            "Charging Station(in public)": 6,
            "charging_station": 6,
            "Grocery Shopping": 7,
            "grocery_shopping": 7,
            "Routing": 8,
            "routing": 8,
            "Traffic and Conditions": 9,
            "traffic_and_conditions": 9,
            "Parking": 10,
            "parking": 10,
            "Climate Control": 11,
            "climate_control": 11,
            "Lighting and Ambience": 12,
            "lighting_and_ambience": 12,
            "Music": 13,
            "music": 13,
            "Radio and Podcasts": 14,
            "radio_and_podcast": 14,
        }
    elif category=="detail_category":
        mapping = {
            "Favorite Cuisine": 15,
            "favourite_cuisine": 15,
            "Preferred Restaurant Type": 16,
            "preferred_restaurant_type": 16,
            "Fast Food Preference": 17,
            "fast_food_preference": 17,
            "Desired Price Range": 18,
            "desired_price_range": 18,
            "Dietary Preferences": 19,
            "dietary_preference": 19,
            "Preferred Payment method": 20,
            "preferred_payment_method": 20,
            "Preferred Gas Station": 21,
            "preferred_gas_station": 21,
            "Willingness to Pay Extra for Green Fuel": 22,
            "willingness_to_pay_extra_for_green_fuel": 22,
            "Price Sensitivity for Fuel": 23,
            "price_sensitivity_for_fuel": 23,
            "Preferred Charging Network": 24,
            "preferred_charging_network": 24,
            "Preferred type of Charging while traveling": 25,
            "preferred_type_of_charging_while_traveling": 25,
            "Preferred type of Charging when being at everyday points (f.e. work, grocery, restaurant)": 26,
            "preferred_type_of_charging_at_everyday_points": 26,
            "Charging Station Amenities": 27,
            "charging_station_onsite_amenities": 27,
            "Preferred Supermarket Chains": 28,
            "preferred_supermarket_chain": 28,
            "Preference for Local Markets/Farms or Supermarket": 29,
            "preference_for_local_markets_farms_or_supermarket": 29,
            "Avoidance of Specific Road Types": 30,
            "avoidance_of_specific_road_types": 30,
            "Priority for Shortest Time or Shortest Distance": 31,
            "priority_for_shortest_time_or_shortest_distance": 31,
            "Tolerance for Traffic": 32,
            "tolerance_for_traffic": 32,
            "traffic_information_source_preferences": 33,
            "Traffic Information Source Preferences": 33,
            "Willingness to Take Longer Route to Avoid Traffic": 34,
            "willingness_to_take_longer_route_to_avoid_traffic": 34,
            "Preferred Parking Type": 35,
            "preferred_parking_type": 35,
            "Price Sensitivity for Paid Parking": 36,
            "price_sensitivity_for_paid_parking": 36,
            "Distance Willing to Walk from Parking to Destination": 37,
            "distance_willing_to_walk_from_parking_to_destination": 37,
            "Preference for Covered Parking": 38,
            "preference_for_covered_parking": 38,
            "Need for Handicapped Accessible Parking": 39,
            "need_for_handicapped_accessible_parking": 39,
            "Preference for Parking with Security": 40,
            "preference_for_parking_with_security": 40,
            "Preferred Temperature": 41,
            "preferred_temperature": 41,
            "Fan Speed Preferences": 42,
            "fan_speed_preferences": 42,
            "Airflow Direction Preferences": 43,
            "airflow_direction_preferences": 43,
            "Seat Heating Preferences": 44,
            "seat_heating_preferences": 44,
            "Interior Lighting Brightness Preferences": 45,
            "interior_lighting_brightness_preferences": 45,
            "Interior Lighting Ambient Preferences": 46,
            "interior_lighting_ambient_preferences": 46,
            "Interior Lightning Color Preferences": 47,
            "interior_lighting_color_preferences": 47,
            "Favorite Genres": 48,
            "favorite_genres": 48,
            "Favorite Artists/Bands": 49,
            "favorite_artists_or_bands": 49,
            "Favorite Songs": 50,
            "favorite_songs": 50,
            "Preferred Music Streaming Service": 51,
            "preferred_music_streaming_service": 51,
            "Preferred Radio Station": 52,
            "preferred_radio_station": 52,
            "Favorite Podcast Genres": 53,
            "favorite_podcast_genres": 53,
            "Favorite Podcast Shows": 54,
            "favorite_podcast_shows": 54,
            "General News Source": 55,
            "general_news_source": 55,
            "other": 56
        }
    else:
        mapping = {}
   
    # Check if the input string is in the mapping
    if input_string in mapping:
        return int(mapping[input_string])
    else:
        # Raise an exception if the string is not found
        raise ValueError(f"No mapping found for string: {input_string}")
    
def convert_preference_to_labels(preference) -> np.ndarray:
    label_list = []
    for key, value in preference.items():
        label_list.append(string_to_number(input_string=value, category=key))
    return label_list