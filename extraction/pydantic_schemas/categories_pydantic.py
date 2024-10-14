"""
This file contains the definition of the function parameter schema defined with pydantic.
Note that the schema is rebuild, depending of the ground-truth subcategory should be in-schema or out-of-schema.
The actual ground-truth preference is always excluded from the examples.
"""

import json
from copy import deepcopy
from typing import List, Optional

from pydantic import BaseModel, Field

default = (None,)


def pretty_print_json(json_dict: dict, indent: int = 2):
    print(json.dumps(json_dict, indent=indent))


EXAMPLES_ORIGINAL = {
    "favourite_cuisine": ["Italian", "Chinese", "Mexican", "Indian", "American"],
    "preferred_restaurant_type": [
        "Fast food",
        "Casual dining",
        "Fine dining",
        "Buffet",
    ],
    "fast_food_preference": [
        "BiteBox Burgers",
        "GrillGusto",
        "SnackSprint",
        "ZippyZest",
        "WrapRapid",
    ],
    "desired_price_range": ["cheap", "normal", "expensive"],
    "dietary_preference": [
        "Vegetarian",
        "Vegan",
        "Gluten-Free",
        "Dairy-Free",
        "Halal",
        "Kosher",
        "Nut Allergies",
        "Seafood Allergies",
    ],
    "preferred_payment_method": ["Cash", "Card"],
    "preferred_gas_station": [
        "PetroLux",
        "FuelNexa",
        "GasGlo",
        "ZephyrFuel",
        "AeroPump",
    ],
    "willingness_to_pay_extra_for_green_fuel": ["Yes", "No"],
    "price_sensitivity_for_fuel": [
        "Always cheapest",
        "Rather cheapest",
        "Price is irrelevant",
    ],
    "preferred_charging_network": [
        "ChargeSwift",
        "EcoPulse Energy",
        "VoltRise Charging",
        "AmpFlow Solutions",
        "ZapGrid Power",
    ],
    "preferred_type_of_charging_while_traveling": ["AC", "DC", "HPC"],
    "preferred_type_of_charging_at_everyday_points": ["AC", "DC", "HPC"],
    "charging_station_onsite_amenities": [
        "On-site amenities (Restaurant/cafes)",
        "Wi-Fi availability",
        "Seating area",
        "Restroom facilities",
    ],
    "preferred_supermarket_chain": [
        "MarketMingle",
        "FreshFare Hub",
        "GreenGroove Stores",
        "BasketBounty Markets",
        "PantryPulse Retail",
    ],
    "preference_for_local_markets_farms_or_supermarket": [
        "Local Markets/Farms",
        "Supermarket",
    ],
    "avoidance_of_specific_road_types": ["Highways", "Toll roads", "Unpaved roads"],
    "priority_for_shortest_time_or_shortest_distance": [
        "Shortest Time",
        "Shortest Distance",
    ],
    "tolerance_for_traffic": ["Low", "Medium", "High"],
    "traffic_information_source_preferences": [
        "In-car system",
        "NavFlow Updates",
        "RouteWatch Alerts",
        "TrafficTrendz Insights",
    ],
    "willingness_to_take_longer_route_to_avoid_traffic": ["Yes", "No"],
    "preferred_parking_type": ["On-street", "Off-street", "Parking-house"],
    "price_sensitivity_for_paid_parking": [
        "Always considers price first",
        "Sometimes considers price",
        "Never considers price",
    ],
    "distance_willing_to_walk_from_parking_to_destination": [
        "less than 5 min",
        "less than 10 min",
        "not relevant",
    ],
    "preference_for_covered_parking": ["Yes", "Indifferent to Covered Parking"],
    "need_for_handicapped_accessible_parking": ["Yes"],
    "preference_for_parking_with_security": ["Yes", "Indifferent to Parking Security"],
    "preferred_temperature": ["18", "19", "20", "21", "22", "23", "24", "25"],
    "fan_speed_preferences": ["Low", "Medium", "High"],
    "airflow_direction_preferences": ["Face", "Feet", "Centric", "Combined"],
    "seat_heating_preferences": ["Low", "Medium", "High"],
    "interior_lighting_brightness_preferences": ["Low", "Medium", "High"],
    "interior_lighting_ambient_preferences": ["Warm", "Cool"],
    "interior_lighting_color_preferences": [
        "Red",
        "Blue",
        "Green",
        "Yellow",
        "White",
        "Pink",
    ],
    "favorite_genres": ["Pop", "Rock", "Jazz", "Classical", "Country", "Rap"],
    "favorite_artists_or_bands": [
        "Max Jettison (Pop)",
        "Melody Raven (Pop)",
        "Melvin Dunes (Jazz)",
        "Ludwig van Beatgroove (Classical)",
        "Wolfgang Amadeus Harmonix (Classical)",
        "Taylor Winds (Country/Pop)",
        "Ed Sherwood (Pop/Folk)",
        "TwoPacks (Rap)",
    ],
    "favorite_songs": [
        "Envision by Jon Lemon (Rock)",
        "Dreamer's Canvas by Lenny Visionary (Folk)",
        "Jenny's Dance by Max Rythmo (Disco)",
        "Clasp My Soul by The Harmonic Five (Soul)",
        "Echoes of the Heart by Adeena (R&B)",
        "Asphalt Anthems by Gritty Lyricist (Rap)",
        "Cosmic Verses by Nebula Rhymes (Hip-Hop/Rap)",
    ],
    "preferred_music_streaming_service": [
        "SonicStream",
        "MelodyMingle",
        "TuneTorrent",
        "HarmonyHive",
        "RhythmRipple",
    ],
    "preferred_radio_station": [
        "EchoWave FM",
        "RhythmRise Radio",
        "SonicSphere 101.5",
        "VibeVault 88.3",
        "HarmonyHaven 94.7",
    ],
    "favorite_podcast_genres": [
        "News",
        "Technology",
        "Entertainment",
        "Health",
        "Science",
    ],
    "favorite_podcast_shows": [
        "GlobalGlimpse News",
        "ComedyCraze",
        "ScienceSync",
        "FantasyFrontier",
        "WellnessWave",
    ],
    "general_news_source": [
        "NewsNexus",
        "WorldPulse",
        "CurrentConnect",
        "ReportRealm",
        "InfoInsight",
    ],
}

EXAMPLES = deepcopy(EXAMPLES_ORIGINAL)


class OutputFormat(BaseModel):
    user_sentence_preference_revealed: Optional[str] = Field(
        default=None,
        description="user sentence (exclude username) where the user revealed the preference, must be from user sentences, must include the 'user_preference'",
    )
    user_preference: Optional[str] = Field(
        default=None,
        description="The preference of the user, must be included in the 'user_sentence_preference_revealed'",
    )

    class Config:
        extra = "forbid"


# Points of Interest


class Restaurant(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Restaurant', or put here a preference that does not fit into the other categories.",
    )
    favourite_cuisine: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Favourite Cuisine'.",
        examples=EXAMPLES["favourite_cuisine"],
    )
    preferred_restaurant_type: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preferred Restaurant Type'.",
        examples=EXAMPLES["preferred_restaurant_type"],
    )
    fast_food_preference: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Fast Food Preference'.",
        examples=EXAMPLES["fast_food_preference"],
    )
    desired_price_range: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Desired Price Range'.",
        examples=EXAMPLES["desired_price_range"],
    )
    dietary_preference: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Dietary Preferences'.",
        examples=EXAMPLES["dietary_preference"],
    )
    preferred_payment_method: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preferred Payment method'.",
        examples=EXAMPLES["preferred_payment_method"],
    )

    class Config:
        extra = "forbid"


class GasStation(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Gas Station', or put here a preference that does not fit into the other categories.",
    )
    preferred_gas_station: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preferred Gas Stations'.",
        examples=EXAMPLES["preferred_gas_station"],
    )
    willingness_to_pay_extra_for_green_fuel: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Willingness to Pay Extra for Green Fuel'.",
        examples=EXAMPLES["willingness_to_pay_extra_for_green_fuel"],
    )
    price_sensitivity_for_fuel: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Price Sensitivity for Fuel'.",
        examples=EXAMPLES["price_sensitivity_for_fuel"],
    )

    class Config:
        extra = "forbid"


class ChargingStation(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Charging Station', or put here a preference that does not fit into the other categories.",
    )
    preferred_charging_network: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preferred Charging Network'.",
        examples=EXAMPLES["preferred_charging_network"],
    )
    preferred_type_of_charging_while_traveling: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preferred type of Charging while traveling'.",
        examples=EXAMPLES["preferred_type_of_charging_while_traveling"],
    )
    preferred_type_of_charging_at_everyday_points: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preferred type of Charging when being at everyday points (f.e. work, grocery, restaurant)'.",
        examples=EXAMPLES["preferred_type_of_charging_at_everyday_points"],
    )
    charging_station_onsite_amenities: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Charging Station Amenities: On-site amenities (Restaurant/cafes)'.",
        examples=EXAMPLES["charging_station_onsite_amenities"],
    )

    class Config:
        extra = "forbid"


class GroceryShopping(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Grocery Shopping', or put here a preference that does not fit into the other categories.",
    )
    preferred_supermarket_chain: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preferred Supermarket Chains'.",
        examples=EXAMPLES["preferred_supermarket_chain"],
    )
    preference_for_local_markets_farms_or_supermarket: Optional[List[OutputFormat]] = (
        Field(
            default=[],
            description="The user's preference in the topic 'Preference for Local Markets/Farms or Supermarket'.",
            examples=EXAMPLES["preference_for_local_markets_farms_or_supermarket"],
        )
    )

    class Config:
        extra = "forbid"


class PointsOfInterest(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Points of Interest', or put here a preference that does not fit into the other categories.",
    )
    restaurant: Optional[Restaurant] = Field(
        default=None,
        description="The user's preferences in the category 'Restaurant'. This includes preferences in the topics 'favourite_cuisine', 'preferred_restaurant_type', 'fast_food_preference', 'desired_price_range', 'dietary_preference', 'preferred_payment_method'.",
    )
    gas_station: Optional[GasStation] = Field(
        default=None,
        description="The user's preferences in the category 'Gas Station'. This includes preferences in the topics 'preferred_gas_station', 'willingness_to_pay_extra_for_green_fuel', 'price_sensitivity_for_fuel'.",
    )
    charging_station: Optional[ChargingStation] = Field(
        default=None,
        description="The user's preferences in the category 'Charging Station (in public)'. This includes preferences in the topics 'preferred_charging_network', 'preferred_type_of_charging_while_traveling', 'preferred_type_of_charging_at_everyday_points', 'charging_station_onsite_amenities'.",
    )
    grocery_shopping: Optional[GroceryShopping] = Field(
        default=None,
        description="The user's preferences in the category 'Grocery Shopping'. This includes preferences in the topics 'preferred_supermarket_chain', 'preference_for_local_markets_farms_or_supermarket'.",
    )

    class Config:
        extra = "forbid"


# Navigation and Routing


class Routing(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Routing', or put here a preference that does not fit into the other categories.",
    )
    avoidance_of_specific_road_types: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Avoidance of Specific Road Types'.",
        examples=EXAMPLES["avoidance_of_specific_road_types"],
    )
    priority_for_shortest_time_or_shortest_distance: Optional[List[OutputFormat]] = (
        Field(
            default=[],
            description="The user's preference in the topic 'Priority for Shortest Time or Shortest Distance'.",
            examples=EXAMPLES["priority_for_shortest_time_or_shortest_distance"],
        )
    )
    tolerance_for_traffic: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Tolerance for Traffic'. Only tolerance, not if he would take a longer route.",
        examples=EXAMPLES["tolerance_for_traffic"],
    )

    class Config:
        extra = "forbid"


class TrafficAndConditions(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Traffic And Conditions', or put here a preference that does not fit into the other categories.",
    )
    traffic_information_source_preferences: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Traffic Information Source Preferences', so where the user get's traffic information from.",
        examples=EXAMPLES["traffic_information_source_preferences"],
    )
    willingness_to_take_longer_route_to_avoid_traffic: Optional[List[OutputFormat]] = (
        Field(
            default=[],
            description="The user's preference in the topic 'Willingness to Take Longer Route to Avoid Traffic'. Only fill here, if traffic is explicitely mentioned, not general tolerance.",
            examples=EXAMPLES["willingness_to_take_longer_route_to_avoid_traffic"],
        )
    )

    class Config:
        extra = "forbid"


class Parking(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Parking', or put here a preference that does not fit into the other categories.",
    )
    preferred_parking_type: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preferred Parking Type'.",
        examples=EXAMPLES["preferred_parking_type"],
    )
    price_sensitivity_for_paid_parking: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Price Sensitivity for Paid Parking'.",
        examples=EXAMPLES["price_sensitivity_for_paid_parking"],
    )
    distance_willing_to_walk_from_parking_to_destination: Optional[
        List[OutputFormat]
    ] = Field(
        default=[],
        description="The user's preference in the topic 'Distance Willing to Walk from Parking to Destination'.",
        examples=EXAMPLES["distance_willing_to_walk_from_parking_to_destination"],
    )
    preference_for_covered_parking: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preference for Covered Parking'.",
        examples=EXAMPLES["preference_for_covered_parking"],
    )
    need_for_handicapped_accessible_parking: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Need for Handicapped Accessible Parking'.",
        examples=EXAMPLES["need_for_handicapped_accessible_parking"],
    )
    preference_for_parking_with_security: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preference for Parking with Security'.",
        examples=EXAMPLES["preference_for_parking_with_security"],
    )

    class Config:
        extra = "forbid"


class NavigationAndRouting(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Navigation and Routing', or put here a preference that does not fit into the other categories.",
    )
    routing: Optional[Routing] = Field(
        default=None,
        description="The user's preferences in the category 'Routing'. This includes preferences in the topics 'avoidance_of_specific_road_types', 'priority_for_shortest_time_or_shortest_distance', 'tolerance_for_traffic'.",
    )
    traffic_and_conditions: Optional[TrafficAndConditions] = Field(
        default=None,
        description="The user's preferences in the category 'Traffic and Conditions'. This includes preferences in the topics 'traffic_information_source_preferences', 'willingness_to_take_longer_route_to_avoid_traffic'.",
    )
    parking: Optional[Parking] = Field(
        default=None,
        description="The user's preferences in the category 'Parking'. This includes preferences in the topics 'preferred_parking_type', 'price_sensitivity_for_paid_parking', 'distance_willing_to_walk_from_parking_to_destination', 'preference_for_covered_parking', 'need_for_handicapped_accessible_parking', 'preference_for_parking_with_security'.",
    )

    class Config:
        extra = "forbid"


# Vehicle Settings and Comfort


class ClimateControl(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'ClimateControl', or put here a preference that does not fit into the other categories.",
    )
    preferred_temperature: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preferred Temperature'.",
        examples=EXAMPLES["preferred_temperature"],
    )
    fan_speed_preferences: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Fan Speed Preferences'.",
        examples=EXAMPLES["fan_speed_preferences"],
    )
    airflow_direction_preferences: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Airflow Direction Preferences'.",
        examples=EXAMPLES["airflow_direction_preferences"],
    )
    seat_heating_preferences: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Seat Heating Preferences'.",
        examples=EXAMPLES["seat_heating_preferences"],
    )

    class Config:
        extra = "forbid"


class LightingAndAmbience(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'LightingAndAmbience', or put here a preference that does not fit into the other categories.",
    )
    interior_lighting_brightness_preferences: Optional[List[OutputFormat]] = Field(
        default=[],
        description="Interior Lighting Brightness Preferences'.",
        examples=EXAMPLES["interior_lighting_brightness_preferences"],
    )
    interior_lighting_ambient_preferences: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Interior Lighting Ambient Preferences'.",
        examples=EXAMPLES["interior_lighting_ambient_preferences"],
    )
    interior_lighting_color_preferences: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Interior Lightning Color Preferences'.",
        examples=EXAMPLES["interior_lighting_color_preferences"],
    )

    class Config:
        extra = "forbid"


class VehicleSettingsAndComfort(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Vehicle Settings and Comfort', or put here a preference that does not fit into the other categories.",
    )
    climate_control: Optional[ClimateControl] = Field(
        default=None,
        description="The user's preferences in the category 'Climate Control'. This includes preferences in the topics 'preferred_temperature', 'fan_speed_preferences', 'airflow_direction_preferences', 'seat_heating_preferences'.",
    )
    lighting_and_ambience: Optional[LightingAndAmbience] = Field(
        default=None,
        description="The user's preferences in the category 'Lighting and Ambience'. This includes preferences in the topics 'interior_lighting_brightness_preferences', 'interior_lighting_ambient_preferences', 'interior_lighting_color_preferences'.",
    )

    class Config:
        extra = "forbid"


# Entereinment and Media


class Music(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Music', or put here a preference that does not fit into the other categories.",
    )
    favorite_genres: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Favorite Genres'.",
        examples=EXAMPLES["favorite_genres"],
    )
    favorite_artists_or_bands: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Favorite Artists or Bands'.",
        examples=EXAMPLES["favorite_artists_or_bands"],
    )
    favorite_songs: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Favorite Songs'.",
        examples=EXAMPLES["favorite_songs"],
    )
    preferred_music_streaming_service: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preferred Music Streaming Service'.",
        examples=EXAMPLES["preferred_music_streaming_service"],
    )

    class Config:
        extra = "forbid"


class RadioAndPodcast(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Radio And Podcast', or put here a preference that does not fit into the other categories.",
    )
    preferred_radio_station: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Preferred Radio Station'.",
        examples=EXAMPLES["preferred_radio_station"],
    )
    favorite_podcast_genres: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Favorite Podcast Genres'.",
        examples=EXAMPLES["favorite_podcast_genres"],
    )
    favorite_podcast_shows: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'Favorite Podcast Shows'.",
        examples=EXAMPLES["favorite_podcast_shows"],
    )
    general_news_source: Optional[List[OutputFormat]] = Field(
        default=[],
        description="The user's preference in the topic 'General News Source'.",
        examples=EXAMPLES["general_news_source"],
    )

    class Config:
        extra = "forbid"


class EntertainmentAndMedia(BaseModel):
    no_or_other_preferences: Optional[str] = Field(
        default=None,
        description="Put here 'No' if there is no preference in the conversation for the category 'Entertainment and Media', or put here a preference that does not fit into the other categories.",
    )
    music: Optional[Music] = Field(
        default=None,
        description="The user's preferences in the category 'Music'. This includes preferences in the topics 'favorite_genres', 'favorite_artists_or_bands', 'favorite_songs', 'preferred_music_streaming_service'.",
    )
    radio_and_podcast: Optional[RadioAndPodcast] = Field(
        default=None,
        description="The user's preferences in the category 'Radio and Podcast'. This includes preferences in the topics 'preferred_radio_station', 'favorite_podcast_genres', 'favorite_podcast_shows', 'general_news_source'.",
    )

    class Config:
        extra = "forbid"


class PreferencesFunctionOutput(BaseModel):
    points_of_interest: Optional[PointsOfInterest] = Field(
        default=None,
        title="Preferences Points of Interest",
        description="The user's preferences in the category 'Points of Interest'. This includes preferences in the topics 'restaurant', 'gas_station', 'charging_station', 'grocery_shopping'.",
    )
    navigation_and_routing: Optional[NavigationAndRouting] = Field(
        default=None,
        title="Preferences Navigation and Routing",
        description="The user's preferences in the category 'Navigation and Routing'. This includes preferences in the topics 'routing', 'traffic_and_conditions', 'parking'.",
    )
    vehicle_settings_and_comfort: Optional[VehicleSettingsAndComfort] = Field(
        default=None,
        title="Preferences Vehicle Settings and Comfort",
        description="The user's preferences in the category 'Vehicle Settings and Comfort'. This includes preferences in the topics 'climate_control', 'lighting_and_ambience'.",
    )
    entertainment_and_media: Optional[EntertainmentAndMedia] = Field(
        default=None,
        title="Preferences Entertainment and Media",
        description="The user's preferences in the category 'Entertainment and Media'. This includes preferences in the topics 'music', 'radio_and_podcast'.",
    )

    class Config:
        extra = "forbid"


def return_pydantic_schema(detail_category_variable, attribute):
    EXAMPLES = deepcopy(EXAMPLES_ORIGINAL)

    if (
        detail_category_variable == "preferred_temperature"
        or detail_category_variable == "willingness_to_pay_extra_for_green_fuel"
        or detail_category_variable
        == "willingness_to_take_longer_route_to_avoid_traffic"
    ):
        attribute = attribute.split(" ")[0].strip()
    if (
        detail_category_variable
        == "distance_willing_to_walk_from_parking_to_destination"
    ):
        attribute = attribute.split("(")[0].strip()
    if detail_category_variable and attribute:
        EXAMPLES[detail_category_variable].remove(attribute)

    class OutputFormat(BaseModel):
        user_sentence_preference_revealed: Optional[str] = Field(
            default=None,
            description="user sentence (exclude username) where the user revealed the preference, must be from user sentences, must include the 'user_preference'",
        )
        user_preference: Optional[str] = Field(
            default=None,
            description="The preference of the user, must be included in the 'user_sentence_preference_revealed'",
        )

        class Config:
            extra = "forbid"

    # Points of Interest

    class Restaurant(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Restaurant', or put here a preference that does not fit into the other categories.",
        )
        favourite_cuisine: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favourite Cuisine'.",
            examples=EXAMPLES["favourite_cuisine"],
        )
        preferred_restaurant_type: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Restaurant Type'.",
            examples=EXAMPLES["preferred_restaurant_type"],
        )
        fast_food_preference: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Fast Food Preference'.",
            examples=EXAMPLES["fast_food_preference"],
        )
        desired_price_range: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Desired Price Range'.",
            examples=EXAMPLES["desired_price_range"],
        )
        dietary_preference: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Dietary Preferences'.",
            examples=EXAMPLES["dietary_preference"],
        )
        preferred_payment_method: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Payment method'.",
            examples=EXAMPLES["preferred_payment_method"],
        )

        class Config:
            extra = "forbid"

    class GasStation(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Gas Station', or put here a preference that does not fit into the other categories.",
        )
        preferred_gas_station: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Gas Stations'.",
            examples=EXAMPLES["preferred_gas_station"],
        )
        willingness_to_pay_extra_for_green_fuel: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Willingness to Pay Extra for Green Fuel'.",
            examples=EXAMPLES["willingness_to_pay_extra_for_green_fuel"],
        )
        price_sensitivity_for_fuel: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Price Sensitivity for Fuel'.",
            examples=EXAMPLES["price_sensitivity_for_fuel"],
        )

        class Config:
            extra = "forbid"

    class ChargingStation(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Charging Station', or put here a preference that does not fit into the other categories.",
        )
        preferred_charging_network: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Charging Network'.",
            examples=EXAMPLES["preferred_charging_network"],
        )
        preferred_type_of_charging_while_traveling: Optional[List[OutputFormat]] = (
            Field(
                default=[],
                description="The user's preference in the topic 'Preferred type of Charging while traveling'.",
                examples=EXAMPLES["preferred_type_of_charging_while_traveling"],
            )
        )
        preferred_type_of_charging_at_everyday_points: Optional[List[OutputFormat]] = (
            Field(
                default=[],
                description="The user's preference in the topic 'Preferred type of Charging when being at everyday points (f.e. work, grocery, restaurant)'.",
                examples=EXAMPLES["preferred_type_of_charging_at_everyday_points"],
            )
        )
        charging_station_onsite_amenities: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Charging Station Amenities: On-site amenities (Restaurant/cafes)'.",
            examples=EXAMPLES["charging_station_onsite_amenities"],
        )

        class Config:
            extra = "forbid"

    class GroceryShopping(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Grocery Shopping', or put here a preference that does not fit into the other categories.",
        )
        preferred_supermarket_chain: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Supermarket Chains'.",
            examples=EXAMPLES["preferred_supermarket_chain"],
        )
        preference_for_local_markets_farms_or_supermarket: Optional[
            List[OutputFormat]
        ] = Field(
            default=[],
            description="The user's preference in the topic 'Preference for Local Markets/Farms or Supermarket'.",
            examples=EXAMPLES["preference_for_local_markets_farms_or_supermarket"],
        )

        class Config:
            extra = "forbid"

    class PointsOfInterest(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Points of Interest', or put here a preference that does not fit into the other categories.",
        )
        restaurant: Optional[Restaurant] = Field(
            default=None,
            description="The user's preferences in the category 'Restaurant'. This includes preferences in the topics 'favourite_cuisine', 'preferred_restaurant_type', 'fast_food_preference', 'desired_price_range', 'dietary_preference', 'preferred_payment_method'.",
        )
        gas_station: Optional[GasStation] = Field(
            default=None,
            description="The user's preferences in the category 'Gas Station'. This includes preferences in the topics 'preferred_gas_station', 'willingness_to_pay_extra_for_green_fuel', 'price_sensitivity_for_fuel'.",
        )
        charging_station: Optional[ChargingStation] = Field(
            default=None,
            description="The user's preferences in the category 'Charging Station (in public)'. This includes preferences in the topics 'preferred_charging_network', 'preferred_type_of_charging_while_traveling', 'preferred_type_of_charging_at_everyday_points', 'charging_station_onsite_amenities'.",
        )
        grocery_shopping: Optional[GroceryShopping] = Field(
            default=None,
            description="The user's preferences in the category 'Grocery Shopping'. This includes preferences in the topics 'preferred_supermarket_chain', 'preference_for_local_markets_farms_or_supermarket'.",
        )

        class Config:
            extra = "forbid"

    # Navigation and Routing

    class Routing(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Routing', or put here a preference that does not fit into the other categories.",
        )
        avoidance_of_specific_road_types: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Avoidance of Specific Road Types'.",
            examples=EXAMPLES["avoidance_of_specific_road_types"],
        )
        priority_for_shortest_time_or_shortest_distance: Optional[
            List[OutputFormat]
        ] = Field(
            default=[],
            description="The user's preference in the topic 'Priority for Shortest Time or Shortest Distance'.",
            examples=EXAMPLES["priority_for_shortest_time_or_shortest_distance"],
        )
        tolerance_for_traffic: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Tolerance for Traffic'. Only tolerance, not if he would take a longer route.",
            examples=EXAMPLES["tolerance_for_traffic"],
        )

        class Config:
            extra = "forbid"

    class TrafficAndConditions(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Traffic And Conditions', or put here a preference that does not fit into the other categories.",
        )
        traffic_information_source_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Traffic Information Source Preferences', so where the user get's traffic information from.",
            examples=EXAMPLES["traffic_information_source_preferences"],
        )
        willingness_to_take_longer_route_to_avoid_traffic: Optional[
            List[OutputFormat]
        ] = Field(
            default=[],
            description="The user's preference in the topic 'Willingness to Take Longer Route to Avoid Traffic'. Only fill here, if traffic is explicitely mentioned, not general tolerance.",
            examples=EXAMPLES["willingness_to_take_longer_route_to_avoid_traffic"],
        )

        class Config:
            extra = "forbid"

    class Parking(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Parking', or put here a preference that does not fit into the other categories.",
        )
        preferred_parking_type: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Parking Type'.",
            examples=EXAMPLES["preferred_parking_type"],
        )
        price_sensitivity_for_paid_parking: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Price Sensitivity for Paid Parking'.",
            examples=EXAMPLES["price_sensitivity_for_paid_parking"],
        )
        distance_willing_to_walk_from_parking_to_destination: Optional[
            List[OutputFormat]
        ] = Field(
            default=[],
            description="The user's preference in the topic 'Distance Willing to Walk from Parking to Destination'.",
            examples=EXAMPLES["distance_willing_to_walk_from_parking_to_destination"],
        )
        preference_for_covered_parking: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preference for Covered Parking'.",
            examples=EXAMPLES["preference_for_covered_parking"],
        )
        need_for_handicapped_accessible_parking: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Need for Handicapped Accessible Parking'.",
            examples=EXAMPLES["need_for_handicapped_accessible_parking"],
        )
        preference_for_parking_with_security: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preference for Parking with Security'.",
            examples=EXAMPLES["preference_for_parking_with_security"],
        )

        class Config:
            extra = "forbid"

    class NavigationAndRouting(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Navigation and Routing', or put here a preference that does not fit into the other categories.",
        )
        routing: Optional[Routing] = Field(
            default=None,
            description="The user's preferences in the category 'Routing'. This includes preferences in the topics 'avoidance_of_specific_road_types', 'priority_for_shortest_time_or_shortest_distance', 'tolerance_for_traffic'.",
        )
        traffic_and_conditions: Optional[TrafficAndConditions] = Field(
            default=None,
            description="The user's preferences in the category 'Traffic and Conditions'. This includes preferences in the topics 'traffic_information_source_preferences', 'willingness_to_take_longer_route_to_avoid_traffic'.",
        )
        parking: Optional[Parking] = Field(
            default=None,
            description="The user's preferences in the category 'Parking'. This includes preferences in the topics 'preferred_parking_type', 'price_sensitivity_for_paid_parking', 'distance_willing_to_walk_from_parking_to_destination', 'preference_for_covered_parking', 'need_for_handicapped_accessible_parking', 'preference_for_parking_with_security'.",
        )

        class Config:
            extra = "forbid"

    # Vehicle Settings and Comfort

    class ClimateControl(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'ClimateControl', or put here a preference that does not fit into the other categories.",
        )
        preferred_temperature: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Temperature'.",
            examples=EXAMPLES["preferred_temperature"],
        )
        fan_speed_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Fan Speed Preferences'.",
            examples=EXAMPLES["fan_speed_preferences"],
        )
        airflow_direction_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Airflow Direction Preferences'.",
            examples=EXAMPLES["airflow_direction_preferences"],
        )
        seat_heating_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Seat Heating Preferences'.",
            examples=EXAMPLES["seat_heating_preferences"],
        )

        class Config:
            extra = "forbid"

    class LightingAndAmbience(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'LightingAndAmbience', or put here a preference that does not fit into the other categories.",
        )
        interior_lighting_brightness_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="Interior Lighting Brightness Preferences'.",
            examples=EXAMPLES["interior_lighting_brightness_preferences"],
        )
        interior_lighting_ambient_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Interior Lighting Ambient Preferences'.",
            examples=EXAMPLES["interior_lighting_ambient_preferences"],
        )
        interior_lighting_color_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Interior Lightning Color Preferences'.",
            examples=EXAMPLES["interior_lighting_color_preferences"],
        )

        class Config:
            extra = "forbid"

    class VehicleSettingsAndComfort(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Vehicle Settings and Comfort', or put here a preference that does not fit into the other categories.",
        )
        climate_control: Optional[ClimateControl] = Field(
            default=None,
            description="The user's preferences in the category 'Climate Control'. This includes preferences in the topics 'preferred_temperature', 'fan_speed_preferences', 'airflow_direction_preferences', 'seat_heating_preferences'.",
        )
        lighting_and_ambience: Optional[LightingAndAmbience] = Field(
            default=None,
            description="The user's preferences in the category 'Lighting and Ambience'. This includes preferences in the topics 'interior_lighting_brightness_preferences', 'interior_lighting_ambient_preferences', 'interior_lighting_color_preferences'.",
        )

        class Config:
            extra = "forbid"

    # Entereinment and Media

    class Music(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Music', or put here a preference that does not fit into the other categories.",
        )
        favorite_genres: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favorite Genres'.",
            examples=EXAMPLES["favorite_genres"],
        )
        favorite_artists_or_bands: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favorite Artists or Bands'.",
            examples=EXAMPLES["favorite_artists_or_bands"],
        )
        favorite_songs: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favorite Songs'.",
            examples=EXAMPLES["favorite_songs"],
        )
        preferred_music_streaming_service: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Music Streaming Service'.",
            examples=EXAMPLES["preferred_music_streaming_service"],
        )

        class Config:
            extra = "forbid"

    class RadioAndPodcast(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Radio And Podcast', or put here a preference that does not fit into the other categories.",
        )
        preferred_radio_station: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Radio Station'.",
            examples=EXAMPLES["preferred_radio_station"],
        )
        favorite_podcast_genres: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favorite Podcast Genres'.",
            examples=EXAMPLES["favorite_podcast_genres"],
        )
        favorite_podcast_shows: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favorite Podcast Shows'.",
            examples=EXAMPLES["favorite_podcast_shows"],
        )
        general_news_source: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'General News Source'.",
            examples=EXAMPLES["general_news_source"],
        )

        class Config:
            extra = "forbid"

    class EntertainmentAndMedia(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Entertainment and Media', or put here a preference that does not fit into the other categories.",
        )
        music: Optional[Music] = Field(
            default=None,
            description="The user's preferences in the category 'Music'. This includes preferences in the topics 'favorite_genres', 'favorite_artists_or_bands', 'favorite_songs', 'preferred_music_streaming_service'.",
        )
        radio_and_podcast: Optional[RadioAndPodcast] = Field(
            default=None,
            description="The user's preferences in the category 'Radio and Podcast'. This includes preferences in the topics 'preferred_radio_station', 'favorite_podcast_genres', 'favorite_podcast_shows', 'general_news_source'.",
        )

        class Config:
            extra = "forbid"

    class PreferencesFunctionOutput(BaseModel):
        points_of_interest: Optional[PointsOfInterest] = Field(
            default=None,
            title="Preferences Points of Interest",
            description="The user's preferences in the category 'Points of Interest'. This includes preferences in the topics 'restaurant', 'gas_station', 'charging_station', 'grocery_shopping'.",
        )
        navigation_and_routing: Optional[NavigationAndRouting] = Field(
            default=None,
            title="Preferences Navigation and Routing",
            description="The user's preferences in the category 'Navigation and Routing'. This includes preferences in the topics 'routing', 'traffic_and_conditions', 'parking'.",
        )
        vehicle_settings_and_comfort: Optional[VehicleSettingsAndComfort] = Field(
            default=None,
            title="Preferences Vehicle Settings and Comfort",
            description="The user's preferences in the category 'Vehicle Settings and Comfort'. This includes preferences in the topics 'climate_control', 'lighting_and_ambience'.",
        )
        entertainment_and_media: Optional[EntertainmentAndMedia] = Field(
            default=None,
            title="Preferences Entertainment and Media",
            description="The user's preferences in the category 'Entertainment and Media'. This includes preferences in the topics 'music', 'radio_and_podcast'.",
        )

        class Config:
            extra = "forbid"

    rebuild = Restaurant.model_rebuild(force=True)
    rebuild = GasStation.model_rebuild(force=True)
    rebuild = ChargingStation.model_rebuild(force=True)
    rebuild = GroceryShopping.model_rebuild(force=True)
    rebuild = Routing.model_rebuild(force=True)
    rebuild = TrafficAndConditions.model_rebuild(force=True)
    rebuild = Parking.model_rebuild(force=True)
    rebuild = ClimateControl.model_rebuild(force=True)
    rebuild = LightingAndAmbience.model_rebuild(force=True)
    rebuild = Music.model_rebuild(force=True)
    rebuild = RadioAndPodcast.model_rebuild(force=True)
    rebuild = EntertainmentAndMedia.model_rebuild(force=True)
    rebuild = NavigationAndRouting.model_rebuild(force=True)
    rebuild = VehicleSettingsAndComfort.model_rebuild(force=True)
    rebuild = PointsOfInterest.model_rebuild(force=True)
    # rebuild = UserPreferences.model_rebuild(force=True)
    rebuild = PreferencesFunctionOutput.model_rebuild(force=True)
    return PreferencesFunctionOutput


def return_pydantic_schema_wo_category(
    maincategory_variable,
    subcategory_variable,
    subcategory_class,
    detail_category_variable,
):

    class OutputFormat(BaseModel):
        user_sentence_preference_revealed: Optional[str] = Field(
            default=None,
            description="user sentence (exclude username) where the user revealed the preference, must be from user sentences, must include the 'user_preference'",
        )
        user_preference: Optional[str] = Field(
            default=None,
            description="The preference of the user, must be included in the 'user_sentence_preference_revealed'",
        )

        class Config:
            extra = "forbid"

    # Points of Interest

    class Restaurant(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Restaurant', or put here a preference that does not fit into the other categories.",
        )
        favourite_cuisine: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favourite Cuisine'.",
            examples=EXAMPLES["favourite_cuisine"],
        )
        preferred_restaurant_type: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Restaurant Type'.",
            examples=EXAMPLES["preferred_restaurant_type"],
        )
        fast_food_preference: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Fast Food Preference'.",
            examples=EXAMPLES["fast_food_preference"],
        )
        desired_price_range: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Desired Price Range'.",
            examples=EXAMPLES["desired_price_range"],
        )
        dietary_preference: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Dietary Preferences'.",
            examples=EXAMPLES["dietary_preference"],
        )
        preferred_payment_method: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Payment method'.",
            examples=EXAMPLES["preferred_payment_method"],
        )

        class Config:
            extra = "forbid"

    class GasStation(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Gas Station', or put here a preference that does not fit into the other categories.",
        )
        preferred_gas_station: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Gas Stations'.",
            examples=EXAMPLES["preferred_gas_station"],
        )
        willingness_to_pay_extra_for_green_fuel: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Willingness to Pay Extra for Green Fuel'.",
            examples=EXAMPLES["willingness_to_pay_extra_for_green_fuel"],
        )
        price_sensitivity_for_fuel: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Price Sensitivity for Fuel'.",
            examples=EXAMPLES["price_sensitivity_for_fuel"],
        )

        class Config:
            extra = "forbid"

    class ChargingStation(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Charging Station', or put here a preference that does not fit into the other categories.",
        )
        preferred_charging_network: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Charging Network'.",
            examples=EXAMPLES["preferred_charging_network"],
        )
        preferred_type_of_charging_while_traveling: Optional[List[OutputFormat]] = (
            Field(
                default=[],
                description="The user's preference in the topic 'Preferred type of Charging while traveling'.",
                examples=EXAMPLES["preferred_type_of_charging_while_traveling"],
            )
        )
        preferred_type_of_charging_at_everyday_points: Optional[List[OutputFormat]] = (
            Field(
                default=[],
                description="The user's preference in the topic 'Preferred type of Charging when being at everyday points (f.e. work, grocery, restaurant)'.",
                examples=EXAMPLES["preferred_type_of_charging_at_everyday_points"],
            )
        )
        charging_station_onsite_amenities: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Charging Station Amenities: On-site amenities (Restaurant/cafes)'.",
            examples=EXAMPLES["charging_station_onsite_amenities"],
        )

        class Config:
            extra = "forbid"

    class GroceryShopping(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Grocery Shopping', or put here a preference that does not fit into the other categories.",
        )
        preferred_supermarket_chain: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Supermarket Chains'.",
            examples=EXAMPLES["preferred_supermarket_chain"],
        )
        preference_for_local_markets_farms_or_supermarket: Optional[
            List[OutputFormat]
        ] = Field(
            default=[],
            description="The user's preference in the topic 'Preference for Local Markets/Farms or Supermarket'.",
            examples=EXAMPLES["preference_for_local_markets_farms_or_supermarket"],
        )

        class Config:
            extra = "forbid"

    class PointsOfInterest(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Points of Interest', or put here a preference that does not fit into the other categories.",
        )
        restaurant: Optional[Restaurant] = Field(
            default=None,
            description="The user's preferences in the category 'Restaurant'. This includes preferences in the topics 'favourite_cuisine', 'preferred_restaurant_type', 'fast_food_preference', 'desired_price_range', 'dietary_preference', 'preferred_payment_method'.",
        )
        gas_station: Optional[GasStation] = Field(
            default=None,
            description="The user's preferences in the category 'Gas Station'. This includes preferences in the topics 'preferred_gas_station', 'willingness_to_pay_extra_for_green_fuel', 'price_sensitivity_for_fuel'.",
        )
        charging_station: Optional[ChargingStation] = Field(
            default=None,
            description="The user's preferences in the category 'Charging Station (in public)'. This includes preferences in the topics 'preferred_charging_network', 'preferred_type_of_charging_while_traveling', 'preferred_type_of_charging_at_everyday_points', 'charging_station_onsite_amenities'.",
        )
        grocery_shopping: Optional[GroceryShopping] = Field(
            default=None,
            description="The user's preferences in the category 'Grocery Shopping'. This includes preferences in the topics 'preferred_supermarket_chain', 'preference_for_local_markets_farms_or_supermarket'.",
        )

        class Config:
            extra = "forbid"

    # Navigation and Routing

    class Routing(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Routing', or put here a preference that does not fit into the other categories.",
        )
        avoidance_of_specific_road_types: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Avoidance of Specific Road Types'.",
            examples=EXAMPLES["avoidance_of_specific_road_types"],
        )
        priority_for_shortest_time_or_shortest_distance: Optional[
            List[OutputFormat]
        ] = Field(
            default=[],
            description="The user's preference in the topic 'Priority for Shortest Time or Shortest Distance'.",
            examples=EXAMPLES["priority_for_shortest_time_or_shortest_distance"],
        )
        tolerance_for_traffic: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Tolerance for Traffic'. Only tolerance, not if he would take a longer route.",
            examples=EXAMPLES["tolerance_for_traffic"],
        )

        class Config:
            extra = "forbid"

    class TrafficAndConditions(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Traffic And Conditions', or put here a preference that does not fit into the other categories.",
        )
        traffic_information_source_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Traffic Information Source Preferences', so where the user get's traffic information from.",
            examples=EXAMPLES["traffic_information_source_preferences"],
        )
        willingness_to_take_longer_route_to_avoid_traffic: Optional[
            List[OutputFormat]
        ] = Field(
            default=[],
            description="The user's preference in the topic 'Willingness to Take Longer Route to Avoid Traffic'. Only fill here, if traffic is explicitely mentioned, not general tolerance.",
            examples=EXAMPLES["willingness_to_take_longer_route_to_avoid_traffic"],
        )

        class Config:
            extra = "forbid"

    class Parking(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Parking', or put here a preference that does not fit into the other categories.",
        )
        preferred_parking_type: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Parking Type'.",
            examples=EXAMPLES["preferred_parking_type"],
        )
        price_sensitivity_for_paid_parking: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Price Sensitivity for Paid Parking'.",
            examples=EXAMPLES["price_sensitivity_for_paid_parking"],
        )
        distance_willing_to_walk_from_parking_to_destination: Optional[
            List[OutputFormat]
        ] = Field(
            default=[],
            description="The user's preference in the topic 'Distance Willing to Walk from Parking to Destination'.",
            examples=EXAMPLES["distance_willing_to_walk_from_parking_to_destination"],
        )
        preference_for_covered_parking: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preference for Covered Parking'.",
            examples=EXAMPLES["preference_for_covered_parking"],
        )
        need_for_handicapped_accessible_parking: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Need for Handicapped Accessible Parking'.",
            examples=EXAMPLES["need_for_handicapped_accessible_parking"],
        )
        preference_for_parking_with_security: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preference for Parking with Security'.",
            examples=EXAMPLES["preference_for_parking_with_security"],
        )

        class Config:
            extra = "forbid"

    class NavigationAndRouting(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Navigation and Routing', or put here a preference that does not fit into the other categories.",
        )
        routing: Optional[Routing] = Field(
            default=None,
            description="The user's preferences in the category 'Routing'. This includes preferences in the topics 'avoidance_of_specific_road_types', 'priority_for_shortest_time_or_shortest_distance', 'tolerance_for_traffic'.",
        )
        traffic_and_conditions: Optional[TrafficAndConditions] = Field(
            default=None,
            description="The user's preferences in the category 'Traffic and Conditions'. This includes preferences in the topics 'traffic_information_source_preferences', 'willingness_to_take_longer_route_to_avoid_traffic'.",
        )
        parking: Optional[Parking] = Field(
            default=None,
            description="The user's preferences in the category 'Parking'. This includes preferences in the topics 'preferred_parking_type', 'price_sensitivity_for_paid_parking', 'distance_willing_to_walk_from_parking_to_destination', 'preference_for_covered_parking', 'need_for_handicapped_accessible_parking', 'preference_for_parking_with_security'.",
        )

        class Config:
            extra = "forbid"

    # Vehicle Settings and Comfort

    class ClimateControl(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'ClimateControl', or put here a preference that does not fit into the other categories.",
        )
        preferred_temperature: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Temperature'.",
            examples=EXAMPLES["preferred_temperature"],
        )
        fan_speed_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Fan Speed Preferences'.",
            examples=EXAMPLES["fan_speed_preferences"],
        )
        airflow_direction_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Airflow Direction Preferences'.",
            examples=EXAMPLES["airflow_direction_preferences"],
        )
        seat_heating_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Seat Heating Preferences'.",
            examples=EXAMPLES["seat_heating_preferences"],
        )

        class Config:
            extra = "forbid"

    class LightingAndAmbience(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'LightingAndAmbience', or put here a preference that does not fit into the other categories.",
        )
        interior_lighting_brightness_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="Interior Lighting Brightness Preferences'.",
            examples=EXAMPLES["interior_lighting_brightness_preferences"],
        )
        interior_lighting_ambient_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Interior Lighting Ambient Preferences'.",
            examples=EXAMPLES["interior_lighting_ambient_preferences"],
        )
        interior_lighting_color_preferences: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Interior Lightning Color Preferences'.",
            examples=EXAMPLES["interior_lighting_color_preferences"],
        )

        class Config:
            extra = "forbid"

    class VehicleSettingsAndComfort(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Vehicle Settings and Comfort', or put here a preference that does not fit into the other categories.",
        )
        climate_control: Optional[ClimateControl] = Field(
            default=None,
            description="The user's preferences in the category 'Climate Control'. This includes preferences in the topics 'preferred_temperature', 'fan_speed_preferences', 'airflow_direction_preferences', 'seat_heating_preferences'.",
        )
        lighting_and_ambience: Optional[LightingAndAmbience] = Field(
            default=None,
            description="The user's preferences in the category 'Lighting and Ambience'. This includes preferences in the topics 'interior_lighting_brightness_preferences', 'interior_lighting_ambient_preferences', 'interior_lighting_color_preferences'.",
        )

        class Config:
            extra = "forbid"

    # Entereinment and Media

    class Music(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Music', or put here a preference that does not fit into the other categories.",
        )
        favorite_genres: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favorite Genres'.",
            examples=EXAMPLES["favorite_genres"],
        )
        favorite_artists_or_bands: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favorite Artists or Bands'.",
            examples=EXAMPLES["favorite_artists_or_bands"],
        )
        favorite_songs: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favorite Songs'.",
            examples=EXAMPLES["favorite_songs"],
        )
        preferred_music_streaming_service: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Music Streaming Service'.",
            examples=EXAMPLES["preferred_music_streaming_service"],
        )

        class Config:
            extra = "forbid"

    class RadioAndPodcast(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Radio And Podcast', or put here a preference that does not fit into the other categories.",
        )
        preferred_radio_station: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Preferred Radio Station'.",
            examples=EXAMPLES["preferred_radio_station"],
        )
        favorite_podcast_genres: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favorite Podcast Genres'.",
            examples=EXAMPLES["favorite_podcast_genres"],
        )
        favorite_podcast_shows: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'Favorite Podcast Shows'.",
            examples=EXAMPLES["favorite_podcast_shows"],
        )
        general_news_source: Optional[List[OutputFormat]] = Field(
            default=[],
            description="The user's preference in the topic 'General News Source'.",
            examples=EXAMPLES["general_news_source"],
        )

        class Config:
            extra = "forbid"

    class EntertainmentAndMedia(BaseModel):
        no_or_other_preferences: Optional[str] = Field(
            default=None,
            description="Put here 'No' if there is no preference in the conversation for the category 'Entertainment and Media', or put here a preference that does not fit into the other categories.",
        )
        music: Optional[Music] = Field(
            default=None,
            description="The user's preferences in the category 'Music'. This includes preferences in the topics 'favorite_genres', 'favorite_artists_or_bands', 'favorite_songs', 'preferred_music_streaming_service'.",
        )
        radio_and_podcast: Optional[RadioAndPodcast] = Field(
            default=None,
            description="The user's preferences in the category 'Radio and Podcast'. This includes preferences in the topics 'preferred_radio_station', 'favorite_podcast_genres', 'favorite_podcast_shows', 'general_news_source'.",
        )

        class Config:
            extra = "forbid"

    class PreferencesFunctionOutput(BaseModel):
        points_of_interest: Optional[PointsOfInterest] = Field(
            default=None,
            title="Preferences Points of Interest",
            description="The user's preferences in the category 'Points of Interest'. This includes preferences in the topics 'restaurant', 'gas_station', 'charging_station', 'grocery_shopping'.",
        )
        navigation_and_routing: Optional[NavigationAndRouting] = Field(
            default=None,
            title="Preferences Navigation and Routing",
            description="The user's preferences in the category 'Navigation and Routing'. This includes preferences in the topics 'routing', 'traffic_and_conditions', 'parking'.",
        )
        vehicle_settings_and_comfort: Optional[VehicleSettingsAndComfort] = Field(
            default=None,
            title="Preferences Vehicle Settings and Comfort",
            description="The user's preferences in the category 'Vehicle Settings and Comfort'. This includes preferences in the topics 'climate_control', 'lighting_and_ambience'.",
        )
        entertainment_and_media: Optional[EntertainmentAndMedia] = Field(
            default=None,
            title="Preferences Entertainment and Media",
            description="The user's preferences in the category 'Entertainment and Media'. This includes preferences in the topics 'music', 'radio_and_podcast'.",
        )

        class Config:
            extra = "forbid"

    if (
        subcategory_class == "Restaurant"
        or subcategory_class == "GasStation"
        or subcategory_class == "ChargingStation"
        or subcategory_class == "GroceryShopping"
    ):
        del PointsOfInterest.__annotations__[subcategory_variable]
        del PointsOfInterest.model_fields[subcategory_variable]
        description = PreferencesFunctionOutput.model_fields[
            maincategory_variable
        ].description
        new_description = description.replace(f"'{subcategory_variable}'", "")
        PreferencesFunctionOutput.model_fields[maincategory_variable].description = (
            new_description
        )
    elif (
        subcategory_class == "Routing"
        or subcategory_class == "TrafficAndConditions"
        or subcategory_class == "Parking"
    ):
        del NavigationAndRouting.__annotations__[subcategory_variable]
        del NavigationAndRouting.model_fields[subcategory_variable]
        description = PreferencesFunctionOutput.model_fields[
            maincategory_variable
        ].description
        new_description = description.replace(f"'{subcategory_variable}'", "")
        PreferencesFunctionOutput.model_fields[maincategory_variable].description = (
            new_description
        )
    elif (
        subcategory_class == "ClimateControl"
        or subcategory_class == "LightingAndAmbience"
    ):
        del VehicleSettingsAndComfort.__annotations__[subcategory_variable]
        del VehicleSettingsAndComfort.model_fields[subcategory_variable]
        description = PreferencesFunctionOutput.model_fields[
            maincategory_variable
        ].description
        new_description = description.replace(f"'{subcategory_variable}'", "")
        PreferencesFunctionOutput.model_fields[maincategory_variable].description = (
            new_description
        )
    elif subcategory_class == "Music" or subcategory_class == "RadioAndPodcast":
        del EntertainmentAndMedia.__annotations__[subcategory_variable]
        del EntertainmentAndMedia.model_fields[subcategory_variable]
        description = PreferencesFunctionOutput.model_fields[
            maincategory_variable
        ].description
        new_description = description.replace(f"'{subcategory_variable}'", "")
        PreferencesFunctionOutput.model_fields[maincategory_variable].description = (
            new_description
        )
    else:
        print("ERROR: subcategory_class does not exist")

    rebuild = Restaurant.model_rebuild(force=True)
    rebuild = GasStation.model_rebuild(force=True)
    rebuild = ChargingStation.model_rebuild(force=True)
    rebuild = GroceryShopping.model_rebuild(force=True)
    rebuild = Routing.model_rebuild(force=True)
    rebuild = TrafficAndConditions.model_rebuild(force=True)
    rebuild = Parking.model_rebuild(force=True)
    rebuild = ClimateControl.model_rebuild(force=True)
    rebuild = LightingAndAmbience.model_rebuild(force=True)
    rebuild = Music.model_rebuild(force=True)
    rebuild = RadioAndPodcast.model_rebuild(force=True)
    rebuild = EntertainmentAndMedia.model_rebuild(force=True)
    rebuild = NavigationAndRouting.model_rebuild(force=True)
    rebuild = VehicleSettingsAndComfort.model_rebuild(force=True)
    rebuild = PointsOfInterest.model_rebuild(force=True)
    # rebuild = UserPreferences.model_rebuild(force=True)
    rebuild = PreferencesFunctionOutput.model_rebuild(force=True)
    return PreferencesFunctionOutput
