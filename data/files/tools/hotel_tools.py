import os
import requests
import serpapi
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv("../env/.env")


@tool
def hotel_specialist(query: str) -> str:
    """
    Ask the hotel specialist.
    """

    from agents.hotel_agent import run_hotel_agent

    return run_hotel_agent(query)


@tool
def search_hotels_google_maps(
    query: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    children: int = 0,
    children_ages: Optional[List[int]] = None,
    sort_by: Optional[int] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    hotel_class: Optional[List[int]] = None,
    amenities: Optional[List[int]] = None,
    currency: str = "USD",
    gl: str = "us",
    hl: str = "en",
) -> List[Dict[str, Any]]:
    """
    Searches Google Hotels via SerpApi and returns a list of matching properties.

    Args:
        query: Destination or hotel search query (e.g., "Bali Resorts").
        check_in_date: Check-in date in YYYY-MM-DD format.
        check_out_date: Check-out date in YYYY-MM-DD format.
        adults: Number of adult guests. Defaults to 2.
        children: Number of child guests. Defaults to 0.
        children_ages: List of child ages (e.g., [5, 8]). Must match 'children'
          count.
        sort_by: 3 (Lowest price), 8 (Highest rating), or 13 (Most reviewed).
        min_price: Minimum price filter.
        max_price: Maximum price filter.
        hotel_class: Filter by star ratings (e.g., [4, 5] for 4 and 5-star
          hotels).
        amenities: Filter by amenity IDs (e.g., [35, 9]).
        currency: 3-letter currency code (e.g., "USD", "EUR").
        gl: 2-letter country code for localization.
        hl: 2-letter language code for localization.

    Returns:
        List of property dictionary results.
    """

    api_key = os.getenv("SERP_API_KEY") or ""
    if not api_key:
        raise ValueError("SerpApi key is not set")

    client = serpapi.Client(api_key=api_key)

    params: Dict[str, Any] = {
        "engine": "google_hotels",
        "q": query,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "children": children,
        "currency": currency,
        "gl": gl,
        "hl": hl,
    }

    if children > 0 and children_ages:
        params["children_ages"] = ",".join(map(str, children_ages))

    if sort_by is not None:
        params["sort_by"] = sort_by

    if min_price is not None:
        params["min_price"] = min_price

    if max_price is not None:
        params["max_price"] = max_price

    if hotel_class:
        params["hotel_class"] = ",".join(map(str, hotel_class))

    if amenities:
        params["amenities"] = ",".join(map(str, amenities))

    try:
        results = client.search(params)
        return results.get("properties", [])
    except Exception as err:
        print(f"Error fetching hotel data: {err}")
        return []


@tool
def search_hotels_priceline(
    location_name: str,
    check_in_date: str,
    check_out_date: str,
    rooms: int = 1,
    adults: int = 1,
    children_ages: Optional[List[int]] = None,
    sort_by: Optional[str] = "HDR",
    hotels_type: Optional[str] = "ALL_HOTELS",
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    guest_score: Optional[int] = None,
    star_level: Optional[int] = None,
    neighborhoods: Optional[List[str]] = None,
    amenities: Optional[List[str]] = None,
    property_type: Optional[List[str]] = None,
    hotel_name: Optional[str] = None,
    limit: int = 30,
    page: int = 1,
) -> List[Dict[str, Any]]:
    """
    Searches hotels via the Priceline RapidAPI endpoint and returns results.

    Args:
        location_name: Name of the location to search for (e.g., "New York, NY").
        check_in_date: Check-in date in YYYY-MM-DD format.
        check_out_date: Check-out date in YYYY-MM-DD format.
        rooms: Number of rooms. Defaults to 1.
        adults: Number of adult guests. Defaults to 1.
        children_ages: List of child ages 0-17 (e.g., [0, 12, 17]).
        sort_by: Sorting order ("PRICE", "REVIEW_SCORE", "HDR", "DEALS").
          Defaults to "HDR".
        hotels_type: Type of hotels ("ALL_HOTELS", "EXPRESS_DEALS"). Defaults to
          "ALL_HOTELS".
        min_price: Minimum price filter.
        max_price: Maximum price filter.
        guest_score: Minimum guest rating (7 for 7.0+, 8 for 8.0+, 9 for 9.0+).
        star_level: Minimum star rating (3 for 3+, 4 for 4+, 5 for 5+).
        neighborhoods: List of neighborhood IDs (e.g., ["910052856",
          "910051961"]).
        amenities: List of amenity codes (e.g., ["FREE_CANCELLATION",
          "FINTRNT"]).
        property_type: List of property type IDs (e.g., ["201", "203"]).
        hotel_name: Filter results by hotel name string.
        limit: Number of records per API call. Defaults to 30.
        page: Page index for pagination. Defaults to 1.

    Returns:
        JSON response object containing hotel listings and metadata.
    """

    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        raise ValueError("RAPIDAPI_KEY environment variable is not set")

    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "priceline-com2.p.rapidapi.com",
        "Content-Type": "application/json",
    }

    try:
        auto_complete_url = "https://priceline-com2.p.rapidapi.com/hotels/auto-complete"
        ac_response = requests.get(
            auto_complete_url, headers=headers, params={"query": location_name}
        )
        ac_response.raise_for_status()

        search_items = ac_response.json().get("data", {}).get("searchItems", [])
        location_id = None
        for item in search_items:
            if item.get("type") == "CITY" and "cityID" in item:
                location_id = item["cityID"]
                break

        if not location_id and search_items:
            location_id = search_items[0].get("id")

        if not location_id:
            print(f"No location ID found for: {location_name}")
            return []

        search_url = "https://priceline-com2.p.rapidapi.com/hotels/search"
        params: Dict[str, Any] = {
            "locationId": location_id,
            "checkIn": check_in_date,
            "checkOut": check_out_date,
            "rooms": rooms,
            "adults": adults,
            "limit": limit,
            "page": page,
            "sort": sort_by,
            "hotelsType": hotels_type,
        }

        if children_ages:
            params["children"] = ",".join(map(str, children_ages))
        if min_price is not None:
            params["minPrice"] = str(min_price)
        if max_price is not None:
            params["maxPrice"] = str(max_price)
        if guest_score is not None:
            params["guestScore"] = guest_score
        if star_level is not None:
            params["starLevel"] = star_level
        if neighborhoods:
            params["neighborhoods"] = ",".join(neighborhoods)
        if amenities:
            params["amenities"] = ",".join(amenities)
        if property_type:
            params["propertyType"] = ",".join(property_type)
        if hotel_name:
            params["hotelName"] = hotel_name

        search_response = requests.get(search_url, headers=headers, params=params)
        search_response.raise_for_status()

        hotels_raw = search_response.json().get("data", {}).get("hotels", [])
        cleaned_hotels = []

        for hotel in hotels_raw:
            extracted_info = {
                "name": hotel.get("name"),
                "star_rating": hotel.get("starRating"),
                "guest_rating": hotel.get("overallGuestRating"),
                "total_price": hotel.get("ratesSummary", {}).get("grandTotal"),
                "address": hotel.get("location", {})
                .get("address", {})
                .get("addressLine1"),
                "city": hotel.get("location", {}).get("address", {}).get("cityName"),
            }
            cleaned_hotels.append(extracted_info)

        return cleaned_hotels

    except requests.RequestException as err:
        print(f"API Error: {err}")
        return []
