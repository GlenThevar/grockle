import os
import requests
import serpapi
from dotenv import load_dotenv
from typing import Optional
from langchain_core.tools import tool

from data.files.agents.weather_agent import run_weather_agent

load_dotenv("../env/.env")


@tool
def tourism_specialist(query: str) -> str:
    """
    Ask the tourism specialist.
    """

    from agents.tourism_agent import run_tourism_agent

    return run_tourism_agent(query)


@tool
def search_tripadvisor_attractions(
    query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category_type: str = "ATTRACTIONOVERVIEW:-true",
    sort: str = "POPULARITY",
    filters: Optional[str] = None,
    page: int = 1,
) -> str:
    """
    Search attractions, tours, and activities on TripAdvisor.

    Args:
        query: Destination name (e.g. 'new york city') or geoId/Entity ID (e.g. '60763').
        start_date: Optional start date in YYYY-MM-DD format.
        end_date: Optional end date in YYYY-MM-DD format.
        category_type: Filter by type (e.g. 'ATTRACTIONOVERVIEW:-true' for Attractions,
                       'ATTRACTIONOVERVIEW:42-false' for Tours, 'ATTRACTIONOVERVIEW:36-false' for Food & Drink).
        sort: Ranking order ('POPULARITY', 'BEST_VALUE', 'PRICE_LOW_TO_HIGH', 'DISTANCE_FROM_CITY_CENTER').
        filters: Custom filter string format: name(value1;value2).
                 Example: 'neighborhood(15565677;7102352),anyTag(11295)'.
        page: Page number for pagination.

    Returns:
        Formatted summary string of matching attractions.
    """
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        return "Error: RAPIDAPI_KEY environment variable is not set."

    url = "https://tripadvisor-data.p.rapidapi.com/attraction/search"

    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "tripadvisor-data.p.rapidapi.com",
        "Content-Type": "application/json",
    }

    params = {
        "query": query,
        "categoryType": category_type,
        "sort": sort,
        "page": page,
    }

    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    if filters:
        params["filters"] = filters

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        res_data = response.json()

        attractions = res_data.get("data", {}).get("attractions", [])
        if not attractions:
            return f"No attractions found for query '{query}'."

        output = []
        for item in attractions[:5]:  # Top 5
            title = item.get("cardTitle", {}).get("string", "N/A")
            category = item.get("primaryInfo", {}).get("text", "N/A")
            location_info = item.get("secondaryInfo", {}).get("text", "N/A")
            rating = item.get("bubbleRating", {}).get("rating", "N/A")
            reviews = (
                item.get("bubbleRating", {})
                .get("numberReviews", {})
                .get("string", "(0)")
            )

            merch_text = item.get("merchandisingText")
            pricing = merch_text.get("htmlString") if merch_text else "Varies / Free"

            output.append(
                f"• {title}\n"
                f"  Category: {category}\n"
                f"  Location: {location_info}\n"
                f"  Rating: {rating}/5 {reviews}\n"
                f"  Pricing: {pricing}"
            )

        return f"Results for '{query}':\n\n" + "\n\n".join(output)

    except requests.exceptions.RequestException as e:
        return f"TripAdvisor API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"


@tool
def search_tripadvisor_things_to_do(
    location: str,
    limit: int = 30,
    offset: int = 0,
    category: str = "any",
    free_cancellation: bool = False,
    skip_the_line: bool = False,
    private_tour: bool = False,
    traveler_rating: str = "any",
    duration: str = "any",
    currency: str = "USD",
) -> str:
    """
    Search for attractions and activities using TripAdvisor.

    Args:
        location: Target city or destination (e.g., 'Bangkok').
        limit: Max results to fetch (1-100, default 30).
        offset: Zero-based result index offset.
        category: Category filter (default 'any').
        free_cancellation: Only show activities offering free cancellation.
        skip_the_line: Only show skip-the-line activities.
        private_tour: Only show private tours.
        traveler_rating: Minimum traveler rating filter.
        duration: Activity duration filter.
        currency: 3-letter currency code (e.g., 'USD').

    Returns:
        Formatted summary string of top activities.
    """
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        return "Error: RAPIDAPI_KEY environment variable is not set."

    url = "https://tripadvisor34.p.rapidapi.com/api/v1/things-to-do/search"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "tripadvisor34.p.rapidapi.com",
    }

    params = {
        "location": location,
        "offset": offset,
        "limit": min(limit, 100),
        "free_cancellation": str(free_cancellation).lower(),
        "category": category,
        "skip_the_line": str(skip_the_line).lower(),
        "currency": currency,
        "private_tour": str(private_tour).lower(),
        "traveler_rating": traveler_rating,
        "duration": duration,
    }

    try:
        res = requests.get(url, headers=headers, params=params, timeout=30)
        res.raise_for_status()
        payload = res.json()

        items = payload.get("data", {}).get("thingsToDo", [])
        if not items:
            return f"No activities found for '{location}'."

        results = []
        for item in items[:5]:
            name = item.get("name", "N/A")
            rating = item.get("rating", "N/A")
            reviews = item.get("reviews", 0)
            url_link = item.get("url", "")

            results.append(
                f"• {name}\n"
                f"  Rating: {rating}/5 ({reviews:,} reviews)\n"
                f"  Link: {url_link}"
            )

        return f"Top Activities in {location}:\n\n" + "\n\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"TripAdvisor API error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def search_tripadvisor_cruises(
    destination: str,
    departure_month: Optional[str] = None,
    length: str = "any",
    cabin_type: str = "any",
    limit: int = 20,
    offset: int = 0,
    currency: str = "USD",
    locale: str = "en_US",
) -> str:
    """
    Search for cruise itineraries and ship details using TripAdvisor.

    Args:
        destination: Target cruise area (e.g., 'Caribbean', 'Bahamas').
        departure_month: Optional departure month in YYYY-MM format (e.g., '2027-01').
        length: Cruise duration filter ('any', or specific length enum).
        cabin_type: Cabin type filter ('any', 'inside', 'balcony', etc.).
        limit: Max results to return (1-100, default 20).
        offset: Zero-based result index offset.
        currency: 3-letter currency code (default 'USD').
        locale: Language/locale setting (default 'en_US').

    Returns:
        Formatted text summary of available cruises.
    """
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        return "Error: RAPIDAPI_KEY environment variable is not set."

    url = "https://tripadvisor34.p.rapidapi.com/api/v1/cruises/search"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "tripadvisor34.p.rapidapi.com",
    }

    params = {
        "destination": destination,
        "offset": offset,
        "limit": min(limit, 100),
        "locale": locale,
        "length": length,
        "cabinType": cabin_type,
        "currency": currency,
    }

    if departure_month:
        params["departureMonth"] = departure_month

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        cruises = payload.get("data", {}).get("cruises", [])
        if not cruises:
            return f"No cruises found for destination '{destination}'."

        results = []
        for c in cruises[:5]:
            name = c.get("name", "N/A")
            ship = c.get("shipName", "N/A")
            line = c.get("cruiseLineName", "N/A")
            nights = c.get("lengthNights", "N/A")
            rating = c.get("score", "N/A")
            reviews = c.get("reviewCount", 0)

            sailings = c.get("sailingsItems", [])
            next_date = sailings[0].get("departureDate") if sailings else "N/A"

            results.append(
                f"• {name}\n"
                f"  Line/Ship: {line} — {ship}\n"
                f"  Duration: {nights} nights | Rating: {rating}/5 ({reviews} reviews)\n"
                f"  Next Sailing: {next_date}"
            )

        return f"Top Cruises for '{destination}':\n\n" + "\n\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"


@tool
def search_tourist_attractions_google_maps(
    query: str,
    gps_coordinates: Optional[str] = None,
    min_rating: Optional[float] = None,
    open_now: bool = False,
) -> str:
    """
    Search for tourist attractions, landmarks, and places to visit using Google Maps.

    Args:
        query: Specific search term or category (e.g., 'tourist attractions in Jersey City', 'museums').
        gps_coordinates: Optional origin GPS location format '@latitude,longitude,zoom' (e.g., '@40.7455,-74.0083,14z').
        min_rating: Preferred minimum rating filter (e.g., 4.0, 4.5).
        open_now: If True, filters locations currently open.

    Returns:
        Formatted summary list of tourist attractions and details.
    """
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        return "Error: SERPAPI_KEY environment variable is missing."

    client = serpapi.Client(api_key=api_key)

    params = {
        "engine": "google_maps",
        "type": "search",
        "q": query,
        "hl": "en",
        "gl": "us",
    }

    if gps_coordinates:
        params["ll"] = gps_coordinates

    if min_rating:
        params["min_rating"] = min_rating  # type: ignore

    if open_now:
        params["open_state"] = "now"

    try:
        results = client.search(params)
        places = results.get("local_results", [])

        if not places:
            return f"No attractions found for query: '{query}'."

        formatted_places = []
        for place in places[:7]:  # Top 7 locations
            title = place.get("title", "Unknown Location")
            rating = place.get("rating", "No rating")
            reviews = place.get("reviews", 0)
            category = place.get("type", "Attraction")
            address = place.get("address", "Address unavailable")
            open_status = place.get("open_state", "Hours unavailable")

            user_review = place.get("user_review", None)
            review_snippet = f"\n  Review snippet: {user_review}" if user_review else ""

            formatted_places.append(
                f"• {title} ({category})\n"
                f"  Rating: {rating}/5⭐ ({reviews} reviews)\n"
                f"  Address: {address}\n"
                f"  Status: {open_status}"
                f"{review_snippet}"
            )

        return f"Top Attractions for '{query}':\n\n" + "\n\n".join(formatted_places)

    except Exception as e:
        return f"SerpAPI Error: {str(e)}"
