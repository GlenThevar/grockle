import os
import json
import requests
import serpapi
import sqlite3
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from datetime import date

load_dotenv("../env/.env")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "agentData.db")


@tool
def flight_specialist(query: str) -> str:
    """
    Ask the flight specialist.
    """

    from agents.flight_agent import run_flight_agent

    return run_flight_agent(query)


@tool
def get_flight_information_duffel(
    origin: str,
    destination: str,
    departure_date: str,
    passenger_type: str = "adult",
    cabin_class: str = "economy",
) -> str:
    """
    This tool allows you to get flight information using the Duffel API. However it is test data and not real data. So only use this tool when the other tools are not available / working. Use this as only a last resort

    Args:
        origin: Origin IATA code (e.g. BOM)
        destination: Destination IATA code (e.g. DEL)
        departure_date: YYYY-MM-DD
        passenger_type: adult, child, infant_without_seat
        cabin_class: economy, premium_economy, business, first

    Returns:
        Flight information as a formatted string.
    """

    api_key = os.getenv("DUFFEL_API_KEY")

    if not api_key:
        return "DUFFEL_API_KEY not found."

    url = "https://api.duffel.com/air/offer_requests"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Duffel-Version": "v2",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "data": {
            "slices": [
                {
                    "origin": origin,
                    "destination": destination,
                    "departure_date": departure_date,
                }
            ],
            "passengers": [
                {
                    "type": passenger_type,
                }
            ],
            "cabin_class": cabin_class,
        }
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        offers = data.get("data", {}).get("offers", [])

        if not offers:
            return "No flight offers found."

        offers = sorted(offers, key=lambda x: float(x["total_amount"]))

        results = []

        for offer in offers[:5]:

            segment = offer["slices"][0]["segments"][0]

            airline = offer["owner"]["name"]

            price = offer["total_amount"]

            currency = offer["total_currency"]

            departure = segment["departing_at"]

            arrival = segment["arriving_at"]

            duration = segment["duration"]

            results.append(f"""
                Airline: {airline}
                Price: {price} {currency}
                Departure: {departure}
                Arrival: {arrival}
                Duration: {duration}
                """.strip())

        return "\n\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"Duffel API error: {str(e)}"

    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def get_flight_information_nolayovers_serpapi(
    departure_id: str,
    arrival_id: str,
    currency: str = "INR",
    outbound_date: str = date.today().strftime("%Y-%m-%d"),
    adults: int = 1,
    children: int = 0,
    sort_by: int = 2,
) -> str:
    """
    This tool uses SerpAPI to retrieve one-way flight (which do not have layovers) data from Google Flights. Only use this when you want to get direct flights. If the user wants flights with layovers, use the other tool.

    Args:
        departure_id: The departure airport code (e.g., "BOM" for Mumbai)

        arrival_id: The arrival airport code (e.g., "DEL" for Delhi)

        currency: The currency code for the flight prices (default is "INR")

        outbound_date: The date of departure in "YYYY-MM-DD" format (default is today's date)

        adults: The number of adult passengers (default is 1)

        children: The number of child passengers (default is 0)

        sort_by: The sorting criterion for the search (default is 2). 1 stands for Top flights, 2 stands for Price, 3 stands for Departure time, 4 stands for Arrival time, 5 stands for Duration and 6 stands for Emissions

    Returns:
        A string containing the flight information retrieved from Google Flights.
    """

    api_key = os.getenv("SERP_API_KEY")

    if not api_key:
        return "SERPAPI_API_KEY not found."

    client = serpapi.Client(api_key=api_key)

    try:

        response = client.search(
            {
                "engine": "google_flights",
                "departure_id": departure_id,
                "arrival_id": arrival_id,
                "outbound_date": outbound_date,
                "currency": currency,
                "hl": "en",
                "adults": adults,
                "children": children,
                "sort_by": sort_by,
                "type": 2,
                "stops": 1,
            }
        )

        offers = response.get("other_flights", [])

        if not offers:
            return "No flight offers found."

        results = []

        for offer in offers:

            departure = offer["flights"][0]["departure_airport"]["name"]
            departure_time = offer["flights"][0]["departure_airport"]["time"]
            arrival = offer["flights"][0]["arrival_airport"]["name"]
            arrival_time = offer["flights"][0]["arrival_airport"]["time"]
            airline = offer["flights"][0]["airline"]
            price = offer["price"]
            ticket_currency = currency
            duration = offer["total_duration"]

            results.append(f"""
                Airline: {airline}
                Price: {price} {currency}
                Departure: {departure}
                Departure Time: {departure_time}
                Arrival: {arrival}
                Arrival Time: {arrival_time}
                Duration: {duration}
                Ticket Currency: {ticket_currency}
                """.strip())

        return "\n\n".join(results)

    except Exception as e:
        return f"Serp API error: {str(e)}"


@tool
def get_flight_information_layovers_serpapi(
    departure_id: str,
    arrival_id: str,
    currency: str = "INR",
    outbound_date: str = date.today().strftime("%Y-%m-%d"),
    adults: int = 1,
    children: int = 0,
    sort_by: int = 2,
) -> list[dict]:
    """
    This tool uses SerpAPI to retrieve one-way flight (which have layovers) data from Google Flights. Only use this when you want to get flights with layovers. If the user wants direct flights, use the other tool.

    Args:
        departure_id: Origin airport IATA code.
        arrival_id: Destination airport IATA code.
        currency: Currency for pricing.
        outbound_date: Departure date (YYYY-MM-DD).
        adults: Number of adults.
        children: Number of children.
        sort_by:
            1 = Top flights
            2 = Price
            3 = Departure time
            4 = Arrival time
            5 = Duration
            6 = Emissions

    Returns:
        List of flight itinerary dictionaries.
    """

    api_key = os.getenv("SERP_API_KEY")

    if not api_key:
        return [{"error": "SERPAPI_API_KEY not found."}]

    client = serpapi.Client(api_key=api_key)

    try:

        response = client.search(
            {
                "engine": "google_flights",
                "departure_id": departure_id,
                "arrival_id": arrival_id,
                "outbound_date": outbound_date,
                "currency": currency,
                "hl": "en",
                "adults": adults,
                "children": children,
                "sort_by": sort_by,
                "type": 2,
                "stops": 2,
            }
        )

        offers = response.get("other_flights", [])

        if not offers:
            return []

        results = []

        for offer in offers:

            itinerary = {
                "price": offer.get("price"),
                "currency": currency,
                "trip_type": offer.get("type"),
                "total_duration_minutes": offer.get("total_duration"),
                "stops": len(offer.get("layovers", [])),
                "layovers": [],
                "segments": [],
                "carbon_emissions": offer.get("carbon_emissions", {}),
            }

            for layover in offer.get("layovers", []):

                itinerary["layovers"].append(
                    {
                        "airport": layover.get("name"),
                        "airport_code": layover.get("id"),
                        "duration_minutes": layover.get("duration"),
                    }
                )

            for flight in offer.get("flights", []):

                itinerary["segments"].append(
                    {
                        "airline": flight.get("airline"),
                        "flight_number": flight.get("flight_number"),
                        "aircraft": flight.get("airplane"),
                        "travel_class": flight.get("travel_class"),
                        "departure_airport": flight.get("departure_airport", {}).get(
                            "name"
                        ),
                        "departure_airport_code": flight.get(
                            "departure_airport", {}
                        ).get("id"),
                        "departure_time": flight.get("departure_airport", {}).get(
                            "time"
                        ),
                        "arrival_airport": flight.get("arrival_airport", {}).get(
                            "name"
                        ),
                        "arrival_airport_code": flight.get("arrival_airport", {}).get(
                            "id"
                        ),
                        "arrival_time": flight.get("arrival_airport", {}).get("time"),
                        "duration_minutes": flight.get("duration"),
                        "legroom": flight.get("legroom"),
                        "ticket_also_sold_by": flight.get("ticket_also_sold_by", []),
                        "overnight": flight.get("overnight", False),
                        "often_delayed": flight.get(
                            "often_delayed_by_over_30_min", False
                        ),
                        "extensions": flight.get("extensions", []),
                    }
                )

            results.append(itinerary)

        return results

    except Exception as e:
        return [{"error": f"Serp API error: {str(e)}"}]


@tool
def get_flight_information_bookingdotcom(
    from_id: str,
    to_id: str,
    depart_date: str = date.today().strftime("%Y-%m-%d"),
    adults: int = 1,
    children: str = "0",
    stops: str = "none",
    sort: str = "CHEAPEST",
    currency_code: str = "INR",
) -> list[dict]:
    """
    Search for flights on Booking.com via RapidAPI.

    Args:
        from_id: Origin location ID (e.g., 'BOM.AIRPORT').
        to_id: Destination location ID (e.g., 'DEL.AIRPORT').
        depart_date: Departure date in YYYY-MM-DD format.
        adults: Number of adult passengers.
        children: Child ages as comma-separated string (e.g., '0' or '8,12').
        stops: 'none', '0' (direct), '1', or '2'.
        sort: 'CHEAPEST', 'BEST', or 'FASTEST'.
        currency_code: 3-letter currency code (e.g., 'INR', 'USD').

    Returns:
        A list of simplified flight offer dictionaries.
    """
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        return [{"error": "RAPIDAPI_KEY environment variable not found."}]

    url = "https://booking-com15.p.rapidapi.com/api/v1/flights/searchFlights"

    params = {
        "fromId": from_id,
        "toId": to_id,
        "departDate": depart_date,
        "stops": stops,
        "pageNo": "1",
        "adults": str(adults),
        "children": children,
        "sort": sort,
        "cabinClass": "ECONOMY",
        "currency_code": currency_code,
    }

    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "booking-com15.p.rapidapi.com",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        flight_offers = data.get("data", {}).get("flightOffers", [])
        if not flight_offers:
            return []

        results = []
        for offer in flight_offers:
            total_price_obj = offer.get("priceBreakdown", {}).get("total", {})
            units = total_price_obj.get("units", 0)
            nanos = total_price_obj.get("nanos", 0)
            total_price = units + (nanos / 1e9)

            itinerary = {
                "trip_type": offer.get("tripType"),
                "total_price": round(total_price, 2),
                "currency": total_price_obj.get("currencyCode", currency_code),
                "seats_remaining": offer.get("seatAvailability", {}).get(
                    "numberOfSeatsAvailable"
                ),
                "segments": [],
            }

            for segment in offer.get("segments", []):
                for leg in segment.get("legs", []):
                    carrier_data = leg.get("carriersData", [{}])[0]
                    itinerary["segments"].append(
                        {
                            "airline": carrier_data.get("name"),
                            "flight_number": leg.get("flightInfo", {}).get(
                                "flightNumber"
                            ),
                            "departure_airport": leg.get("departureAirport", {}).get(
                                "code"
                            ),
                            "departure_time": leg.get("departureTime"),
                            "arrival_airport": leg.get("arrivalAirport", {}).get(
                                "code"
                            ),
                            "arrival_time": leg.get("arrivalTime"),
                            "cabin_class": leg.get("cabinClass"),
                            "duration_seconds": leg.get("totalTime"),
                        }
                    )

            results.append(itinerary)

        return results

    except requests.exceptions.RequestException as e:
        return [{"error": f"API Request failed: {str(e)}"}]
    except Exception as e:
        return [{"error": f"Unexpected error: {str(e)}"}]


@tool
def save_flight_direct_serpapi(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    airline: str,
    price: float,
    currency: str,
    departure_time: str,
    arrival_time: str,
    duration: int,
) -> str:
    """Saves direct flight data (SerpAPI) into FlightData_SerpApi_Direct table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO FlightData_SerpApi_Direct 
                (departure_airport, arrival_airport, outbound_date, airline, price, currency, departure_time, arrival_time, duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    departure_airport,
                    arrival_airport,
                    outbound_date,
                    airline,
                    price,
                    currency,
                    departure_time,
                    arrival_time,
                    duration,
                ),
            )
        return "Successfully saved direct flight search result."
    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def save_flight_layovers_serpapi(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    price: float,
    currency: str,
    stops: int,
    total_duration_minutes: int,
    itinerary_details: Dict[str, Any],
) -> str:
    """Saves layover flight itinerary data (SerpAPI) into FlightData_SerpApi_Layovers table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO FlightData_SerpApi_Layovers 
                (departure_airport, arrival_airport, outbound_date, price, currency, stops, total_duration_minutes, raw_itinerary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    departure_airport,
                    arrival_airport,
                    outbound_date,
                    price,
                    currency,
                    stops,
                    total_duration_minutes,
                    json.dumps(itinerary_details),
                ),
            )
        return "Successfully saved layover flight search result."
    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def save_flight_bookingdotcom(
    from_id: str,
    to_id: str,
    depart_date: str,
    total_price: float,
    currency: str,
    seats_remaining: Optional[int],
    itinerary_details: Dict[str, Any],
) -> str:
    """Saves Booking.com flight search results into FlightData_BookingDotCom table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO FlightData_BookingDotCom 
                (from_id, to_id, depart_date, total_price, currency, seats_remaining, raw_itinerary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    from_id,
                    to_id,
                    depart_date,
                    total_price,
                    currency,
                    seats_remaining,
                    json.dumps(itinerary_details),
                ),
            )
        return "Successfully saved Booking.com flight search result."
    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def read_flight_database(
    table_name: str,
    departure_code: Optional[str] = None,
    arrival_code: Optional[str] = None,
    depart_date: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Reads saved flight search records from any flight database table.

    Args:
        table_name: Must be one of 'FlightData_SerpApi_Direct',
          'FlightData_SerpApi_Layovers', or 'FlightData_BookingDotCom'.
        departure_code: Filter by origin airport/location ID (optional).
        arrival_code: Filter by destination airport/location ID (optional).
        depart_date: Filter by date YYYY-MM-DD (optional).
        limit: Max rows to return (default 10).
    """
    valid_tables = {
        "FlightData_SerpApi_Direct": (
            "departure_airport",
            "arrival_airport",
            "outbound_date",
        ),
        "FlightData_SerpApi_Layovers": (
            "departure_airport",
            "arrival_airport",
            "outbound_date",
        ),
        "FlightData_BookingDotCom": ("from_id", "to_id", "depart_date"),
    }

    if table_name not in valid_tables:
        return f"Invalid table name. Options: {list(valid_tables.keys())}"

    dep_col, arr_col, date_col = valid_tables[table_name]

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []

            if departure_code:
                query += f" AND {dep_col} LIKE ?"
                params.append(f"%{departure_code}%")
            if arrival_code:
                query += f" AND {arr_col} LIKE ?"
                params.append(f"%{arrival_code}%")
            if depart_date:
                query += f" AND {date_col} = ?"
                params.append(depart_date)

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                return f"No records found in {table_name} matching criteria."

            return str([dict(row) for row in rows])

    except Exception as e:
        return f"Database query error: {str(e)}"
