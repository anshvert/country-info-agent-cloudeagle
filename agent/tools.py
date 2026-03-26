import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_BASE_URL = "https://restcountries.com/v3.1/name/{name}"

_FIELD_MAP = {
    "name": ["name"],
    "population": ["population"],
    "capital": ["capital"],
    "currency": ["currencies"],
    "currencies": ["currencies"],
    "language": ["languages"],
    "languages": ["languages"],
    "region": ["region"],
    "subregion": ["subregion"],
    "area": ["area"],
    "flag": ["flags"],
    "timezone": ["timezones"],
    "timezones": ["timezones"],
    "continent": ["continents"],
    "borders": ["borders"],
    "calling code": ["idd"],
    "phone code": ["idd"],
}

_API_FIELDS = [
    "name", "population", "capital", "currencies", "languages",
    "region", "subregion", "area", "flags", "timezones",
    "continents", "borders", "idd",
]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(httpx.TransportError),
    reraise=True,
)
async def fetch_country(country_name: str) -> dict:
    """Fetch country data from REST Countries API. Retries up to 3x on network errors."""
    url = _BASE_URL.format(name=country_name)
    params = {"fields": ",".join(_API_FIELDS)}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)

    if resp.status_code == 404:
        raise ValueError(f"Country '{country_name}' not found.")
    resp.raise_for_status()
    results = resp.json()
    return results[0] if results else {}
