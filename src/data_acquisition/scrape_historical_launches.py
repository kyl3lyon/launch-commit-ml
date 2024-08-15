import requests                         # Making HTTP requests
from bs4 import BeautifulSoup            # Parsing HTML
from typing import List, Dict, Optional  # Type hinting
import logging                           # Logging
from datetime import datetime            # Datetime
from urllib.parse import urlencode       # URL encoding
import random                            # Selecting random User-Agent
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of common User-Agent strings
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

# --- Main Execution ---

def scrape_space_launches(
    search: str = "",
    launch_service_provider: str = "",
    status: str = "",
    location: str = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_pages: int = 10,
    output_file: str = "space_launches.json"
) -> List[Dict[str, str]]:
    """
    Scrape launch data from Space Launch Now website based on provided filters.

    Args:
        search (str): Search term to filter launches.
        launch_service_provider (str): Launch service provider to filter launches.
        status (str): Launch status to filter launches.
        location (str): Launch location to filter launches.
        start_date (Optional[str]): Start date to filter launches.
        end_date (Optional[str]): End date to filter launches.
        max_pages (int): Maximum number of pages to scrape.
        output_file (str): Path to save the JSON output file.

    Returns:
        List[Dict[str, str]]: List of dictionaries containing launch data.
    """

    base_url = "https://spacelaunchnow.me/launch/"
    launches = []

    # --- Validate data formats ---
    if start_date is not None:
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid start date format. Expected YYYY-MM-DD.")
    if end_date is not None:
        try:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid end date format. Expected YYYY-MM-DD.")

    # --- Scrape data ---
    session = requests.Session()

    for page in range(1, max_pages + 1):
        params = {
            "search": search,
            "launch_service_provider": launch_service_provider,
            "status": status,
            "location": location,
            "start_date": start_date.strftime("%Y-%m-%d") if start_date else None,
            "end_date": end_date.strftime("%Y-%m-%d") if end_date else None,
            "page": page,
        }
        url = f"{base_url}?{urlencode(params)}"

        # Rotate User-Agent for each request
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
        }

        try:
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error fetcing page {page}: {e}")
            break

        # --- Parse HTML ---
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", class_="table")

        if not table:
            logger.info(f"No table found on page {page}.")
            break

        rows = table.find_all("tr")[1:]  # Skip the header row
        for row in rows:
            cols = row.find_all("td")
            if len(cols)  == 7:
                launch = {
                    "name": cols[0].text.strip(),
                    "status": cols[1].text.strip(),
                    "provider": cols[2].text.strip(),
                    "rocket": cols[3].text.strip(),
                    "mission": cols[4].text.strip(),
                    "date": cols[5].text.strip(),
                    "location": cols[6].text.strip()
                }
                launches.append(launch)

            logger.info(f"Scraped page {page}, total launches: {len(launches)}")

            # Check if we've reached the max number of pages
            next_button = soup.find("li", class_="next")
            if not next_button:
                logger.info("Reached last page.")
                break

    # Save data as JSON
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(launches, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {output_file}")
    except IOError as e:
        logger.error(f"Error saving data to {output_file}: {e}")

    return launches


# --- Usage ---
if __name__ == "__main__":
    output_directory = "src/data"
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, "historical_launches.json")

    results = scrape_space_launches(
        search="",
        start_date="1957-10-04",
        end_date="2024-09-15",
        max_pages=278,
        output_file=output_file
    )
    print(f"Total launches found: {len(results)}")
    print(f"Data saved to: {output_file}")

    # Print first 5 launches as a sample
    for launch in results[:5]:
        print(json.dumps(launch, indent=2))