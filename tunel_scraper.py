#!/usr/bin/env python3
"""
Tunel.az API Scraper
Scrapes car listings data from tunel.az API including phone numbers
"""

import requests
import json
import csv
import time
from typing import Dict, List, Optional
from urllib.parse import quote
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TunelScraper:
    def __init__(self):
        self.base_url = "https://api.tunel.az/api/announcements"
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'az',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Origin': 'https://tunel.az',
            'Referer': 'https://tunel.az/',
            'DNT': '1',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'X-Browser': 'chrome',
            'X-Browser-Version': '139.0',
            'X-Device-Type': 'mobile',
            'X-OS': 'ipad',
            'X-Session-Type': 'web',
            'X-Timezone': 'Asia/Baku'
        }
        self.session.headers.update(self.headers)
        self.delay = 1  # Delay between requests in seconds
    
    def fetch_listings_page(self, page: int = 1, limit: int = 24, announcement_type: str = "sale") -> Optional[Dict]:
        """
        Fetch a page of listings from the API
        
        Args:
            page: Page number to fetch
            limit: Number of listings per page
            announcement_type: Type of announcement ('sale', 'rent', etc.)
        
        Returns:
            API response data or None if error
        """
        try:
            params = {
                'page': page,
                'limit': limit,
                'announcement_type': announcement_type
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') == 200 and data.get('status'):
                return data
            else:
                logger.error(f"API error on page {page}: {data.get('message', 'Unknown error')}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Request failed for page {page}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for page {page}: {str(e)}")
            return None
    
    def fetch_listing_details(self, slug: str) -> Optional[Dict]:
        """
        Fetch detailed information for a specific listing using its slug
        
        Args:
            slug: The slug identifier for the listing
        
        Returns:
            Detailed listing data or None if error
        """
        try:
            # URL encode the slug to handle special characters
            encoded_slug = quote(slug, safe='-')
            url = f"{self.base_url}/{encoded_slug}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') == 200 and data.get('status'):
                return data.get('data')
            else:
                logger.error(f"API error for slug {slug}: {data.get('message', 'Unknown error')}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Request failed for slug {slug}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for slug {slug}: {str(e)}")
            return None
    
    def get_all_listings(self, max_pages: Optional[int] = None, announcement_type: str = "sale") -> List[str]:
        """
        Fetch all listing slugs from all pages
        
        Args:
            max_pages: Maximum number of pages to fetch (None for all)
            announcement_type: Type of announcement to fetch
        
        Returns:
            List of slugs
        """
        slugs = []
        page = 1
        
        logger.info(f"Starting to fetch listing slugs for {announcement_type} announcements...")
        
        while True:
            if max_pages and page > max_pages:
                break
                
            logger.info(f"Fetching page {page}...")
            data = self.fetch_listings_page(page, announcement_type=announcement_type)
            
            if not data or not data.get('data', {}).get('data'):
                logger.info(f"No more data available. Stopped at page {page}")
                break
            
            listings = data['data']['data']
            page_slugs = [listing.get('slug') for listing in listings if listing.get('slug')]
            slugs.extend(page_slugs)
            
            logger.info(f"Page {page}: Found {len(page_slugs)} slugs")
            
            # Check if this is the last page
            current_page = data['data'].get('current_page', page)
            last_page = data['data'].get('last_page', page)
            
            if current_page >= last_page:
                logger.info(f"Reached last page: {last_page}")
                break
            
            page += 1
            time.sleep(self.delay)  # Rate limiting
        
        logger.info(f"Total slugs collected: {len(slugs)}")
        return slugs
    
    def scrape_all_data(self, max_pages: Optional[int] = None, announcement_type: str = "sale") -> List[Dict]:
        """
        Scrape all detailed data for all listings
        
        Args:
            max_pages: Maximum number of pages to process
            announcement_type: Type of announcement to scrape
        
        Returns:
            List of detailed listing data
        """
        # Get all slugs first
        slugs = self.get_all_listings(max_pages, announcement_type)
        
        if not slugs:
            logger.warning("No slugs found to process")
            return []
        
        detailed_data = []
        logger.info(f"Starting to fetch detailed data for {len(slugs)} listings...")
        
        for i, slug in enumerate(slugs, 1):
            logger.info(f"Processing listing {i}/{len(slugs)}: {slug}")
            
            details = self.fetch_listing_details(slug)
            if details:
                detailed_data.append(details)
                logger.info(f"Successfully fetched data for {slug}")
            else:
                logger.warning(f"Failed to fetch data for {slug}")
            
            time.sleep(self.delay)  # Rate limiting
            
            # Progress update every 10 items
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(slugs)} completed")
        
        logger.info(f"Scraping completed. Successfully fetched {len(detailed_data)} detailed listings")
        return detailed_data
    
    def extract_key_data(self, detailed_data: List[Dict]) -> List[Dict]:
        """
        Extract key information from detailed listing data
        
        Args:
            detailed_data: List of detailed listing data
        
        Returns:
            List of extracted key data
        """
        extracted = []
        
        for listing in detailed_data:
            try:
                # Extract key information
                extracted_item = {
                    # Basic info
                    'id': listing.get('id'),
                    'slug': listing.get('slug'),
                    'announcement_number': listing.get('announcement_number'),
                    'description': listing.get('description'),
                    
                    # Vehicle info
                    'price_value': listing.get('price_value'),
                    'price_unit': listing.get('price_unit', {}).get('label'),
                    'year': listing.get('year'),
                    'mileage_value': listing.get('mileage_value'),
                    'mileage_unit': listing.get('mileage_unit', {}).get('label'),
                    'brand': listing.get('marka', {}).get('label'),
                    'model': listing.get('model', {}).get('label'),
                    'color': listing.get('color', {}).get('label'),
                    'fuel': listing.get('fuel', {}).get('label'),
                    'gearbox': listing.get('gearbox', {}).get('label'),
                    'transmission': listing.get('transmission', {}).get('label'),
                    'engine_capacity': listing.get('engine_capacity', {}).get('label'),
                    'engine_power': listing.get('engine_power'),
                    'seats': listing.get('seats', {}).get('label'),
                    'status': listing.get('status', {}).get('label'),
                    'ban': listing.get('ban', {}).get('label'),
                    'target_country': listing.get('target_country', {}).get('label'),
                    
                    # Contact info (most important for scraping)
                    'seller_name': listing.get('publishment', {}).get('name'),
                    'seller_phone': listing.get('publishment', {}).get('phone'),
                    'seller_city': listing.get('publishment', {}).get('city', {}).get('label'),
                    'seller_account_type': listing.get('publishment', {}).get('account_type', {}).get('label'),
                    'seller_username': listing.get('publishment', {}).get('account_username'),
                    'seller_verified': listing.get('publishment', {}).get('verified'),
                    
                    # Additional info
                    'view_count': listing.get('view_count'),
                    'barter': listing.get('barter'),
                    'credit': listing.get('credit'),
                    'created_at': listing.get('created_at'),
                    'updated_at': listing.get('updated_at'),
                    'images_count': len(listing.get('images', [])),
                    'equipments_count': len(listing.get('equipments', [])),
                    
                    # Equipment list
                    'equipments': [eq.get('label') for eq in listing.get('equipments', [])],
                }
                
                extracted.append(extracted_item)
                
            except Exception as e:
                logger.error(f"Error extracting data for listing {listing.get('id', 'unknown')}: {str(e)}")
                continue
        
        return extracted
    
    def save_to_csv(self, data: List[Dict], filename: str = "tunel_listings.csv"):
        """
        Save data to CSV file
        
        Args:
            data: List of dictionaries to save
            filename: Output filename
        """
        if not data:
            logger.warning("No data to save to CSV")
            return
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for row in data:
                    # Convert lists to string for CSV
                    row_copy = row.copy()
                    if 'equipments' in row_copy and isinstance(row_copy['equipments'], list):
                        row_copy['equipments'] = '; '.join(row_copy['equipments'])
                    writer.writerow(row_copy)
            
            logger.info(f"Data saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
    
    def save_to_json(self, data: List[Dict], filename: str = "tunel_listings.json"):
        """
        Save data to JSON file
        
        Args:
            data: List of dictionaries to save
            filename: Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, ensure_ascii=False, indent=2)
            
            logger.info(f"Data saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")


def main():
    """
    Main function to run the scraper
    """
    scraper = TunelScraper()
    
    # You can modify these parameters as needed
    MAX_PAGES = 100  # Set to a number to limit pages, None for all pages
    ANNOUNCEMENT_TYPE = "sale"  # "sale", "rent", etc.
    
    try:
        # Scrape all data
        detailed_data = scraper.scrape_all_data(max_pages=MAX_PAGES, announcement_type=ANNOUNCEMENT_TYPE)
        
        if not detailed_data:
            logger.error("No data was scraped")
            return
        
        # Extract key information
        extracted_data = scraper.extract_key_data(detailed_data)
        
        # Save to files
        scraper.save_to_csv(extracted_data, "data/tunel_listings_extracted.csv")
        scraper.save_to_json(extracted_data, "data/tunel_listings_extracted.json")
        scraper.save_to_json(detailed_data, "data/tunel_listings_full.json")
        
        logger.info(f"Scraping completed successfully! Found {len(extracted_data)} listings with phone numbers")
        
        # Print some statistics
        phone_numbers = [item['seller_phone'] for item in extracted_data if item.get('seller_phone')]
        logger.info(f"Phone numbers found: {len(phone_numbers)}")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()