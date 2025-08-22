# Tunel.az Car Listings Scraper

A comprehensive Python web scraper for extracting car listing data from the Tunel.az API. This tool efficiently collects detailed vehicle information including seller contact details, specifications, and pricing data.

## 🚗 Features

- **Complete API Coverage**: Scrapes all available car listings from Tunel.az
- **Detailed Data Extraction**: Collects comprehensive vehicle information including:
  - Vehicle specifications (brand, model, year, mileage, etc.)
  - Pricing and financial details
  - Seller contact information (names, phone numbers, locations)
  - Equipment and features lists
  - Images and view statistics
- **Multiple Output Formats**: Exports data to both CSV and JSON formats
- **Rate Limiting**: Built-in delays to respect server resources
- **Robust Error Handling**: Comprehensive logging and error recovery
- **Progress Tracking**: Real-time progress updates during scraping operations

## 📋 Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`:
  - `requests==2.32.3` - HTTP requests handling
  - `zstandard==0.23.0` - Data compression support

## 🚀 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd tunel_az
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

Run the scraper with default settings:
```bash
python tunel_scraper.py
```

### Customization

Edit the `main()` function parameters in `tunel_scraper.py`:

```python
# Configuration options
MAX_PAGES = 100  # Limit number of pages (None for all)
ANNOUNCEMENT_TYPE = "sale"  # "sale", "rent", etc.
```

### API Methods

The `TunelScraper` class provides several methods:

#### Core Scraping Methods
- `fetch_listings_page(page, limit, announcement_type)` - Fetch a single page of listings
- `fetch_listing_details(slug)` - Get detailed information for a specific listing
- `get_all_listings(max_pages, announcement_type)` - Collect all listing slugs
- `scrape_all_data(max_pages, announcement_type)` - Full scraping workflow

#### Data Processing Methods
- `extract_key_data(detailed_data)` - Extract essential information from raw data
- `save_to_csv(data, filename)` - Export data to CSV format
- `save_to_json(data, filename)` - Export data to JSON format

## 📁 Project Structure

```
tunel_az/
├── tunel_scraper.py           # Main scraper implementation
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── data/                      # Output directory
    ├── tunel_listings_extracted.csv    # Key data in CSV format
    ├── tunel_listings_extracted.json   # Key data in JSON format
    ├── tunel_listings_full.json        # Complete raw data
    ├── listings api parameters.txt     # API documentation
    └── one listing api parameters.txt  # Single listing API docs
```

## 📊 Data Fields Extracted

### Vehicle Information
- ID, slug, announcement number
- Price (value and currency)
- Year, mileage, and mileage unit
- Brand, model, color
- Fuel type, gearbox, transmission
- Engine capacity and power
- Number of seats, vehicle status

### Seller Information
- Name and phone number
- City and account type
- Username and verification status

### Additional Details
- View count, barter/credit options
- Creation and update timestamps
- Equipment list and image count
- Description and target country

## 🛡️ Rate Limiting & Best Practices

- Default delay of 1 second between requests
- Comprehensive error handling for network issues
- Respectful scraping practices with proper headers
- Progress logging every 10 processed items

## 📝 Logging

The scraper uses Python's logging module with INFO level by default. Logs include:
- Progress updates during scraping
- Error messages for failed requests
- Statistics about collected data
- Completion status and summary

## ⚠️ Legal Considerations

This scraper is designed for educational and research purposes. Users should:
- Respect Tunel.az's terms of service
- Use collected data responsibly
- Consider rate limiting and server load
- Ensure compliance with local data protection laws

## 🔧 Technical Details

### Request Headers
The scraper uses realistic browser headers to ensure proper API access:
- User-Agent mimicking Chrome browser
- Proper Accept and encoding headers
- Device and session information
- Timezone and screen resolution data

### Error Handling
- HTTP request timeouts (30 seconds)
- JSON parsing error recovery
- API response validation
- Graceful handling of missing data

## 🤝 Contributing

Contributions are welcome! Please consider:
- Adding new data extraction fields
- Improving error handling
- Optimizing performance
- Adding export format options

## 📄 License

This project is provided as-is for educational purposes. Please ensure compliance with applicable laws and terms of service.

---

*Last updated: August 2025*