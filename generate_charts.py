#!/usr/bin/env python3
"""
Business Analytics Dashboard Generator
Generates executive-level insights and visualizations for car marketplace data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set professional styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11

class MarketplaceAnalytics:
    def __init__(self, csv_path):
        """Initialize analytics with data loading and preparation"""
        print("Loading marketplace data...")
        self.df = pd.read_csv(csv_path)
        self.charts_dir = Path('charts')
        self.charts_dir.mkdir(exist_ok=True)

        # Data preparation
        self._prepare_data()

        print(f"Loaded {len(self.df):,} vehicle listings")
        print(f"Data period: {self.df['created_at'].min()} to {self.df['created_at'].max()}")

    def _prepare_data(self):
        """Clean and prepare data for analysis"""
        # Convert dates
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
        self.df['updated_at'] = pd.to_datetime(self.df['updated_at'], errors='coerce')

        # Extract time features
        self.df['listing_age_days'] = (pd.Timestamp.now() - self.df['created_at']).dt.days
        self.df['created_month'] = self.df['created_at'].dt.to_period('M')

        # Clean numeric fields
        self.df['price_value'] = pd.to_numeric(self.df['price_value'], errors='coerce')
        self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
        self.df['mileage_value'] = pd.to_numeric(self.df['mileage_value'], errors='coerce')
        self.df['view_count'] = pd.to_numeric(self.df['view_count'], errors='coerce')
        self.df['engine_power'] = pd.to_numeric(self.df['engine_power'], errors='coerce')

        # Calculate vehicle age
        current_year = pd.Timestamp.now().year
        self.df['vehicle_age'] = current_year - self.df['year']

        # Clean boolean fields
        self.df['barter'] = self.df['barter'].fillna(False)
        self.df['credit'] = self.df['credit'].fillna(False)
        self.df['seller_verified'] = self.df['seller_verified'].fillna(False)

    def generate_all_charts(self):
        """Generate all business insights charts"""
        print("\nGenerating business analytics charts...")

        # Market Overview
        self.chart_inventory_by_brand()
        self.chart_price_distribution_by_segment()
        self.chart_market_supply_trends()

        # Pricing Intelligence
        self.chart_average_price_by_brand()
        self.chart_price_vs_age_analysis()
        self.chart_price_by_condition()

        # Seller Performance
        self.chart_seller_distribution()
        self.chart_top_sellers_market_share()

        # Market Demand Indicators
        self.chart_most_viewed_categories()
        self.chart_engagement_by_price_range()

        # Vehicle Characteristics
        self.chart_fuel_type_distribution()
        self.chart_transmission_preferences()
        self.chart_vehicle_age_distribution()

        # Geographic Insights
        self.chart_inventory_by_city()

        # Payment & Features
        self.chart_payment_flexibility()
        self.chart_year_distribution()

        print(f"\n✓ All charts generated successfully in '{self.charts_dir}/' directory")

    def chart_inventory_by_brand(self):
        """Market Share: Top brands by inventory count"""
        top_brands = self.df['brand'].value_counts().head(15)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(top_brands)), top_brands.values, color='#2E86AB')
        ax.set_yticks(range(len(top_brands)))
        ax.set_yticklabels(top_brands.index)
        ax.set_xlabel('Number of Listings')
        ax.set_title('Market Inventory Distribution - Top 15 Brands', fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_brands.values)):
            ax.text(value + 50, i, f'{value:,}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '01_market_inventory_by_brand.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_price_distribution_by_segment(self):
        """Price segmentation analysis"""
        # Define price segments
        price_data = self.df[self.df['price_value'] > 0].copy()
        price_data['price_segment'] = pd.cut(
            price_data['price_value'],
            bins=[0, 10000, 20000, 30000, 50000, 100000, float('inf')],
            labels=['Under 10K', '10K-20K', '20K-30K', '30K-50K', '50K-100K', 'Over 100K']
        )

        segment_counts = price_data['price_segment'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(segment_counts)), segment_counts.values, color='#A23B72', alpha=0.8)
        ax.set_xticks(range(len(segment_counts)))
        ax.set_xticklabels(segment_counts.index, rotation=0)
        ax.set_ylabel('Number of Listings')
        ax.set_title('Inventory Distribution by Price Segment', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels and percentages
        total = segment_counts.sum()
        for i, (bar, value) in enumerate(zip(bars, segment_counts.values)):
            pct = (value / total) * 100
            ax.text(i, value + 50, f'{value:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '02_price_segment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_market_supply_trends(self):
        """New listing trends over time"""
        monthly_listings = self.df.groupby('created_month').size()

        # Get last 12 months
        recent_months = monthly_listings.tail(12)

        fig, ax = plt.subplots(figsize=(14, 6))
        x = range(len(recent_months))
        ax.plot(x, recent_months.values, marker='o', linewidth=2.5, markersize=8, color='#F18F01')
        ax.fill_between(x, recent_months.values, alpha=0.3, color='#F18F01')

        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in recent_months.index], rotation=45)
        ax.set_ylabel('New Listings')
        ax.set_title('Market Supply Trends - Monthly New Listings (Last 12 Months)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, value in enumerate(recent_months.values):
            ax.text(i, value + 20, f'{value:,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '03_market_supply_trends.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_average_price_by_brand(self):
        """Average pricing by top brands"""
        # Calculate average price per brand (top 15 by volume)
        top_brands = self.df['brand'].value_counts().head(15).index
        price_by_brand = self.df[self.df['brand'].isin(top_brands)].groupby('brand')['price_value'].mean().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(price_by_brand)), price_by_brand.values, color='#06A77D')
        ax.set_yticks(range(len(price_by_brand)))
        ax.set_yticklabels(price_by_brand.index)
        ax.set_xlabel('Average Price (Currency Units)')
        ax.set_title('Average Vehicle Price by Brand (Top 15 Brands)', fontsize=14, fontweight='bold', pad=20)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, price_by_brand.values)):
            ax.text(value + 500, i, f'{value:,.0f}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '04_average_price_by_brand.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_price_vs_age_analysis(self):
        """Price depreciation analysis by vehicle age"""
        # Filter reasonable data
        valid_data = self.df[
            (self.df['vehicle_age'] >= 0) &
            (self.df['vehicle_age'] <= 20) &
            (self.df['price_value'] > 0) &
            (self.df['price_value'] < 200000)
        ].copy()

        # Calculate average price by age
        age_price = valid_data.groupby('vehicle_age')['price_value'].agg(['mean', 'median', 'count'])
        age_price = age_price[age_price['count'] >= 50]  # Filter groups with sufficient data

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(age_price.index, age_price['mean'], marker='o', linewidth=2.5, label='Average Price', color='#D62828', markersize=8)
        ax.plot(age_price.index, age_price['median'], marker='s', linewidth=2.5, label='Median Price', color='#003049', markersize=7, linestyle='--')

        ax.set_xlabel('Vehicle Age (Years)')
        ax.set_ylabel('Price (Currency Units)')
        ax.set_title('Price Depreciation Analysis - Average Price by Vehicle Age', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '05_price_depreciation_by_age.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_price_by_condition(self):
        """Price comparison by vehicle condition"""
        condition_price = self.df.groupby('ban')['price_value'].median().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(range(len(condition_price)), condition_price.values, color='#6A4C93', alpha=0.8)
        ax.set_xticks(range(len(condition_price)))
        ax.set_xticklabels(condition_price.index, rotation=45, ha='right')
        ax.set_ylabel('Median Price (Currency Units)')
        ax.set_title('Median Price by Vehicle Condition Type', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, condition_price.values)):
            ax.text(i, value + 500, f'{value:,.0f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '06_price_by_condition.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_seller_distribution(self):
        """Seller type distribution"""
        seller_dist = self.df['seller_account_type'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(seller_dist)), seller_dist.values, color=['#E63946', '#F77F00', '#06A77D'], alpha=0.8)
        ax.set_xticks(range(len(seller_dist)))
        ax.set_xticklabels(seller_dist.index, rotation=0)
        ax.set_ylabel('Number of Listings')
        ax.set_title('Market Composition - Listings by Seller Type', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels and percentages
        total = seller_dist.sum()
        for i, (bar, value) in enumerate(zip(bars, seller_dist.values)):
            pct = (value / total) * 100
            ax.text(i, value + 100, f'{value:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.charts_dir / '07_seller_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_top_sellers_market_share(self):
        """Top sellers by listing volume"""
        # Analyze sellers with multiple listings (likely dealers)
        seller_counts = self.df['seller_username'].value_counts().head(20)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(seller_counts)), seller_counts.values, color='#8338EC')
        ax.set_yticks(range(len(seller_counts)))
        ax.set_yticklabels([f"Seller {i+1}" for i in range(len(seller_counts))])  # Anonymize sellers
        ax.set_xlabel('Number of Active Listings')
        ax.set_title('Top 20 Sellers by Inventory Volume', fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, seller_counts.values)):
            ax.text(value + 2, i, f'{value:,}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '08_top_sellers_inventory.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_most_viewed_categories(self):
        """Average engagement by brand"""
        # Top brands by average views
        top_brands = self.df['brand'].value_counts().head(15).index
        brand_views = self.df[self.df['brand'].isin(top_brands)].groupby('brand')['view_count'].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(brand_views)), brand_views.values, color='#FF6B35')
        ax.set_yticks(range(len(brand_views)))
        ax.set_yticklabels(brand_views.index)
        ax.set_xlabel('Average Views per Listing')
        ax.set_title('Market Demand Indicators - Average Views by Brand', fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, brand_views.values)):
            ax.text(value + 5, i, f'{value:.0f}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '09_demand_by_brand.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_engagement_by_price_range(self):
        """Engagement analysis by price segment"""
        price_data = self.df[(self.df['price_value'] > 0) & (self.df['view_count'] > 0)].copy()
        price_data['price_segment'] = pd.cut(
            price_data['price_value'],
            bins=[0, 10000, 20000, 30000, 50000, 100000, float('inf')],
            labels=['Under 10K', '10K-20K', '20K-30K', '30K-50K', '50K-100K', 'Over 100K']
        )

        engagement = price_data.groupby('price_segment')['view_count'].mean().sort_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(engagement)), engagement.values, color='#FB5607', alpha=0.8)
        ax.set_xticks(range(len(engagement)))
        ax.set_xticklabels(engagement.index, rotation=0)
        ax.set_ylabel('Average Views per Listing')
        ax.set_title('Customer Engagement by Price Segment', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, engagement.values)):
            ax.text(i, value + 2, f'{value:.0f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '10_engagement_by_price.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_fuel_type_distribution(self):
        """Market preference for fuel types"""
        fuel_dist = self.df['fuel'].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(range(len(fuel_dist)), fuel_dist.values, color='#3A86FF', alpha=0.8)
        ax.set_xticks(range(len(fuel_dist)))
        ax.set_xticklabels(fuel_dist.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Listings')
        ax.set_title('Inventory Distribution by Fuel Type', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels and percentages
        total = fuel_dist.sum()
        for i, (bar, value) in enumerate(zip(bars, fuel_dist.values)):
            pct = (value / total) * 100
            ax.text(i, value + 50, f'{value:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '11_fuel_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_transmission_preferences(self):
        """Transmission type distribution"""
        trans_dist = self.df['gearbox'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(trans_dist)), trans_dist.values, color='#8338EC', alpha=0.8)
        ax.set_xticks(range(len(trans_dist)))
        ax.set_xticklabels(trans_dist.index, rotation=0)
        ax.set_ylabel('Number of Listings')
        ax.set_title('Inventory Distribution by Transmission Type', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels and percentages
        total = trans_dist.sum()
        for i, (bar, value) in enumerate(zip(bars, trans_dist.values)):
            pct = (value / total) * 100
            ax.text(i, value + 100, f'{value:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.charts_dir / '12_transmission_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_vehicle_age_distribution(self):
        """Distribution of vehicle ages in inventory"""
        age_data = self.df[(self.df['vehicle_age'] >= 0) & (self.df['vehicle_age'] <= 30)]

        fig, ax = plt.subplots(figsize=(14, 6))
        age_counts = age_data['vehicle_age'].value_counts().sort_index()

        ax.bar(age_counts.index, age_counts.values, color='#06A77D', alpha=0.7)
        ax.set_xlabel('Vehicle Age (Years)')
        ax.set_ylabel('Number of Listings')
        ax.set_title('Inventory Age Distribution - Vehicles by Age', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '13_vehicle_age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_inventory_by_city(self):
        """Geographic distribution of inventory"""
        city_dist = self.df['seller_city'].value_counts().head(15)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(city_dist)), city_dist.values, color='#D62828')
        ax.set_yticks(range(len(city_dist)))
        ax.set_yticklabels(city_dist.index)
        ax.set_xlabel('Number of Listings')
        ax.set_title('Geographic Distribution - Top 15 Cities by Inventory', fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()

        # Add value labels and percentages
        total = city_dist.sum()
        for i, (bar, value) in enumerate(zip(bars, city_dist.values)):
            pct = (value / total) * 100
            ax.text(value + 50, i, f'{value:,} ({pct:.1f}%)', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '14_inventory_by_city.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_payment_flexibility(self):
        """Payment options analysis"""
        payment_data = pd.DataFrame({
            'Credit Available': [self.df['credit'].sum(), len(self.df) - self.df['credit'].sum()],
            'Barter Accepted': [self.df['barter'].sum(), len(self.df) - self.df['barter'].sum()]
        }, index=['Yes', 'No'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Credit availability
        colors = ['#06A77D', '#CCCCCC']
        bars1 = axes[0].bar(range(2), payment_data['Credit Available'].values, color=colors, alpha=0.8)
        axes[0].set_xticks(range(2))
        axes[0].set_xticklabels(payment_data.index)
        axes[0].set_ylabel('Number of Listings')
        axes[0].set_title('Credit Financing Availability', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        for i, (bar, value) in enumerate(zip(bars1, payment_data['Credit Available'].values)):
            pct = (value / payment_data['Credit Available'].sum()) * 100
            axes[0].text(i, value + 100, f'{value:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Barter acceptance
        bars2 = axes[1].bar(range(2), payment_data['Barter Accepted'].values, color=colors, alpha=0.8)
        axes[1].set_xticks(range(2))
        axes[1].set_xticklabels(payment_data.index)
        axes[1].set_ylabel('Number of Listings')
        axes[1].set_title('Barter/Trade-in Acceptance', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        for i, (bar, value) in enumerate(zip(bars2, payment_data['Barter Accepted'].values)):
            pct = (value / payment_data['Barter Accepted'].sum()) * 100
            axes[1].text(i, value + 100, f'{value:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.suptitle('Payment Flexibility Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.charts_dir / '15_payment_options.png', dpi=300, bbox_inches='tight')
        plt.close()

    def chart_year_distribution(self):
        """Distribution of vehicle manufacturing years"""
        year_data = self.df[(self.df['year'] >= 2000) & (self.df['year'] <= 2024)]
        year_counts = year_data['year'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.bar(year_counts.index, year_counts.values, color='#F18F01', alpha=0.7)
        ax.set_xlabel('Manufacturing Year')
        ax.set_ylabel('Number of Listings')
        ax.set_title('Inventory Distribution by Manufacturing Year (2000-2024)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(self.charts_dir / '16_year_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def print_summary_statistics(self):
        """Print key business metrics"""
        print("\n" + "="*70)
        print("EXECUTIVE SUMMARY - KEY BUSINESS METRICS")
        print("="*70)

        print(f"\n📊 MARKET SIZE")
        print(f"   Total Active Listings: {len(self.df):,}")
        print(f"   Unique Brands: {self.df['brand'].nunique():,}")
        print(f"   Unique Models: {self.df['model'].nunique():,}")

        print(f"\n💰 PRICING INSIGHTS")
        print(f"   Average Listing Price: {self.df['price_value'].mean():,.0f}")
        print(f"   Median Listing Price: {self.df['price_value'].median():,.0f}")
        print(f"   Price Range: {self.df['price_value'].min():,.0f} - {self.df['price_value'].max():,.0f}")

        print(f"\n👥 SELLER COMPOSITION")
        seller_types = self.df['seller_account_type'].value_counts()
        for seller_type, count in seller_types.items():
            pct = (count / len(self.df)) * 100
            print(f"   {seller_type}: {count:,} ({pct:.1f}%)")

        print(f"\n🔍 MARKET DEMAND")
        print(f"   Average Views per Listing: {self.df['view_count'].mean():.0f}")
        print(f"   Median Views per Listing: {self.df['view_count'].median():.0f}")
        print(f"   Total Market Views: {self.df['view_count'].sum():,.0f}")

        print(f"\n🚗 INVENTORY CHARACTERISTICS")
        print(f"   Average Vehicle Age: {self.df['vehicle_age'].mean():.1f} years")
        print(f"   Median Vehicle Age: {self.df['vehicle_age'].median():.1f} years")
        print(f"   Average Mileage: {self.df['mileage_value'].mean():,.0f} km")

        print(f"\n💳 PAYMENT FLEXIBILITY")
        credit_pct = (self.df['credit'].sum() / len(self.df)) * 100
        barter_pct = (self.df['barter'].sum() / len(self.df)) * 100
        print(f"   Listings Offering Credit: {self.df['credit'].sum():,} ({credit_pct:.1f}%)")
        print(f"   Listings Accepting Barter: {self.df['barter'].sum():,} ({barter_pct:.1f}%)")

        print(f"\n🌍 GEOGRAPHIC REACH")
        print(f"   Cities Covered: {self.df['seller_city'].nunique():,}")
        top_city = self.df['seller_city'].value_counts().index[0]
        top_city_count = self.df['seller_city'].value_counts().values[0]
        top_city_pct = (top_city_count / len(self.df)) * 100
        print(f"   Top City: {top_city} ({top_city_count:,} listings, {top_city_pct:.1f}%)")

        print("\n" + "="*70)


def main():
    """Main execution function"""
    print("="*70)
    print("CAR MARKETPLACE BUSINESS ANALYTICS")
    print("Generating Executive Insights & Visualizations")
    print("="*70)

    # Initialize analytics
    analytics = MarketplaceAnalytics('data/tunel_listings_extracted.csv')

    # Generate all charts
    analytics.generate_all_charts()

    # Print summary statistics
    analytics.print_summary_statistics()

    print("\n✓ Analysis complete! Check the 'charts/' directory for all visualizations.")
    print("✓ Proceed to README.md for executive business insights.\n")


if __name__ == "__main__":
    main()
