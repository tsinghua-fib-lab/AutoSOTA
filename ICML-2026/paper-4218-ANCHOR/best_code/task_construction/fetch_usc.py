#!/usr/bin/env python3
"""
Fetch USC (United States Code) content from Cornell Law website.
Usage: python fetch_usc.py <title> <section>
Example: python fetch_usc.py 11 111
"""

import os
import sys
import requests
from bs4 import BeautifulSoup
import re
import json


def fetch_usc_content(title, section):
    """
    Fetch USC content from Cornell Law website for given title and section.

    Args:
        title (int): USC Title number
        section (int): USC Section number

    Returns:
        dict: Dictionary with section_name and section_definition
    """
    url = f"https://www.law.cornell.edu/uscode/text/{title}/{section}"

    try:
        # Fetch the webpage
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Initialize result dictionary
        result = {
            "section_name": "",
            "section_definition": ""
        }

        # Get the section title/name - try multiple approaches
        # Method 1: Look for h1 element (most common on Cornell Law)
        title_element = soup.find('h1')
        if not title_element:
            # Method 2: Look for h3 with section-head class
            title_element = soup.find('h3', class_='section-head')
        if not title_element:
            # Method 3: Look for any h2-h4 containing the section number
            for tag in ['h2', 'h3', 'h4']:
                title_element = soup.find(tag, string=re.compile(f'§\\s*{section}'))
                if title_element:
                    break

        if title_element:
            title_text = title_element.get_text(strip=True)
            # Extract the descriptive part after the section number and dash
            patterns = [
                r'§\s*\d+\s*[-–—]\s*(.+)',  # Various dash types
                r'§\s*\d+\s+(.+)',  # Space separator
                r'Section\s+\d+\s*[-–—]\s*(.+)'  # Alternative format
            ]

            for pattern in patterns:
                match = re.search(pattern, title_text)
                if match:
                    result["section_name"] = match.group(1).strip()
                    break

            if not result["section_name"] and '-' in title_text:
                # Simple split approach
                parts = title_text.split('-', 1)
                if len(parts) > 1:
                    result["section_name"] = parts[1].strip()

        # Get the main legal text - improved extraction
        content_text = ""

        # Method 1: Look for the field-items div (common in Cornell Law)
        field_items = soup.find('div', class_='field-items')
        if field_items:
            # Get all text, preserving structure
            for elem in field_items.find_all(text=True):
                text = elem.strip()
                if text and not any(skip in text.lower() for skip in ['prev|next', 'u.s. code toolbox']):
                    content_text += text + " "

        # Method 2: If field-items didn't work, look for content area
        if not content_text:
            content_area = soup.find('div', {'id': 'content'})
            if content_area:
                # Remove navigation elements
                for nav in content_area.find_all(['nav', 'header', 'footer']):
                    nav.decompose()

                # Extract text
                content_text = content_area.get_text()

        # Clean up and format the extracted text
        if content_text:
            # Remove excessive whitespace
            content_text = re.sub(r'\s+', ' ', content_text)

            # Find where the actual statute text begins (usually starts with "(a)")
            statute_start = re.search(r'\(a\)', content_text)
            if statute_start:
                content_text = content_text[statute_start.start():]

            # Remove trailing metadata (amendments, notes, etc.)
            # These usually start with specific markers
            end_markers = [
                '(Added',
                'Editorial Notes',
                'Statutory Notes',
                'Effective Date',
                'CFR Title',
                'prev|next'
            ]

            for marker in end_markers:
                marker_pos = content_text.find(marker)
                if marker_pos > 0:
                    content_text = content_text[:marker_pos]

            result["section_definition"] = content_text.strip()

        return result

    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return {"error": "404", "message": "Section not found"}
        return {"error": f"Error fetching content: {e}"}
    except requests.RequestException as e:
        return {"error": f"Error fetching content: {e}"}
    except Exception as e:
        return {"error": f"Error processing content: {e}"}


def save_usc_content_json(title, section, content_dict):
    """
    Save USC content to JSON file in the specified directory structure.

    Args:
        title (int): USC Title number
        section (int): USC Section number
        content_dict (dict): Dictionary with section_name and section_definition
    """
    # Create directory if it doesn't exist
    directory = "legal_definitions"
    os.makedirs(directory, exist_ok=True)

    # Create filename with .json extension
    filename = f"usc_{title}_{section}.json"
    filepath = os.path.join(directory, filename)

    # Add metadata to the content
    content_dict["title"] = title
    content_dict["section"] = section
    content_dict["source"] = f"https://www.law.cornell.edu/uscode/text/{title}/{section}"

    # Save content to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(content_dict, f, indent=2, ensure_ascii=False)

    print(f"Content saved to: {filepath}")
    return filepath


def main(title, section):
#     """Main function to handle command-line arguments."""
#     if len(sys.argv) != 3:
#         print("Usage: python fetch_usc.py <title> <section>")
#         print("Example: python fetch_usc.py 11 111")
#         sys.exit(1)

#     try:
#         title = int(sys.argv[1])
#         section = int(sys.argv[2])
#     except ValueError:
#         print("Error: Title and section must be integers")
#         sys.exit(1)

#     print(f"Fetching USC Title {title}, Section {section}...")

    # Fetch content
    content_dict = fetch_usc_content(title, section)

    if "error" in content_dict:
        # Skip 404 errors silently
        if content_dict.get("error") == "404":
            print(f"Skipping USC {title} § {section} (not found)")
            return
        # Print other errors
        print(content_dict["error"])
        return

    # Save content to JSON
    filepath = save_usc_content_json(title, section, content_dict)

    # Display results
    print(f"\nExtracted content:")
    print("-" * 50)
    print(f"Section name: {content_dict['section_name']}")
    print(f"\nSection definition preview (first 500 chars):")
    definition = content_dict['section_definition']
    print(definition[:500] + "..." if len(definition) > 500 else definition)


if __name__ == "__main__":
    # List of config files to process
    config_files = ['usc_config_2.json', 'usc_config_3.json',
                    'usc_config_4.json', 'usc_config_5.json', 'usc_config_6.json']

    for config_file in config_files:
        print(f"\n{'#'*60}\nProcessing: {config_file}\n{'#'*60}")

        # Load USC codes from config file
        with open(config_file, 'r') as f:
            config = json.load(f)
        titles = config['titles']
        sections = config['sections']

        for title, section in zip(titles, sections):
            main(title, section)