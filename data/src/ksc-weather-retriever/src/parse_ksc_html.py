# -- Imports --
import json
from bs4 import BeautifulSoup

# -- Functions --
def parse_field_mill_html(input_file, output_file):
    # Read the HTML file
    with open(input_file, 'r') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    result = {}

    for tr in soup.find_all('tr', id=lambda x: x and x.startswith('listviewtr-')):
        filename_td = tr.find('td', id='filename')
        if filename_td:
            a_tag = filename_td.find('a')
            if a_tag:
                href = a_tag['href']
                filename = a_tag.find('span').text.strip()
                # Extract last 6 digits before .zip and insert a hyphen
                folder_name = filename[-10:-4]
                folder_name = f"{folder_name[:4]}-{folder_name[4:]}"
                result[folder_name] = href

    # Convert to JSON
    json_result = json.dumps(result, indent=2)

    # Save to a file
    with open(output_file, 'w') as f:
        f.write(json_result)

    return json_result

# -- Execution --
if __name__ == "__main__":
    input_file = 'data/field_mill.html'
    output_file = 'data/field_mill.json'
    json_output = parse_field_mill_html(input_file, output_file)
    print(json_output)