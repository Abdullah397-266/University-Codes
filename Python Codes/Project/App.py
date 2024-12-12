from flask import Flask, render_template, request, send_file
import json
import os
from web_scraper import get_data_from_page

app = Flask(__name__)


# Route to display the data in Bootstrap cards
@app.route("/")
def index():
    # Load the JSON file
    file_path = os.path.join(os.path.dirname(__file__), "categorized_data.json")
    # file_path = 'categorized_data.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    return render_template("index.html", comments=data)

@app.route("/scraper")
def scraper():
    return render_template("scraper.html")

@app.route("/scrape", methods=["POST"])
def scrape():
    # Get the URL from the user input
    url = request.form.get("url")
    url = url + "&page=2"

    if not url:
        return "Error: URL is required"

    try:
        # Call the scraping function
        df = get_data_from_page(url)

        # Save the data to a CSV file
        csv_path = "scraped_data.csv"
        df.to_csv(csv_path, index=False)

        print("Scraping complete")
        # Provide the file for download
        return send_file(
            csv_path,
            as_attachment=True,
            download_name="scraped_data.csv",
            mimetype="text/csv"
        )
    
    except Exception as e:
        return f"Error occurred: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)
