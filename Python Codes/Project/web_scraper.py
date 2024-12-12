from requests_html import HTMLSession
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


def get_parser(soup, review_list, overall_rating_list, date_list, name):

    # name = soup.find("h1", {"class": "E-vwXONV9nc-"}).get_text(
    #     separator=" ", strip=True
    # )

    reviews = soup.find_all(
        "span",
        {
            "data-testid": "wrapper-tag",
            "class": "l9bbXUdC9v0- ZatlKKd1hyc- ukvN6yaH1Ds-",
        },
    )

    ratings = soup.find_all("span", {"class": "-y00OllFiMo-"})

    dates = soup.find_all("p", {"class": "iLkEeQbexGs-"})

    for review in reviews:
        review = review.get_text(separator=" ", strip=True)
        review = review.replace("\n", " ").replace(",", ".")
        # print(review)
        review_list.append(review)

    for i in range(0, len(ratings), 4):
        ratingArr = []
        ratingArr.append(ratings[i].get_text(separator=" ", strip=True))
        ratingArr.append(ratings[i + 1].get_text(separator=" ", strip=True))
        ratingArr.append(ratings[i + 2].get_text(separator=" ", strip=True))
        ratingArr.append(ratings[i + 3].get_text(separator=" ", strip=True))
        # print(ratingArr)
        overall_rating_list.append(ratingArr)

    for date in dates:
        date = date.get_text(separator=" ", strip=True)
        if "Dined on" in date:
            date = date[9:]
        else:
            date = date[6:]
        date = date.replace(",", "")
        # print(date)
        date_list.append(date)


def get_reviews_from_all_pages(first_page_url, max_page):
    page = 1
    review_list = []
    overall_rating_list = []
    date_list = []
    res_name = ""
    p1, p2 = first_page_url.split("page=")

    while page != (max_page + 1):
        url = p1 + "page=" + str(page) + p2[1:]
        session = HTMLSession()
        response = session.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        get_parser(soup, review_list, overall_rating_list, date_list, res_name)
        page += 1

    print("Resturant Name:", res_name)
    overall_rating_list_np = np.array(overall_rating_list)

    df = pd.DataFrame(
        {
            "Reviews": review_list,
            "Ratings": list(overall_rating_list_np[:, 0]),
            "Food": list(overall_rating_list_np[:, 1]),
            "Service": list(overall_rating_list_np[:, 2]),
            "Ambience": list(overall_rating_list_np[:, 3]),
            "Dates": date_list,
        }
    )
    return df


def get_data_from_page(url):
    # 10 reviews per page
    max_page = 1600 // 10
    df = get_reviews_from_all_pages(url, max_page)
    return df


if __name__ == "__main__":

    # url of page 2
    url = "https://www.opentable.com/r/bacchanal-buffet-caesars-palace-las-vegas?originId=c09c1481-a8ad-44d6-9394-a07339e3fd55&corrid=c09c1481-a8ad-44d6-9394-a07339e3fd55&avt=eyJ2IjoyLCJtIjoxLCJwIjowLCJzIjowLCJuIjowfQ&page=2"

    df = get_data_from_page(url)
    # df.to_csv("./newData.csv", index=False)
