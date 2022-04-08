import csv
import sys
from pathlib import Path

import requests
from lxml import etree
from tqdm import tqdm, trange

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_url():
    # The listing url = url_base + page_num
    url_base = "https://www.srx.com.sg/singapore-property-listings/condo-for-sale?page="
    url_head = "https://www.srx.com.sg"

    for page in trange(2296):
        # crawl the listing webpage and determine whether it is successful
        r = requests.get(url_base + str(page))
        if r.status_code != 200:
            continue

        # parse the webpage
        # the urls lie in two parts, so we use two different XPath to get urls separately
        html = etree.HTML(r.text)
        result = html.xpath(
            "/html/body/div[6]/div/div[2]/div[2]/div/div/a/@href"
        ) + html.xpath("/html/body/div[8]/div[1]/div/div/div/div/a/@href")

        # modify the result: add the url_head to make it complete and add the "\n" for recording
        result = list(map(lambda x: url_head + x + "\n", result))
        with open(str(ROOT) + "/Data/urls.txt", "a") as f:
            f.writelines(result)


def parse_info():

    with open(str(ROOT) + "/Data/urls.txt", "r") as f:

        head = True

        for url in tqdm(f):
            # crawl the conda detail webpage and determine whether it is successful
            r = requests.get(url[:-1])
            if r.status_code != 200:
                continue

            html = etree.HTML(r.text)
            info = dict(
                zip(
                    [
                        "listing_id",
                        "name",
                        "street",
                        "type",
                        "model",
                        "market_segment",
                        "type_of_area",
                        "bedrooms",
                        "bathrooms",
                        "district",
                        "region",
                        "planning_area",
                        "subszone",
                        "lat",
                        "lng",
                        "tenure",
                        "built_year",
                        "no_of_units",
                        "area_size",
                        "eco_category",
                        "accessibility",
                        "date_listed",
                        "price",
                    ],
                    [""] * 23,
                )
            )

            info["listing_id"] = int(url.split("/")[4])
            # the items of property information are not fixed,
            # so we have to crawl all the information and determine which attribution is included
            basic = html.xpath(
                '/html/body/div[7]/div/div/div[2]/div/div/div/descendant::*[@itemprop="name" or @itemprop="value"]/text()'
            )
            # modify the attribution value according to different requirements
            for i in range(0, len(basic), 2):
                if basic[i] == "Address":
                    info["street"] = basic[i + 1].lower()
                elif basic[i] == "Property Name":
                    info["name"] = basic[i + 1].lower()
                elif basic[i] == "Property Type":
                    info["type"] = basic[i + 1].lower()
                elif basic[i] == "Model":
                    info["model"] = basic[i + 1].lower()
                elif basic[i] == "Bedrooms":
                    info["bedrooms"] = basic[i + 1]
                elif basic[i] == "Bathrooms":
                    info["bathrooms"] = float(basic[i + 1])
                elif basic[i] == "Tenure":
                    info["tenure"] = basic[i + 1].lower()
                elif basic[i] == "Built Year":
                    info["built_year"] = float(basic[i + 1])
                elif basic[i] == "No. of Units":
                    info["no_of_units"] = float(basic[i + 1])
                elif basic[i] == "District":
                    info["district"] = int(basic[i + 1][1:3].rstrip())
                elif basic[i] == "Size":
                    info["area_size"] = float(
                        basic[i + 1].split(" ")[0].replace(",", "")
                    )
                elif basic[i] == "Date Listed":
                    info["date_listed"] = basic[i + 1]

            info["lat"] = float(html.xpath('//*[@id="listing-latitude"]/@value')[0])
            info["lng"] = float(html.xpath('//*[@id="listing-longitude"]/@value')[0])

            price = html.xpath(
                '/html/body/div[6]/div/div/div/div/div/div[1]/div[2]/div[1]/div/div[1]/div[2]/div/span[@itemprop="price"]/text()'
            )
            if price:
                info["price"] = float(price[0].replace(",", ""))
            else:
                continue

            with open(str(ROOT) + "/Data/crawler.csv", "a") as csvfile:
                writer = csv.writer(csvfile)

                if head:
                    writer.writerow(list(info.keys()))
                    head = False

                writer.writerow(list(info.values()))


if __name__ == "__main__":
    # The crawler process is divided into two parts:
    # Firstly, we need to crawl every listing page to get all urls of the conda detail
    # Secondly, we crawl and parse every conda webpage to get the information we need
    # parse_url()
    parse_info()
