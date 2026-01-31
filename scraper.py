from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import re
import time
import csv

# -------------------- INPUT PRODUCT URL --------------------
PRODUCT_URL = input("Enter Amazon product URL: ").strip()

def get_asin(product_url):
    match = re.search(r"/dp/([A-Z0-9]{10})|/product/([A-Z0-9]{10})", product_url)
    if not match:
        raise ValueError("❌ Invalid Amazon product URL")
    return match.group(1) or match.group(2)

ASIN = get_asin(PRODUCT_URL)

# -------------------- REVIEW PAGE (PAGE 1) --------------------
REVIEW_START_URL = (
    f"https://www.amazon.in/product-reviews/{ASIN}/"
    "ref=cm_cr_dp_d_show_all_btm"
    "?ie=UTF8&reviewerType=all_reviews"
)

# -------------------- SELENIUM SETUP --------------------
options = Options()
options.add_argument(r"--user-data-dir=C:\Users\Param\selenium_profile")
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 20)

try:
    # -------------------- LOGIN --------------------
    driver.get("https://www.amazon.in")
    print("👉 Please log in manually if required.")
    print("👉 DO NOT close the browser.")

    while True:
        if driver.find_elements(By.ID, "nav-link-accountList"):
            print("✅ Login detected")
            break
        time.sleep(2)

    # -------------------- OPEN REVIEW PAGE --------------------
    driver.get(REVIEW_START_URL)

    # -------------------- CSV FILE --------------------
    with open("amazon_reviews.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review_title", "review_text", "review_date"])

        page_number = 1

        while True:
            print(f"📄 Scraping page {page_number}")

            try:
                reviews = wait.until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, '[data-hook="review"]')
                    )
                )
            except TimeoutException:
                print("⚠️ No reviews found. Stopping.")
                break

            # -------- SCRAPE CURRENT PAGE --------
            for review in reviews:
                try:
                    title = review.find_element(
                        By.CSS_SELECTOR, '[data-hook="review-title"]'
                    ).text.strip()

                    text = review.find_element(
                        By.CSS_SELECTOR, '[data-hook="review-body"]'
                    ).text.strip()

                    raw_date = review.find_element(
                        By.CSS_SELECTOR, '[data-hook="review-date"]'
                    ).text.strip()

                    # 🔹 CLEAN DATE
                    date = raw_date.replace("Reviewed in India on ", "").strip()

                    writer.writerow([title, text, date])

                except Exception:
                    continue

            # -------- CLICK NEXT BUTTON --------
            try:
                next_li = driver.find_element(By.CSS_SELECTOR, "li.a-last")

                if "a-disabled" in next_li.get_attribute("class"):
                    print("✅ Last page reached.")
                    break

                next_btn = next_li.find_element(By.TAG_NAME, "a")
                driver.execute_script("arguments[0].scrollIntoView(true);", next_btn)
                time.sleep(1)
                next_btn.click()

                page_number += 1
                time.sleep(2)

            except (StaleElementReferenceException, TimeoutException):
                print("⚠️ Pagination ended.")
                break

finally:
    driver.quit()
    print("\n✅ ALL REVIEWS SAVED TO amazon_reviews.csv")



