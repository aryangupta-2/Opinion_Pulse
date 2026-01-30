from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import InvalidSessionIdException, WebDriverException, TimeoutException
import re
import time
import csv
import os

# =========================================================
# 🔹 CHANGE ONLY THIS LINE FOR A NEW PRODUCT
# =========================================================
PRODUCT_URL = "https://www.amazon.in/dp/B0G1BS9S4Q"
# =========================================================

# -------------------- FUNCTION --------------------
def get_amazon_review_url(product_url: str) -> str | None:
    """
    Takes an Amazon product URL and returns its review page URL
    """
    match = re.search(r"/dp/([A-Z0-9]{10})|/product/([A-Z0-9]{10})", product_url)
    if not match:
        return None

    asin = match.group(1) or match.group(2)
    return f"https://www.amazon.in/product-reviews/{asin}"

# -------------------- BUILD REVIEW URL --------------------
review_url = get_amazon_review_url(PRODUCT_URL)
if not review_url:
    raise ValueError("❌ Invalid Amazon product URL")

# -------------------- CSV SETUP --------------------
CSV_FILE = "amazon_reviews.csv"
file_exists = os.path.isfile(CSV_FILE)

# -------------------- SELENIUM SETUP --------------------
options = Options()
options.add_argument(r"--user-data-dir=C:\Users\Param\selenium_profile")
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 20)

try:
    # -------------------- LOGIN STEP --------------------
    driver.get("https://www.amazon.in")
    print("👉 Please log in manually (OTP / captcha if asked).")
    print("👉 DO NOT close the browser.")

    login_timeout = time.time() + 300  # 5 minutes
    while True:
        if time.time() > login_timeout:
            raise TimeoutError("Login not completed in time.")

        try:
            if driver.find_elements(By.ID, "nav-link-accountList"):
                print("✅ Login detected")
                break
        except (InvalidSessionIdException, WebDriverException):
            print("❌ Browser closed. Restart script.")
            driver.quit()
            exit()

        time.sleep(2)

    # -------------------- OPEN CSV --------------------
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header once
        if not file_exists:
            writer.writerow(["page", "review_text", "rating"])

        # -------------------- ITERATE REVIEW PAGES --------------------
        MAX_PAGES = 5  # change if needed

        for page in range(1, MAX_PAGES + 1):
            print(f"\n📄 Scraping review page {page}...")

            driver.get(f"{review_url}?pageNumber={page}")

            try:
                reviews = wait.until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, '[data-hook="review"]')
                    )
                )
                print(f"✅ Page {page}: {len(reviews)} reviews found")

            except TimeoutException:
                print(f"⚠️ Page {page}: No reviews found (possibly last page)")
                break

            # -------------------- SAVE REVIEWS --------------------
            for review in reviews:
                try:
                    text = review.find_element(
                        By.CSS_SELECTOR, '[data-hook="review-body"]'
                    ).text.strip()

                    rating = review.find_element(
                        By.CSS_SELECTOR, '[data-hook="review-star-rating"]'
                    ).text.strip()

                    writer.writerow([page, text, rating])

                except Exception:
                    continue  # skip problematic reviews

            time.sleep(2)  # polite delay

finally:
    driver.quit()
    print("\n✅ Scraping completed. Data saved to amazon_reviews.csv")
