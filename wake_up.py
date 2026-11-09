import sys
from playwright.sync_api import sync_playwright

URL = "https://cloudbase-envp6xbcxfvjnybalsb33h.streamlit.app/"
BUTTON = "Yes, get this app back up!"

def attempt(browser):
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(5000)
    btn = page.get_by_text(BUTTON)
    if btn.count() > 0:
        btn.first.click()
        print("App was asleep, clicked wake button")
        page.wait_for_timeout(90000)
    else:
        print("App already awake")
        page.wait_for_timeout(30000)
    still_asleep = page.get_by_text(BUTTON).count() > 0
    page.close()
    return not still_asleep

with sync_playwright() as p:
    browser = p.chromium.launch()
    for i in range(3):
        try:
            if attempt(browser):
                print("OK on attempt", i + 1)
                browser.close()
                sys.exit(0)
        except Exception as e:
            print("Attempt", i + 1, "failed:", e)
    browser.close()

print("FAILED: app still asleep or unreachable")
sys.exit(1)
