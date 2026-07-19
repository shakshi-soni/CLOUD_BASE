from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

try:
    driver.get("https://cloudbase-envp6xbcxfvjnybalsb33h.streamlit.app/")
    wait = WebDriverWait(driver, 15)
    try:
        button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'get this app back up')]"))
        )
        button.click()
        print("App was asleep — clicked wake button")
    except TimeoutException:
        print("App already awake, button not found")
finally:
    driver.quit()
