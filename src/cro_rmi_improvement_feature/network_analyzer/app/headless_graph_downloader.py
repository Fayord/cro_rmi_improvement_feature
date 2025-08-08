from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from PIL import Image
import time
import os
import base64
from io import BytesIO

# from pyvirtualdisplay import Display

# display = Display(visible=0, size=(1920, 1080), use_xauth=True)
# display.start()

dir_path = os.path.dirname(os.path.abspath(__file__))


def download_graph_headless(url, download_dir=f"{dir_path}/downloads"):
    # Setup Chrome options for headless browsing
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    # chrome_options.add_argument("--disable-dev-shm-usage")

    # Set download directory
    prefs = {"download.default_directory": os.path.abspath(download_dir)}
    chrome_options.add_experimental_option("prefs", prefs)

    # Initialize WebDriver
    # Make sure you have chromedriver installed and its path is in your PATH environment variable,
    # or specify the path to chromedriver executable.
    # service = Service(executable_path="/usr/local/bin/chromedriver")
    # driver = webdriver.Chrome(options=chrome_options, service=service)
    print("Installing ChromeDriver...")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=chrome_options
    )
    # driver = webdriver.Remote(
    #     command_executor="http://172.16.100.50:4444/wd/hub",
    #     options=chrome_options,
    # )

    print("ChromeDriver installed.")

    try:
        print(f" URL: {url}")
        driver.get(url)
        print(f"Opened URL: {url}")

        # Wait for the page to load and the button to be clickable
        # set driver window size to max
        driver.maximize_window()
        print("Window size set to max")
        time.sleep(10)
        # scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/4);")
        time.sleep(10)  # Adjust this sleep time if needed
        # save screenshot
        # click on div id tab-png
        tab_png = driver.find_element(By.ID, "tab-png")
        tab_png.click()
        time.sleep(2)
        driver.save_screenshot(os.path.join(dir_path, "screenshot.png"))
        # Find the "as svg" button and click it
        # svg_button = driver.find_element(By.ID, "btn-get-svg")
        # svg_button.click()
        print("Clicked 'as svg' button.")

        # screenshot the page
        # div with id="image-text"
        image_text = driver.find_element(By.ID, "image-text")
        # get string data from image_text
        image_text_str = image_text.text
        # print(f"Image text: {image_text_str}")
        # convert to png and save
        image_text_str = image_text_str.replace("data:image/png;base64,", "")
        image = Image.open(BytesIO(base64.b64decode(image_text_str)))
        image.save(os.path.join(dir_path, "cyto.png"))
        # save to file

        print("Screenshot saved.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()
        print("Browser closed.")


if __name__ == "__main__":
    app_url = "http://172.16.100.50/plot_network/"
    # app_url = "http://172.16.100.50:8050"
    # app_url = "http://127.0.0.1:8050"
    download_folder = "downloaded_graphs"

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
        print(f"Created download directory: {download_folder}")

    download_graph_headless(app_url, download_folder)
    print(f"Graph download initiated. Check the '{download_folder}' directory.")
