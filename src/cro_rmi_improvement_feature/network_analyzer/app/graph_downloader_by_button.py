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
from pynput.mouse import Controller
from pynput.mouse import Button

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
    chrome_options.add_argument("--disable-features=InsecureDownloadWarnings")

    # You can also add other preferences for a better download experience
    prefs = {
        "download.default_directory": os.path.abspath(download_dir),
        "download.prompt_for_download": False,
        "safeBrowse.enabled": False,  # It's still good to include this just in case
    }
    # # Set download directory
    # prefs = {
    #     "download.default_directory": os.path.abspath(download_dir),
    #     "safeBrowse.enabled": False,  # It's still good to include this just in case
    # }
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
        # scroll to bottom of website
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # click on tab-png
        tab_png = driver.find_element(By.ID, "tab-png")
        tab_png.click()
        time.sleep(1)
        # click on button id btn-get-svg

        btn_get_svg = driver.find_element(By.ID, "btn-get-svg")
        btn_get_svg.click()
        time.sleep(10)
        # loop print mouse position for 20 sec every sec
        mouse = Controller()
        # for i in range(20):
        #     x, y = mouse.position
        #     print(f"Mouse position after {i+1} seconds: X={x}, Y={y}")
        #     time.sleep(1)
        # move_mouse_to_position , 1344, 99
        # get current mouse position
        x, y = mouse.position
        print(f"Current mouse position: X={x}, Y={y}")
        target_x = 1344
        target_y = 99
        mouse.move(target_x - x, target_y - y)
        time.sleep(1)
        # click
        mouse.click(Button.left, 1)
        time.sleep(1)
        # move_mouse_to_position , 1172,184
        x, y = mouse.position
        print(f"Current mouse position: X={x}, Y={y}")
        target_x = 1172
        target_y = 184
        mouse.move(target_x - x, target_y - y)
        time.sleep(1)
        # click
        mouse.click(Button.left, 1)
        time.sleep(1)
        # move_mouse_to_position , 1047,275
        x, y = mouse.position
        print(f"Current mouse position: X={x}, Y={y}")
        target_x = 1047
        target_y = 275
        mouse.move(target_x - x, target_y - y)
        time.sleep(1)
        # click
        mouse.click(Button.left, 1)
        time.sleep(1)

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
