"""Simple script to quickly get the observation of a page"""

import json
import re
import time
from typing import Dict, Optional, Tuple, Type, Union, cast

import pytest
from playwright.sync_api import Page, expect
from PIL import Image

from browser_env import (
    ScriptBrowserEnv,
    create_id_based_action,
    create_key_press_action,
    create_playwright_action,
    create_scroll_action,
)
from browser_env.env_config import *

HEADLESS = False


def gen_tmp_storage_state() -> None:
    with open(f"scripts/tmp_storage_state.json", "w") as f:
        json.dump({"storage_state": ".auth/shopping_admin_state.json"}, f)


def get_observation(
    observation_type: str, current_viewport_only: bool, URL: str
) -> None:
    env = ScriptBrowserEnv(
        observation_type=observation_type,
        current_viewport_only=current_viewport_only,
        headless=HEADLESS,
        sleep_after_execution=5.0,
    )
    env.reset(options={"config_file": f"scripts/tmp_storage_state.json"})
    s = f"""page.goto("{URL}")\n"""
    action_seq = s.split("\n")

    action = f"""page.goto("{URL}")"""
    action = action.strip()
    obs, success, _, _, info = env.step(create_playwright_action(action))
    print(obs["text"])

    image_array = obs["image"]
    # image_array here is the np array of the screenshot, convert it back to image and save it
    image = Image.fromarray(image_array)

    # get the file name from the URL, simplify the URL to avoid "/" in the file name
    website_name = re.sub(r"https?://", "", URL)
    website_name = website_name.replace("/", "")
    print(f"Saving the image to data/images/{website_name}.png")
    image.save(f"data/images/{website_name}.png")

    # save the text observation to a file
    with open(f"data/text/{website_name}.txt", "w") as f:
        f.write(obs["text"])


if __name__ == "__main__":
    gen_tmp_storage_state()
    obs_type = "accessibility_tree"
    current_viewport_only = False
    URL = "https://iheartdogs.com/"
    get_observation(obs_type, current_viewport_only, URL)