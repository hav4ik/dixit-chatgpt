import logging
import telebot
import yaml
from PIL import Image
import os
import imagehash
import cv2
from datetime import datetime
import glob
from utils import (
    get_cards_from_image,
    download_image_from_message_to_cache,
    get_game_state_path,
    reset_game_state,
)
import omegaconf


def instantiate_dixitbot():
    secrets = omegaconf.OmegaConf.load('secrets.yaml')
    bot = telebot.TeleBot(token=secrets.telegram_token, parse_mode=None)

    CONFIG_PATH = "config.yaml"
    IMAGE_FOLDER = ".cache/images/"
    GAME_STATE_FOLDER = ".cache/game_state/"
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    os.makedirs(GAME_STATE_FOLDER, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    def describe_images(det, questions_to_blip2=0):
        hashes, cards_images, card_images_paths = det["hashes"], det["cards"], det["cards_paths"],
        ret_dict = dict()
        for idx, (card_hash, card_image, card_image_path) in enumerate(zip(hashes, cards_images, card_images_paths)):
            ret_dict.update({
                card_hash: {
                    "image_path": card_image_path,
                    "text2img_description": "none",
                    "chatgpt_blip2_dialog": "none",
                    "chatgpt_interpretation": "none",
                },
            })
        return ret_dict


    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        logging.log(logging.INFO, f"Received [help] request from {message.from_user.username}.")    
        bot.reply_to(message, "Howdy, how are you doing?")
        get_game_state_path(bot, message, game_state_folder=GAME_STATE_FOLDER)


    @bot.message_handler(commands=['reset'])
    def send_welcome(message):
        logging.log(logging.INFO, f"Received [reset] request from {message.from_user.username}.")    
        reset_game_state(get_game_state_path(bot, message, game_state_folder=GAME_STATE_FOLDER))
        bot.reply_to(message, "Game is reset to initial state.")


    @bot.message_handler(commands=['nuke_cache'])
    def nuke_cache(message):
        logging.log(logging.INFO, f"Received [nuke_cache] request from {message.from_user.username}.")    
        reset_game_state(get_game_state_path(bot, message, game_state_folder=GAME_STATE_FOLDER))
        files = glob.glob(os.path.join(IMAGE_FOLDER, "*"))
        for f in files:
            os.remove(f)
        bot.reply_to(message, "All cache is nuked and game state is reset to initial state.")


    @bot.message_handler(commands=['status'])
    def send_welcome(message):
        logging.log(logging.INFO, f"Received [status] request from {message.from_user.username}.")


    @bot.message_handler(func=lambda m: str(m.caption).startswith("/add"), content_types=['photo'])
    def add_cards_to_hand(message):
        print(message)
        logging.log(logging.INFO, f"Received [add images] request from {message.from_user.username}")

        # Download the image in message
        image_cache_path = download_image_from_message_to_cache(bot, message, image_folder=IMAGE_FOLDER)

        # Get config
        config = omegaconf.OmegaConf.load(CONFIG_PATH)

        # Get game state
        game_state_path = get_game_state_path(bot, message, game_state_folder=GAME_STATE_FOLDER)
        with open(game_state_path, 'r') as fd:
            game_state = yaml.safe_load(fd)

        # Detect cards
        start_detection = datetime.now()
        det = get_cards_from_image(image_cache_path, config=config)
        already_added_indices = [i for i, h in enumerate(det["hashes"]) if h in game_state["my_cards"].keys()]

        # Write logs for detected cards
        detection_time = (datetime.now() - start_detection).total_seconds()
        debug_message = f"Found {len(det['cards'])} cards in {detection_time:0.1f} seconds."
        if len(already_added_indices) > 0:
            debug_message += f" Cards {str(already_added_indices)} were already added."
        logging.log(logging.INFO, debug_message)
        bot.send_photo(message.chat.id, open(det["debug_img_path"], 'rb'),
                    reply_to_message_id=message.message_id, caption=debug_message)

        # Generate descriptions
        descriptions = describe_images(det, questions_to_blip2=config.clue_generation.questions_to_blip2)
        for idx, (img_hash, img_info) in enumerate(descriptions.items()):
            game_state["my_cards"].update(descriptions)

        # Save game state (i.e. our hand)
        with open(game_state_path, 'w') as fd:
            yaml.dump(game_state, fd, default_flow_style=False)

    # return the dixit bot
    return bot


if __name__ == "__main__":
    bot = instantiate_dixitbot()
    bot.infinity_polling()
