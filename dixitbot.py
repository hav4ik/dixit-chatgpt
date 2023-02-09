import logging
import telebot
import yaml
from PIL import Image
import os
import io
import imagehash
import cv2
import numpy as np
import uuid
from datetime import datetime
import glob
from utils import (
    get_cards_from_image,
    download_image_from_message_to_cache,
    get_game_state_path,
    reset_game_state,
    build_image_grid,
)
import omegaconf


def generate_description_and_clues(det, config):
    hashes, cards_images, card_images_paths = det["hashes"], det["cards"], det["cards_paths"],

    # Mock outputs
    ret_dict = dict()
    for idx, (hash, image, image_path) in enumerate(zip(hashes, cards_images, card_images_paths)):
        ret_dict.update({
            hash: {
                "image_path": image_path,
                "captions": "none",
                "pre_qna_interpretation": "none",
                "qna_session": "none",
                "interpretation": "none",
                "association": "none",
                "clue": "none",
            },
        })
    return ret_dict


def choose_best_match_image_for_clue(clue, card_infos, config):
    images = [np.array(Image.open(card_info['image_path'])) for _, card_info in card_infos.items()]

    # Mock outputs
    ret_dict = dict({
        'per_image_explanations': [],
        'final_answer': "none",
    })
    for idx, (card_hash, card_info) in enumerate(card_infos.items()):
        ret_dict['per_image_explanations'].append({
            'idx': idx,
            'image_path': card_info["image_path"],
            'captions': "none",
            "pre_qna_interpretation": "none",
            "qna_session": "none",
            "interpretation": "none",
            "clue_relation": "none",
        })
    return ret_dict


def instantiate_dixitbot():
    secrets = omegaconf.OmegaConf.load('secrets.yaml')
    bot = telebot.TeleBot(token=secrets.telegram_token, parse_mode=None)

    CONFIG_PATH = "config.yaml"
    IMAGE_FOLDER = ".cache/images/"
    GAME_STATE_FOLDER = ".cache/game_state/"
    OUTPUT_LOGS = ".cache/output_logs/"
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    os.makedirs(GAME_STATE_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_LOGS, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        logging.log(logging.INFO, f"Received [help] request from {message.from_user.username}.")    
        bot.reply_to(message, "Howdy, how are you doing?")
        get_game_state_path(bot, message, game_state_folder=GAME_STATE_FOLDER)

    @bot.message_handler(commands=['reset'])
    def reset_state(message):
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

    @bot.message_handler(commands=['hand', 'status'])
    def show_short_hand_clues(message):
        logging.log(logging.INFO, f"Received [hand] request from {message.from_user.username}.")

        # Get game state
        game_state_path = get_game_state_path(bot, message, game_state_folder=GAME_STATE_FOLDER)
        with open(game_state_path, 'r') as fd:
            game_state = yaml.safe_load(fd)

        # If no cards in hand, return
        if game_state["my_cards"] is None or len(game_state["my_cards"]) == 0:
            bot.reply_to(message, "Your hand is empty, there's no card in your hand.")
            return

        # Build a grid of images
        image_paths = []
        for card_hash, card_info in game_state["my_cards"].items():
            image_paths.append(card_info["image_path"])
        grid = build_image_grid(image_paths)
        grid_bytes_stream = io.BytesIO()
        grid.save(grid_bytes_stream, 'jpeg')

        # Display only "interpretation", "association", and "clue" for each card
        response_text = ""
        for image_idx, (card_hash, card_info) in enumerate(game_state["my_cards"].items()):
            response_text += f"Card {image_idx}: {card_info['clue']}\n"
        response_text += "\nTo get detailed explanation to the clues, please use command /hand_detailed."

        # Send message
        bot.send_photo(message.chat.id, grid_bytes_stream.getvalue(), caption=response_text,
                       reply_to_message_id=message.message_id)

    @bot.message_handler(commands=['hand_detailed', 'status_detailed'])
    def show_detailed_hand_clues(message):
        logging.log(logging.INFO, f"Received [hand] request from {message.from_user.username}.")

        # Get game state
        game_state_path = get_game_state_path(bot, message, game_state_folder=GAME_STATE_FOLDER)
        with open(game_state_path, 'r') as fd:
            game_state = yaml.safe_load(fd)

        # If no cards in hand, return
        if game_state["my_cards"] is None or len(game_state["my_cards"]) == 0:
            bot.reply_to(message, "Your hand is empty, there's no card in your hand.")
            return

        # Build a grid of images
        for image_idx, (card_hash, card_info) in enumerate(game_state["my_cards"].items()):
            cap_text = f"Card {image_idx}: {card_info['clue']}\n\n"
            cap_text += f"captions:\n{card_info['captions']}\n\n"
            cap_text += f"pre_qna_interpretation:\n{card_info['pre_qna_interpretation']}\n\n"
            cap_text += f"qna_session:\n{card_info['qna_session']}\n\n"
            cap_text += f"interpretation:\n{card_info['interpretation']}\n\n"
            cap_text += f"association:\n{card_info['association']}"
            bot.send_photo(message.chat.id, open(card_info["image_path"], 'rb'), caption=cap_text)

    @bot.message_handler(func=lambda m: str(m.caption).startswith("/add"), content_types=['photo'])
    def add_cards_to_hand(message):
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
        debug_message = (
            f"Found {len(det['cards'])} cards in {detection_time:0.1f} seconds. "
            "Generating description and clues...")
        if len(already_added_indices) > 0:
            debug_message += f" Cards {str(already_added_indices)} were already added."

        logging.log(logging.INFO, debug_message)
        bot.send_photo(
            message.chat.id, open(det["debug_img_path"], 'rb'),
            reply_to_message_id=message.message_id, caption=debug_message)

        # Generate descriptions and clues
        clue_generation_start = datetime.now()
        descriptions = generate_description_and_clues(det, config=config)
        for idx, (img_hash, img_info) in enumerate(descriptions.items()):
            game_state["my_cards"].update(descriptions)

        # Dump full logs
        event_log_id = str(uuid.uuid4().hex)
        output_logs_path = os.path.join(OUTPUT_LOGS, f"{event_log_id}.yaml")
        with open(output_logs_path, 'w') as fd:
            yaml.dump(descriptions, default_flow_style=False, sort_keys=False)

        # Save game state (i.e. our hand)
        with open(game_state_path, 'w') as fd:
            yaml.dump(game_state, fd, default_flow_style=False, sort_keys=False)
    
        clue_generation_time = (datetime.now() - clue_generation_start).total_seconds()
        bot.reply_to(message, (
            f"Generated descriptions in {clue_generation_time} seconds. "
            f"Cards in hand: {len(game_state['my_cards'])} "
            "You can see your hand using command /hand." 
            f"Full logs can be displayed using command /log_{event_log_id}"))
        
    @bot.message_handler(commands=['del'])
    def remove_card_from_hand(message):
        logging.log(logging.INFO, f"Received [remove card from hand] request from {message.from_user.username}")
        cards_to_delete = [int(x.strip()) for x in message.text.replace('/del', '').split(',')]

        # Get game state
        game_state_path = get_game_state_path(bot, message, game_state_folder=GAME_STATE_FOLDER)
        with open(game_state_path, 'r') as fd:
            game_state = yaml.safe_load(fd)
    
        # Remove cards from game state
        retained_cards = dict()
        for card_idx, (card_hash, card_info) in enumerate(game_state["my_cards"].items()):
            if card_idx not in cards_to_delete:
                retained_cards[card_hash] = card_info
        game_state["my_cards"] = retained_cards

        # Save game state (i.e. our hand)
        with open(game_state_path, 'w') as fd:
            yaml.dump(game_state, fd, default_flow_style=False, sort_keys=False)
        bot.reply_to(message, "Done removing cards. You can see your new hand with command /hand.")

    @bot.message_handler(func=lambda m: str(m.text).startswith("/log"))
    def show_event(message):
        event_id = message.text[len('/log_'):].strip()
        logging.log(logging.INFO, f"Received [show event] request from {message.from_user.username} for event {event_id}")
        event_log_path = os.path.join(OUTPUT_LOGS, f"{event_id}.yaml")
        with open(event_log_path, 'r') as fd:
            event_log_text = fd.read()
        bot.reply_to(message, event_log_text)

    @bot.message_handler(commands=['get'])
    def guess_card_from_clue_from_hand(message):
        logging.log(logging.INFO, f"Received [get card from hand by clue] request from {message.from_user.username}")
        clue = message.text[len('/get'):].strip()

        # Get config
        config = omegaconf.OmegaConf.load(CONFIG_PATH)

        # Get game state
        game_state_path = get_game_state_path(bot, message, game_state_folder=GAME_STATE_FOLDER)
        with open(game_state_path, 'r') as fd:
            game_state = yaml.safe_load(fd)

        # Generate descriptions and clues
        guess_image_start = datetime.now()
        results = choose_best_match_image_for_clue(clue, game_state["my_cards"], config=config)
        
        # Dump full logs
        event_log_id = str(uuid.uuid4().hex)
        output_logs_path = os.path.join(OUTPUT_LOGS, f"{event_log_id}.yaml")
        with open(output_logs_path, 'w') as fd:
            yaml.dump(results, fd)

        # Build a grid of images
        image_paths = []
        for card_hash, card_info in game_state["my_cards"].items():
            image_paths.append(card_info["image_path"])
        grid = build_image_grid(image_paths)
        grid_bytes_stream = io.BytesIO()
        grid.save(grid_bytes_stream, 'jpeg')

        guess_image_time = (datetime.now() - guess_image_start).total_seconds()
        bot.send_photo(
            message.chat.id, grid_bytes_stream.getvalue(), reply_to_message_id=message.message_id,
            caption=f"Clue: {clue}\nAnswer: " + results["final_answer"].strip() + "\n\n" + (
                "-------------------\n"
                f"Guessed the card (from hand) matching given clue in {guess_image_time} seconds. "
                f"Full logs can be displayed using command /log_{event_log_id}"))

    @bot.message_handler(func=lambda m: str(m.caption).startswith("/find"), content_types=['photo'])
    def guess_card_from_clue(message):
        logging.log(logging.INFO, f"Received [get card from hand by clue] request from {message.from_user.username}")
        clue = message.caption[len('/find'):].strip()

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
        
        # Write logs for detected cards
        detection_time = (datetime.now() - start_detection).total_seconds()
        logging.log(logging.INFO, f"Done detecting image for {message.from_user.username}")
        bot.send_photo(
            message.chat.id, open(det["debug_img_path"], 'rb'),
            reply_to_message_id=message.message_id, caption=(
                f"Found {len(det['cards'])} cards in {detection_time:0.1f} seconds. "
                f"Guessing which one matches the clue '{clue}'..."))

        # Build a grid of images
        image_paths = []
        for card_hash, card_info in game_state["my_cards"].items():
            image_paths.append(card_info["image_path"])
        grid = build_image_grid(image_paths)
        grid_bytes_stream = io.BytesIO()
        grid.save(grid_bytes_stream, 'jpeg')

        # Generate descriptions and clues
        guess_image_start = datetime.now()
        card_hashes, card_image_paths = det["hashes"], det["cards_paths"]
        card_infos = {h: {'image_path': p} for h, p in zip(card_hashes, card_image_paths)}
        results = choose_best_match_image_for_clue(clue, card_infos, config=config)
        
        # Dump full logs
        event_log_id = str(uuid.uuid4().hex)
        output_logs_path = os.path.join(OUTPUT_LOGS, f"{event_log_id}.yaml")
        with open(output_logs_path, 'w') as fd:
            yaml.dump(results, fd, default_flow_style=False, sort_keys=False)

        guess_image_time = (datetime.now() - guess_image_start).total_seconds()
        bot.send_photo(
            message.chat.id, grid_bytes_stream.getvalue(), reply_to_message_id=message.message_id,
            caption=f"Clue: {clue}\nAnswer: " + results["final_answer"].strip() + "\n\n" + (
                "-------------------\n"
                f"Guessed the card (from hand) matching given clue in {guess_image_time} seconds. "
                f"Full logs can be displayed using command /log_{event_log_id}"))

    # return the dixit bot
    return bot


if __name__ == "__main__":
    bot = instantiate_dixitbot()
    bot.infinity_polling()
