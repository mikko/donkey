import logging
import os

from car.garage import Garage

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
from telegram.ext.dispatcher import run_async

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

def start(bot, update):
    send_message(bot, update.message.chat_id, "and now maximum AIttack!")

def drive_command(bot, update):
    logging.info("Drive command issued")
    models = os.listdir(models_path)
    keyboard = []
    for model in models:
        keyboard.append([InlineKeyboardButton(model, callback_data=model)])
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Choose your Markku:', reply_markup=reply_markup)

@run_async
def model_selected(bot, update):
    query = update.callback_query
    model = query.data

    message = "{} is going for maximum attack!".format(model)
    bot.edit_message_text(text=message,
                          chat_id=query.message.chat_id,
                          message_id=query.message.message_id)
    send_message(bot, None, message)
    Garage.get_instance().get_vehicle().change_model(models_path, model)

def start_bot(token):
    updater = Updater(token=token)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)
    drive_handler = CommandHandler('drive', drive_command)
    dispatcher.add_handler(drive_handler)
    drive_model_handler = CallbackQueryHandler(model_selected)
    dispatcher.add_handler(drive_model_handler)

    updater.start_polling()

def send_message(bot, chat_id, message, group=True):
    if chat_id is not None:
        bot.send_message(chat_id=chat_id, text=message)
    if "bot_group" in os.environ and group:
        group = os.environ['bot_group']
        bot.send_message(chat_id=group, text=message)

if __name__ == '__main__':
    if (not "botkey" in os.environ):
        logging.info('Environment variable botkey missing')
        exit()
    token = os.environ['botkey']
    if (not "donkey_models_path" in os.environ):
        logging.info('Environment variable donkey_models_path missing')
        exit()
    models_path = os.environ['donkey_models_path']
    start_bot(token)
    
    vehicle = Garage.get_instance().create_vehicle(use_joystick=True)
    vehicle.start()

