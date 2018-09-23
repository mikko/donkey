import logging
import os

from car import drive

from telegram.ext import Updater, CommandHandler

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                     level=logging.INFO)


def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="and now maximum AIttack!")

def drive_command(bot, update):
    logging.info("Drive command issued")
    drive.start_drive(model_path="./todo", joystick=False)
    bot.send_message(chat_id=update.message.chat_id, text="Imma bot, Driving!")

def record(bot, update):
    logging.info("Drive command issued")
    drive.start_drive(joystick=True)
    bot.send_message(chat_id=update.message.chat_id, text="Should now start recording manual driving")

def stop(bot, update):
    logging.info("Stop command issued")
    bot.send_message(chat_id=update.message.chat_id, text="Should now stop")

def start_bot(token):
    updater = Updater(token=token)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)
    drive_handler = CommandHandler('drive', drive_command)
    dispatcher.add_handler(drive_handler)
    stop_handler = CommandHandler('stop', stop)
    dispatcher.add_handler(stop_handler)
    record_handler = CommandHandler('record', record)
    dispatcher.add_handler(record_handler)

    updater.start_polling()

if __name__ == '__main__':
    if (not "botkey" in os.environ):
        print('Environment variable donkey_config missing')
        exit()
    token = os.environ['botkey']
    start_bot(token)

