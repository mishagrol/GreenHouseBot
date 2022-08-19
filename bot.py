import logging
import asyncio
from aiogram import Bot, Dispatcher, executor, types
import uuid
import time, json
import numpy as np

credintails = json.load(open('.token_telegram.json', 'r'))

TOKEN = credintails['token']
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends /start or /help command
    """
    await message.reply("Hi!\n\nI'm CropBot!\n\nI need photos 🥦.")

async def nn_model(photo_name: str)-> str:
    
    time.sleep(10)
    return np.random.choice(['carrot - 🥕', 'potato - 🥔', 'apple - 🍎'])

@dp.message_handler(content_types=['photo'])
async def echo(message: types.Message):
    

    photo_name = str(uuid.uuid4())+'.jpg'

    await message.photo[-1].download(photo_name)


    await bot.send_message(message.from_user.id, "Neural nets started 🥐")


    task = await asyncio.create_task(nn_model(photo_name))
    
    await message.answer(task)

# Handle /cancel command
@dp.message_handler(state='*', commands='cancel')
async def cancel_handler(message: types.Message):  

    # await state.finish()
    await message.reply('Canceled, enter /start', 
                        reply_markup=types.ReplyKeyboardRemove())
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
