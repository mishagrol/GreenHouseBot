import logging
import asyncio
from aiogram import Bot, Dispatcher, executor, types
import uuid
import time, json, io
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pynvml
from utils import Network

credintails = json.load(open(".token_telegram.json", "r"))

TOKEN = credintails["token"]
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


def run_model(image: np.array) -> int:

    ### BEGIN
    cnn_model = ...

    ## END YOUR SOLUTION
    device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu")
    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    my_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((46, 46))]
    )
    image_tensor = my_transforms(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    ### BEGIN YOUR SOLUTION
    output = ...

    ### END YOUR SOLUTION
    pred = output.argmax(dim=1, keepdim=True)
    pred = pred.cpu().detach().numpy()
    del image_tensor, output, cnn_model
    torch.cuda.empty_cache()  # Clear cache
    return pred[0][0]


@dp.message_handler(commands=["start", "help"])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends /start or /help command
    """
    await message.reply("Hi!\n\nI'm CropBot!\n\nI need photos 🥦.")


@dp.message_handler(content_types=["photo", "document"])
async def echo(message: types.Message):

    file_in_io = io.BytesIO()
    if message.content_type == "photo":
        await message.photo[-1].download(destination_file=file_in_io)
    elif message.content_type == "document":
        await message.document.download(destination_file=file_in_io)
    plant_image = np.array(Image.open(io.BytesIO(file_in_io.read())))
    await bot.send_message(message.from_user.id, "Neural nets started!")
    task = run_model(image=plant_image)
    human_dict = {0: "Mint ☘️", 1: "Rausmarine 🍃"}
    await message.answer(human_dict[task])


# Handle /cancel command
@dp.message_handler(state="*", commands="cancel")
async def cancel_handler(message: types.Message):

    await message.reply(
        "Canceled, enter /start", reply_markup=types.ReplyKeyboardRemove()
    )


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
