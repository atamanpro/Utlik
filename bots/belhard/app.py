import asyncio
import logging
import openai
import aiohttp
import re
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher.router import Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import State, StatesGroup
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

CURRENT_USER = 'belhard'
CRM_WEBHOOK_URL = os.environ.get('CRM_WEBHOOK_URL')


DEFAULT_PROMPT = '''
Act as a friendly manager from https://belhard.academy and respond to user queries using the provided knowledge base.
If the user wants to register for a course, ask them to provide their full name, contact phone number, and email address sequentially.
'''
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", DEFAULT_PROMPT)
KNOWLEDGE_BASE = """
    Вопросы:
    1. Хочу уточнить детали курса
    2. Как записаться на курс?
    3. Сколько стоит обучение?
    4. Хотела бы поинтересоваться стоимостью обучения.
    5. Хочу обучиться с 0 и с трудоустройством
    6. Какой самый простой курс?
    7. Посоветуйте курс с нуля
    8. Какой курс выбрать?
    9. Стоимость тренинга?
    10. Сколько стоит интенсив?
    11. Как оплатить?
    12. Как внести оплату?
    13. В каком формате тренинг?
    14. Формат тренинга?
    15. Сколько длится?
    16. Какая длительность?

    Ответы:
    1. Подскажите, пожалуйста, что конкретно Вас интересует?
    2. Уточните, пожалуйста, на какой курс Вы хотели бы записаться? Для записи от Вас на следующих шагах необходимы будут ваши полные ФИО, контактный телефон и электронная почта, на которую мы вышлем письмо-приглашение на курс.
    3. Курс Войти в IT, стоит 50 руб
    4. Курс Войти в IT, стоит 50 руб
    5. Достаточно сложно посоветовать какой-либо курс без дополнительных вводных: образования, опыта работы, характера, предпочтений к новому роду деятельности и многих других факторов, которые могут влиять на выбор направления, поэтому рекомендую Вам пройти наш входной тренинг "Войти в ИТ", на котором Вы сможете лучше узнать обо всех направлениях, получить индивидуальную консультацию, а также определиться с курсом: https://belhard.academy/voitivit
    6. Достаточно сложно посоветовать какой-либо курс без дополнительных вводных: образования, опыта работы, характера, предпочтений к новому роду деятельности и многих других факторов, которые могут влиять на выбор направления, поэтому рекомендую Вам пройти наш входной тренинг "Войти в ИТ", на котором Вы сможете лучше узнать обо всех направлениях, получить индивидуальную консультацию, а также определиться с курсом: https://belhard.academy/voitivit
    7. Достаточно сложно посоветовать какой-либо курс без дополнительных вводных: образования, опыта работы, характера, предпочтений к новому роду деятельности и многих других факторов, которые могут влиять на выбор направления, поэтому рекомендую Вам пройти наш входной тренинг "Войти в ИТ", на котором Вы сможете лучше узнать обо всех направлениях, получить индивидуальную консультацию, а также определиться с курсом: https://belhard.academy/voitivit
    8. Достаточно сложно посоветовать какой-либо курс без дополнительных вводных: образования, опыта работы, характера, предпочтений к новому роду деятельности и многих других факторов, которые могут влиять на выбор направления, поэтому рекомендую Вам пройти наш входной тренинг "Войти в ИТ", на котором Вы сможете лучше узнать обо всех направлениях, получить индивидуальную консультацию, а также определиться с курсом: https://belhard.academy/voitivit
    9. Стоимость тренинга составляет 50 белорусских рублей.
    10. Стоимость тренинга составляет 50 белорусских рублей.
    11. Оплату можно вносить через ЕРИП или же любое отделение банка.
    12. Оплату можно вносить через ЕРИП или же любое отделение банка.
    13. Тренинг на данный момент проходит в 2 форматах одновременно (в очном, удаленном), поэтому самостоятельно можете выбирать формат участия.
    14. Тренинг на данный момент проходит в 2 форматах одновременно (в очном, удаленном), поэтому самостоятельно можете выбирать формат участия.
    15. Время проведения с 10:00 до 16:00.
    16. Время проведения с 10:00 до 16:00.
    """

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Telegram bot token
TOKEN = os.environ.get("TG_TOKEN")
bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()
dp.include_router(router)

# Database setup using SQLAlchemy
Base = declarative_base()
engine = create_engine('sqlite:///temp_data.db')  # SQLite database
Session = sessionmaker(bind=engine)
session = Session()

GPT_MODEL = os.environ.get('GPT_MODEL')
# OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")
if os.environ.get('CUSTOM_OPENAI_BASE_URL') != 'empty':
    openai.api_base = os.environ.get('CUSTOM_OPENAI_BASE_URL')


# Define a model to store client data
class ClientData(Base):
    __tablename__ = 'client_lead_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    surname = Column(String)
    mobile_phone = Column(String)
    email = Column(String)


Base.metadata.create_all(engine)


# FSM states
class ClientDataForm(StatesGroup):
    surname = State()
    mobile_phone = State()
    email = State()


# Labels for client data fields with mandatory information
fields = [
    ("surname", "полное ФИО", True),
    ("mobile_phone", "контактный телефон", True),
    ("email", "электронную почту", True),
]


# Function to send data to CRM
async def send_data_to_crm(data):
    lead_data = {
        'fields': {
            'TITLE': f"{data.get('surname')}",
            'LAST_NAME': data.get('surname'),
            'PHONE': [{'VALUE': data.get('mobile_phone'), 'VALUE_TYPE': 'WORK'}],
            'EMAIL': [{'VALUE': data.get('email'), 'VALUE_TYPE': 'WORK'}],
        }
    }
    async with aiohttp.ClientSession() as cur_session:
        async with cur_session.post(CRM_WEBHOOK_URL, json=lead_data) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"Error sending data to CRM: {response.status}")
                return None


# State handlers for saving data to the database
async def save_data_db(state: FSMContext):
    data = await state.get_data()
    # Remove temporary state data
    data.pop('current_field', None)
    data.pop('current_index', None)
    data.pop('collecting_data', None)
    data.pop('last_response', None)
    data.pop('registering', None)

    try:
        new_entry = ClientData(**data)
        session.add(new_entry)
        session.commit()
    except TypeError as e:
        logger.error(f"Error saving data to database: {str(e)}")
        return False

    crm_response = await send_data_to_crm(data)
    if crm_response:
        logger.info("Data successfully sent to CRM.")
    else:
        logger.error("Failed to send data to CRM.")
        return False
    return True


# Function to ask the next question in the form
async def ask_next_question(state: FSMContext, message: types.Message):
    data = await state.get_data()
    current_index = data.get('current_index', 0)

    if current_index < len(fields):
        field, label, mandatory = fields[current_index]
        prompt = f"Пожалуйста, предоставьте {label}."
        await state.update_data(current_field=field, current_index=current_index)
        await message.answer(prompt)
    else:
        success = await save_data_db(state)
        if success:
            await message.answer("Спасибо, что предоставили всю информацию! Данные успешно отправлены в CRM.")
            await state.clear()
        else:
            await message.answer("Произошла ошибка при сохранении данных. Пожалуйста, попробуйте снова.")


# Function to get a response from GPT-4
async def fetch_gpt_response(prompt):

    try:
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system",
                 "content": SYSTEM_PROMPT},
                {"role": "system", "content": KNOWLEDGE_BASE},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error generating GPT response: {str(e)}")
        return "Извините, произошла ошибка при обработке вашего запроса."


def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None


def is_valid_phone(phone):
    return re.match(r"^\+?\d{10,15}$", phone) is not None


@router.message()
async def generic_message_handler(message: types.Message, state: FSMContext):
    data = await state.get_data()

    # Check if collecting data for client form
    if data.get('collecting_data', False):
        if 'current_field' in data:
            current_field = data['current_field']
            input_value = message.text

            # Validate the input
            if current_field == 'email' and not is_valid_email(input_value):
                await message.answer("Пожалуйста, укажите действительный адрес электронной почты.")
                return
            elif current_field == 'mobile_phone' and not is_valid_phone(input_value):
                await message.answer("Пожалуйста, укажите действительный контактный номер телефона.")
                return

            await state.update_data({current_field: input_value})
            data['current_index'] += 1
            await state.update_data(current_index=data['current_index'])
            await ask_next_question(state, message)
        return

    # Check if the message is related to registration
    if data.get('registering', False):
        if 'current_field' not in data:
            await state.update_data(current_field='surname')
            await message.answer("Пожалуйста, укажите ваше полное ФИО.")
        elif data['current_field'] == 'surname':
            await state.update_data(surname=message.text, current_field='mobile_phone')
            await message.answer(
                "Спасибо за предоставленные данные. Пожалуйста, укажите ваш контактный номер телефона.")
        elif data['current_field'] == 'mobile_phone':
            if not is_valid_phone(message.text):
                await message.answer("Пожалуйста, укажите действительный контактный номер телефона.")
                return
            await state.update_data(mobile_phone=message.text, current_field='email')
            await message.answer("Спасибо за предоставленные данные. Какой у вас адрес электронной почты?")
        elif data['current_field'] == 'email':
            if not is_valid_email(message.text):
                await message.answer("Пожалуйста, укажите действительный адрес электронной почты.")
                return
            await state.update_data(email=message.text, registering=False)
            success = await save_data_db(state)
            if success:
                await message.answer("Спасибо за предоставленные данные. Ваша заявка была успешно отправлена.")
            else:
                await message.answer("Произошла ошибка при сохранении данных. Пожалуйста, попробуйте снова.")
            await state.clear()
        return

    # Get GPT response
    prompt = message.text
    response = await fetch_gpt_response(prompt)
    last_response = data.get('last_response')

    if response != last_response:
        await state.update_data(last_response=response)
        await message.answer(response)

        # Check if GPT suggests registration
        if (('для записи' in response.lower() or 'регистрации' in response.lower())
                and 'курс' in response.lower()):
            await state.update_data(registering=True)


# Start the bot
async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
