import os
import asyncio
import logging
import openai
import aiohttp
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.dispatcher.router import Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import base64
import requests


CURRENT_USER = 'yoowills'
CRM_WEBHOOK_URL = os.environ.get('CRM_WEBHOOK_URL')
CRM_FIELDS_IMAGES = os.environ.get('CRM_FIELDS_IMAGES').split(',')

DEFAULT_PROMPT = '''
Act as a friendly manager from https://yoowills.by and respond to user queries.
If user asks what you can do, you can say to write command /lead to start registration or
command /info for information
'''

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Telegram bot token
TOKEN = os.environ.get('TG_TOKEN')
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
    """
    Создаем модель базы данных для хранения всех запрашиваемых данных заказчика, которые будут необходимы при
    оформлении лизинга. Все поля имеют строковый тип данных. Последние три поля предназначены для получения фото (скана)
    указанных страниц паспорта либо ВНЖ и хранят путь к соответствующим файлам.
    """
    __tablename__ = os.environ.get('client_lead_data')
    id = Column(Integer, primary_key=True, autoincrement=True)
    surname = Column(String)
    first_name = Column(String)
    patronymic = Column(String)
    mobile_phone = Column(String)
    additional_phone = Column(String)
    email = Column(String)
    birth_date = Column(String)
    personal_number = Column(String)
    leasing_item = Column(String)
    leasing_cost = Column(String)
    leasing_quantity = Column(String)
    leasing_advance = Column(String)
    leasing_currency = Column(String)
    leasing_duration = Column(String)
    place_of_birth = Column(String)
    gender = Column(String)
    criminal_record = Column(String)
    document = Column(String)
    citizenship = Column(String)
    series = Column(String)
    number = Column(String)
    issue_date = Column(String)
    expiration_date = Column(String)
    issued_by = Column(String)
    registration_index = Column(String)
    registration_country = Column(String)
    registration_region = Column(String)
    registration_district = Column(String)
    registration_locality = Column(String)
    registration_street = Column(String)
    registration_house = Column(String)
    registration_building = Column(String)
    registration_apartment = Column(String)
    residence_index = Column(String)
    residence_country = Column(String)
    residence_region = Column(String)
    residence_district = Column(String)
    residence_locality = Column(String)
    residence_street = Column(String)
    residence_house = Column(String)
    residence_building = Column(String)
    residence_apartment = Column(String)
    workplace_name = Column(String)
    position = Column(String)
    work_experience = Column(String)
    income = Column(String)
    hr_phone = Column(String)
    marital_status = Column(String)
    dependents_count = Column(String)
    education = Column(String)
    military_duty = Column(String)
    relative_surname = Column(String)
    relative_first_name = Column(String)
    relative_patronymic = Column(String)
    relative_phone = Column(String)
    passport_main_page = Column(String)
    passport_30_31_page = Column(String)
    passport_registration_page = Column(String)


Base.metadata.create_all(engine)

# Labels for client data fields with mandatory information
fields = [
    ("surname", "Фамилия*", True, False),
    ("first_name", "Имя*", True, False),
    ("patronymic", "Отчество*", True, False),
    ("mobile_phone", "Мобильный телефон*", True, False),
    ("additional_phone", "Дополнительный телефон", False, False),
    ("email", "Email", False, False),
    ("birth_date", "Дата рождения*", True, False),
    ("personal_number", "Личный номер*", True, False),
    ("leasing_item", "Наименование предмета лизинга*", True, False),
    ("leasing_cost", "Стоимость*", True, False),
    ("leasing_quantity", "Количество*", True, False),
    ("leasing_advance", "Аванс", False, False),
    ("leasing_currency", "Валюта договора*", True, False),
    ("leasing_duration", "Срок договора*", True, False),
    ("place_of_birth", "Место рождения", False, False),
    ("gender", "Пол*", True, False),
    ("criminal_record", "Наличие судимости*", True, False),
    ("document", "Документ, удостоверяющий личность (паспорт, ВНЖ)", False, False),
    ("citizenship", "Гражданство", False, False),
    ("series", "Серия (паспорта, ВНЖ)*", True, False),
    ("number", "Номер (паспорта, ВНЖ)*", True, False),
    ("issue_date", "Дата Выдачи (паспорта, ВНЖ)*", True, False),
    ("expiration_date", "Срок действия (паспорта, ВНЖ)*", True, False),
    ("issued_by", "Кем выдан (паспорт, ВНЖ)*", True, False),
    ("registration_index", "Индекс по прописке", False, False),
    ("registration_country", "Страна по прописке*", True, False),
    ("registration_region", "Область по прописке", False, False),
    ("registration_district", "Район по прописке", False, False),
    ("registration_locality", "Населенный пункт по прописке*", True, False),
    ("registration_street", "Улица по прописке*", True, False),
    ("registration_house", "Дом по прописке*", True, False),
    ("registration_building", "Строение, корпус по прописке", False, False),
    ("registration_apartment", "Квартира по прописке", False, False),
    ("residence_index", "Индекс фактического места жительства", False, False),
    ("residence_country", "Страна фактического места жительства*", True, False),
    ("residence_region", "Область фактического места жительства", False, False),
    ("residence_district", "Район фактического места жительства", False, False),
    ("residence_locality", "Населенный пункт фактического места жительства*", True, False),
    ("residence_street", "Улица фактического места жительства*", True, False),
    ("residence_house", "Дом фактического места жительства*", True, False),
    ("residence_building", "Строение, корпус фактического места жительства", False, False),
    ("residence_apartment", "Квартира фактического места жительства", False, False),
    ("workplace_name", "Наименование организации, в которой работаете в данный момент*", True, False),
    ("position", "Должность*", True, False),
    ("work_experience", "Стаж*", True, False),
    ("income", "Доход*", True, False),
    ("hr_phone", "Телефон отдела кадров или бухгалтерии*", True, False),
    ("marital_status", "Семейное положение*", True, False),
    ("dependents_count", "Количество иждивенцев", False, False),
    ("education", "Образование*", True, False),
    ("military_duty", "Воинская обязанность*", True, False),
    ("relative_surname", "Фамилия близкого родственника, либо супруга/супруги*", True, False),
    ("relative_first_name", "Имя близкого родственника, либо супруга/супруги*", True, False),
    ("relative_patronymic", "Отчество близкого родственника, либо супруга/супруги*", True, False),
    ("relative_phone", "Телефон близкого родственника, либо супруга/супруги*", True, False),
    ("passport_main_page", "Главный разворот (паспорта, ВНЖ)*", True, True),
    ("passport_30_31_page", "Разворот 30-31 (паспорта, ВНЖ)*", True, True),
    ("passport_registration_page", "Разворот с регистрацией (паспорта, ВНЖ)*", True, True),
]


def encode_image_tobase64(img_path: str = None) -> str:
    """
    Функция кодирования файла изображения в большой бинарный объект (BLOB), который необходим для передачи изображения
    из телеграм-бота в CRM-систему BITRIX24 с помощью веб-хука. На вход функция получает путь к файлу изображения,
    который необходимо передать. На выходе функция возвращает большой бинарный объект в виде строки, которую легко
    передаем в CRM-систему BITRIX24.
    """
    if img_path is not None:
        response = requests.get(img_path)
        image_data = response.content
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return base64_image
    else:
        return None


# Функция отправки данных в Bitrix24
async def send_data_to_bitrix(data):
    """
    Функция передает все значения полей, полученные от заказчика, определяет нужные значения в подготовленные поля
    такие как "Имя", "Фамилия" и т.д., а также формирует входящий ЛИД в CRM-системе BITRIX24.
    """

    lead_data = {
        'fields': {
            'TITLE': f"{data.get('surname')} {data.get('first_name')} {data.get('patronymic')}",
            'NAME': data.get('first_name'),
            'LAST_NAME': data.get('surname'),
            'SECOND_NAME': data.get('patronymic'),
            'PHONE': [
                {'VALUE': data.get('mobile_phone'), 'VALUE_TYPE': 'WORK'},
                {'VALUE': data.get('additional_phone'), 'VALUE_TYPE': 'HOME'}
            ],
            'EMAIL': [{'VALUE': data.get('email'), 'VALUE_TYPE': 'WORK'}],
            'COMMENTS': (
                f"Дата рождения: {data.get('birth_date')}\n"
                f"Личный номер: {data.get('personal_number')}\n"
                f"Место рождения: {data.get('place_of_birth')}\n"
                f"Пол: {data.get('gender')}\n"
                f"Судимость: {data.get('criminal_record')}\n"
                f"Документ: {data.get('document')}\n"
                f"Гражданство: {data.get('citizenship')}\n"
                f"Серия: {data.get('series')}\n"
                f"Номер: {data.get('number')}\n"
                f"Дата выдачи: {data.get('issue_date')}\n"
                f"Срок действия: {data.get('expiration_date')}\n"
                f"Кем выдан: {data.get('issued_by')}\n"
                f"Наименование предмета лизинга: {data.get('leasing_item')}\n"
                f"Стоимость: {data.get('leasing_cost')}\n"
                f"Количество: {data.get('leasing_quantity')}\n"
                f"Аванс: {data.get('leasing_advance')}\n"
                f"Валюта договора: {data.get('leasing_currency')}\n"
                f"Срок договора: {data.get('leasing_duration')}\n"
                f"Наименование организации: {data.get('workplace_name')}\n"
                f"Должность: {data.get('position')}\n"
                f"Стаж: {data.get('work_experience')}\n"
                f"Доход: {data.get('income')}\n"
                f"Телефон отдела кадров или бухгалтерии: {data.get('hr_phone')}\n"
                f"Семейное положение: {data.get('marital_status')}\n"
                f"Количество иждивенцев: {data.get('dependents_count')}\n"
                f"Образование: {data.get('education')}\n"
                f"Воинская обязанность: {data.get('military_duty')}\n"
                f"Фамилия родственника: {data.get('relative_surname')}\n"
                f"Имя родственника: {data.get('relative_first_name')}\n"
                f"Отчество родственника: {data.get('relative_patronymic')}\n"
                f"Телефон родственника: {data.get('relative_phone')}\n"
                f"Дата рождения: {data.get('birth_date')}\n"
                f"issue_date: {data.get('issue_date')}\n"
                f"expiration_date: {data.get('expiration_date')}\n"
                f"registration_index: {data.get('registration_index')}\n"
                f"registration_country: {data.get('registration_country')}\n"
                f"registration_region: {data.get('registration_region')}\n"
                f"registration_district: {data.get('registration_district')}\n"
                f"registration_locality: {data.get('registration_locality')}\n"
                f"registration_street: {data.get('registration_street')}\n"
                f"registration_house: {data.get('registration_house')}\n"
                f"registration_building: {data.get('registration_building')}\n"
                f"registration_apartment: {data.get('registration_apartment')}\n"
                f"residence_index: {data.get('residence_index')}\n"
                f"residence_country: {data.get('residence_country')}\n"
                f"residence_region: {data.get('residence_region')}\n"
                f"residence_district: {data.get('residence_district')}\n"
                f"residence_locality: {data.get('residence_locality')}\n"
                f"residence_street: {data.get('residence_street')}\n"
                f"residence_house: {data.get('residence_house')}\n"
                f"residence_building: {data.get('residence_building')}\n"
                f"residence_apartment: {data.get('residence_apartment')}\n"
            ),
            f"{CRM_FIELDS_IMAGES[0]}": {"fileData": ["scan1.jpg", encode_image_tobase64(data.get('passport_main_page'))]},  # Главный разворот документа
            f"{CRM_FIELDS_IMAGES[1]}": {"fileData": ["scan2.jpg", encode_image_tobase64(data.get('passport_30_31_page'))]},  # Разворот документа 30-31
            f"{CRM_FIELDS_IMAGES[2]}": {"fileData": ["scan3.jpg", encode_image_tobase64(data.get('passport_registration_page'))]}, #Разворот документа с регистрацией
        }
    }

    async with aiohttp.ClientSession() as cur_session:
        """
        Функция установления соединения телеграм_бота с CRM-системой BITRIX24. При удачном соединении происходит 
        передача информации. При невозможности установления соединения выдается ошибка с ее кодом.
        """
        async with cur_session.post(CRM_WEBHOOK_URL, json=lead_data) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"Error sending data to Bitrix24: {response.status}")
                return None


# Обработчики состояний для сохранения данных в базе
async def save_data_db(state: FSMContext):
    data = await state.get_data()
    # Remove temporary state data
    data.pop('current_field', None)
    data.pop('current_index', None)
    data.pop('is_image', None)

    try:
        new_entry = ClientData(**data)
        session.add(new_entry)
        session.commit()
    except TypeError as e:
        logger.error(f"Error saving data to database: {str(e)}")
        return False

    bitrix_response = await send_data_to_bitrix(data)
    if bitrix_response:
        logger.info("Data successfully sent to Bitrix24.")
    else:
        logger.error("Failed to send data to Bitrix24.")
        return False
    return True


@router.message(Command("start"))
async def start_message(message: Message):
    initial_prompt = "Привет! Чем я могу Вам помочь?"
    await message.answer(initial_prompt)
    response = await fetch_gpt_response(initial_prompt)
    await message.answer(response)


# Общий обработчик для всех состояний
@router.message(Command("lead"))
async def start_lead(message: Message, state: FSMContext):
    await message.answer("Начнем сбор вашей информации для заявки.")
    await process_next_field(message, state, 0)


# Обработчик для каждого состояния
async def process_next_field(message: Message, state: FSMContext, index: int):
    if index < len(fields):
        field, label, mandatory, is_image = fields[index]
        await state.update_data(current_field=field, current_index=index, is_image=is_image)
        await message.answer(f"Пожалуйста, предоставьте {label}")
    else:
        success = await save_data_db(state)
        if success:
            await message.answer("Спасибо, что предоставили всю информацию! Данные успешно отправлены в Bitrix24.")
            await continue_conversation(message)
        else:
            await message.answer("Произошла ошибка при сохранении данных. Пожалуйста, попробуйте снова.")


async def continue_conversation(message: Message):
    initial_prompt = "Чем еще я могу вам помочь?"
    response = await fetch_gpt_response(initial_prompt)
    await message.answer(response)


@router.message(Command("info"))
async def info_handler(message: Message):
    await message.answer("Это команда-заглушка для /info. Здесь будет информация.")


# Функция получения ответа от GPT
async def fetch_gpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system",
                 "content": DEFAULT_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error generating GPT response: {str(e)}")
        return "Извините, произошла ошибка при обработке вашего запроса с помощью GPT"


@router.message()
async def generic_message_handler(message: Message, state: FSMContext):
    data = await state.get_data()
    if 'current_field' in data and 'current_index' in data:
        current_field = data.get('current_field')
        current_index = data.get('current_index')
        is_image = data.get('is_image', False)
        if is_image and message.photo:
            photo = message.photo[-1]  # Get the highest resolution photo
            file_id = photo.file_id
            file = await bot.get_file(file_id)
            file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file.file_path}"
            await state.update_data({current_field: file_url})
        elif is_image and message.document:
            doc = message.document
            file_id = doc.file_id
            file = await bot.get_file(file_id)
            file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file.file_path}"
            if file_url.split('.')[-1] in ('jpg', 'jpeg', 'png'):
                await state.update_data({current_field: file_url})
            else:
                await message.answer("Пожалуйста, отправьте изображение в формате: jpg, jpeg, png")
        elif not is_image:
            await state.update_data({current_field: message.text})
        else:
            await message.answer("Пожалуйста, отправьте изображение в формате: jpg, jpeg, png")

        await process_next_field(message, state, current_index + 1)
    else:
        if message.text.startswith('/'):
            return  # Ignore other commands

        prompt = message.text
        response = await fetch_gpt_response(prompt)
        await message.answer(response)


# Запуск бота
async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
