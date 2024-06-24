import logging
from telethon import TelegramClient, events
import openai
import aiohttp
from telethon.tl.functions.messages import SendMessageRequest
from telethon.errors.rpcerrorlist import ChatAdminRequiredError, UserPrivacyRestrictedError
from telethon.tl.types import InputPeerUser
import random
import base64
import requests
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_USER = 'yoowills'
CRM_WEBHOOK_URL = os.environ.get('CRM_WEBHOOK_URL')
CRM_FIELDS_IMAGES = os.environ.get('CRM_FIELDS_IMAGES').split(',')

DEFAULT_PROMPT = '''
Act as a friendly manager from https://yoowills.by and respond to user queries. 
You can say to user write command /lead if user wants to start registration or send a lead or /info for more information"
'''
SYSTEM_PROMPT = os.environ.get('SYSTEM_PROMPT', DEFAULT_PROMPT)

# OpenAI API key

GPT_MODEL = os.environ.get('GPT_MODEL', 'gpt-4o')

# OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")
if os.environ.get('CUSTOM_OPENAI_BASE_URL') != 'empty':
    openai.api_base = os.environ.get('CUSTOM_OPENAI_BASE_URL')

# Your API ID and Hash from https://my.telegram.org
api_id = int(os.environ.get('TG_API_ID'))
api_hash = os.environ.get('TG_API_HASH')
phone = os.environ.get('TG_USERBOT_PHONE')


# Create the client and connect
client = TelegramClient('userbot_session', api_id, api_hash)

# User states for managing data collection
user_states = {}


# Functions to get and set user state
def get_user_state(user_id):
    return user_states.get(user_id, {})


def set_user_state(user_id, state):
    user_states[user_id] = state


def encode_image_to_base64(file_path: str) -> str:
    if file_path is None or not os.path.isfile(file_path):
        logger.error("Invalid file path")
        return ""

    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading local file: {e}")
        return ""


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


# Function to send data to Bitrix24
async def send_data_to_bitrix(data):
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
            f"{CRM_FIELDS_IMAGES[0]}": {"fileData": ["scan1.jpg", data.get('passport_main_page')]},
            f"{CRM_FIELDS_IMAGES[1]}": {"fileData": ["scan2.jpg", data.get('passport_30_31_page')]},
            f"{CRM_FIELDS_IMAGES[2]}": {"fileData": ["scan3.jpg", data.get('passport_registration_page')]}
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(CRM_WEBHOOK_URL, json=lead_data) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"Error sending data to Bitrix24: {response.status}")
                return None


async def process_next_field(event, state, index):
    if index < len(fields):
        field, label, mandatory, is_image = fields[index]
        state['current_field'] = field
        state['current_index'] = index
        state['is_image'] = is_image
        await event.respond(f"Пожалуйста, предоставьте {label}")
    else:
        success = await send_data_to_bitrix(state)
        if success:
            await event.respond("Спасибо, что предоставили всю информацию! Данные успешно отправлены в Bitrix24.")
            await continue_conversation(event)
        else:
            await event.respond("Произошла ошибка при отправке данных. Пожалуйста, попробуйте снова.")


async def continue_conversation(event):
    initial_prompt = "Чем еще я могу вам помочь?"
    response = await fetch_gpt_response(initial_prompt)
    await event.respond(response)


async def fetch_gpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error generating GPT response: {str(e)}")
        return "Извините, произошла ошибка при обработке вашего запроса.Повторите еще раз."


@client.on(events.NewMessage(pattern='/start'))
async def start_handler(event):
    initial_prompt = "Привет! Чем я могу вам помочь? Вы можете использовать команду /lead для начала регистрации или /info для получения информации."
    await event.respond(initial_prompt)


@client.on(events.NewMessage(pattern='/lead'))
async def lead_handler(event):
    state = get_user_state(event.sender_id)
    state['current_index'] = 0
    set_user_state(event.sender_id, state)
    await process_next_field(event, state, 0)


@client.on(events.NewMessage(pattern='/message (.*)'))
async def message_handler(event):
    phone_number = event.pattern_match.group(1)
    try:
        user = await client.get_entity(phone_number)
        if hasattr(user, 'phone'):
            input_peer_user = InputPeerUser(user_id=user.id, access_hash=user.access_hash)
            random_id = random.randint(1, 1018)  # Generate a random long ID for the message
            await client(SendMessageRequest(
                peer=input_peer_user,
                message="Добрый день, Вас приветствует Yoowills! Как я могу вам помочь? Если вам нужна справочная информация напишите /info или если вы хотите оставить заявку напишите /lead",
                random_id=random_id
            ))
            await event.respond(f"Сообщение отправлено пользователю с номером {phone_number}.")
        else:
            await event.respond(f"Не удалось найти пользователя с номером {phone_number}.")
    except ChatAdminRequiredError:
        await event.respond("Ошибка: Необходимы права администратора для отправки сообщений в этот чат.")
    except UserPrivacyRestrictedError:
        await event.respond(f"Ошибка: Пользователь с номером {phone_number} ограничил получение сообщений от незнакомцев.")
    except Exception as e:
        await event.respond(f"Не удалось отправить сообщение пользователю с номером {phone_number}. Ошибка: {str(e)}")


@client.on(events.NewMessage())
async def generic_handler(event):
    if event.message.message.startswith('/'):
        return  # Ignore other commands

    state = get_user_state(event.sender_id)
    if 'current_field' in state and 'current_index' in state:
        current_field = state['current_field']
        current_index = state['current_index']
        is_image = state['is_image']

        if is_image and event.message.photo:
            # Download the photo
            file = await client.download_media(event.message.photo)
            print(f"file's dtype is {type(file)}")
            # Encode the photo to base64
            base64_image = encode_image_to_base64(file)
            if base64_image:
                state[current_field] = base64_image
                os.remove(file)
            else:
                await event.respond("Ошибка при конвертации изображения.")
        elif is_image and event.message.document:
            # Download the document
            file = await client.download_media(event.message.document)
            print(f"file's dtype is {type(file)}")
            # Encode the photo to base64
            base64_image = encode_image_to_base64(file)
            if base64_image:
                state[current_field] = base64_image
                os.remove(file)
        elif not is_image:
            state[current_field] = event.message.message
        else:
            await event.respond("Пожалуйста, отправьте изображение.")

        state['current_index'] = current_index + 1
        set_user_state(event.sender_id, state)
        await process_next_field(event, state, current_index + 1)
    else:
        prompt = event.message.message
        response = await fetch_gpt_response(prompt)
        await event.respond(response)


async def main():
    await client.connect()
    if not await client.is_user_authorized():
        await client.send_code_request(phone)
        await client.sign_in(phone, input('Enter the code: '))
    logger.info("Client started")
    await client.run_until_disconnected()

if __name__ == '__main__':
    try:
        client.loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Client stopped")
