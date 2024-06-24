from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import sqlite3
import requests

app = FastAPI()


def init_metadata_db():
    with sqlite3.connect('test.db') as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        chat_id INTEGER,
        model_id INTEGER,
        source TEXT,
        role TEXT,
        message TEXT,
        tmstmp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        ''')


init_metadata_db()


def get_history_messages(user_id, chat_id, model_id, source='webchat'):
    with sqlite3.connect('test.db') as conn:
        cursor = conn.cursor()
        cursor.execute(f'''
        SELECT message FROM chat_history
        WHERE user_id={user_id} and chat_id={chat_id}
        and model_id={model_id} and source='{source}'
        ORDER BY tmstmp
        ;
        ''')
        rows = cursor.fetchall()
    messages = [row[0] for row in rows]
    return messages


def save_chat_message(user_id, chat_id, model_id, role, message, source='webchat'):
    with sqlite3.connect('test.db') as conn:
        conn.execute(f'''
        INSERT INTO chat_history (user_id, chat_id, model_id, source, role, message) 
        values ('{user_id}', '{chat_id}', '{model_id}', '{source}', '{role}', '{message}') ; ''')


active_connections_set = set()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: int, chat_id: int, model_id: int, client_name: str):

    try:
        history = get_history_messages(user_id, chat_id, model_id, source='webchat')
        await websocket.accept()
        active_connections_set.add(websocket)
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            if data["type"] == "history":
                for message in history:
                    await websocket.send_text(json.dumps({"message": message}))
            elif data["type"] == "message":
                save_chat_message(user_id, chat_id, model_id, 'user', data["message"], source='webchat')
                ### response редиректить на определенную модельку
                url = f"http://{client_name}_chat/rag_chat/"
                headers = {'Content-Type': 'application/json'}
                question_data = {"question": data["message"]}
                response = requests.post(url, json=question_data, headers=headers)
                await websocket.send_text(json.dumps({"message": response, "user_type": "system"}))
                save_chat_message(user_id, chat_id, model_id, 'system', response, source='webchat')
    except (WebSocketDisconnect) as e:
        print(f"Client disconnected + {e}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=9999, ws_ping_timeout=300)
