import sqlite3
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory


class SQLiteChatHistory():
    def __init__(self, current_user, db_path="metadata.db"):
        self.db_path = db_path
        self.current_user = current_user

    def add_message(self, message):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        if isinstance(message, HumanMessage):
            user_type = "human"
            message = message.content
        elif isinstance(message, AIMessage):
            user_type = "ai"
            message = message.content
        elif isinstance(message, SystemMessage):
            user_type = "system"
            message = message.content
        else:
            raise ValueError("Invalid message type")
        c.execute("INSERT INTO history_messages (user_id, user_type, message) VALUES (?, ?, ?)",
                  (self.current_user, user_type, message))
        conn.commit()
        conn.close()

    def messages(self, limit=15):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f"SELECT * FROM history_messages WHERE user_id = '{self.current_user}' ORDER BY id DESC LIMIT {limit}")
        resp = c.fetchall()[::-1]
        chat_history = []
        for row in resp:
            id, user_id, user_type, message, tmstmp = row
            if user_type == "human":
                chat_history.append(HumanMessage(content=message))
            elif user_type == "ai":
                chat_history.append(AIMessage(content=message))
            elif user_type == "system":
                chat_history.append(SystemMessage(content=message))
        conn.commit()
        conn.close()
        messages = ChatMessageHistory(messages=chat_history)
        return messages

    def delete_chat_history_last_n(self, n=10):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f'''
        with max_id as (select max(id) as maxid from history_messages where user_id = '{self.current_user}')
        DELETE FROM history_messages
        WHERE id BETWEEN (select maxid from max_id) - {n} AND (select maxid from max_id)
        ''')
        conn.commit()
        conn.close()
