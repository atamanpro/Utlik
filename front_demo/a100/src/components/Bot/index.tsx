import { useState, useEffect, useRef } from 'react';
import { toast } from 'react-toastify';
import SpeechRecognition, {
  useSpeechRecognition,
} from 'react-speech-recognition';
import style from '../Bot/bot.module.scss';
import microphone from '../../assets/microphone.svg';
import ChatMessage from './components/ChatMessage';

const Bot: React.FC = () => {
  const { transcript, listening } = useSpeechRecognition();
  const [message, setMessage] = useState<{ message: string; userType: string }>(
    { message: '', userType: '' }
  );
  const [isWriting, setIsWriting] = useState(false);
  const [chat, setChat] = useState<{ message: string; userType: string }[]>([
    { message: 'Привет, чем я могу тебе помочь? ', userType: 'system' },
  ]);
  const [ws, setWs] = useState<WebSocket>();

  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const socket = new WebSocket(
      'ws://localhost:9999/ws?user_id=1&chat_id=1&model_id=1'
    );

    // Connection opened
    socket.addEventListener('open', () => {
      console.log('Connected to WS Server');
    });

    const getMessageResponse = (data: string) => {
      const response = JSON.parse(data);
      setIsWriting(false);
      setChat((prevChat) => [
        ...prevChat,
        { message: response.message, userType: response.user_type },
      ]);
    };

    socket.addEventListener('message', (event) => {
      getMessageResponse(event.data);
    });

    // Set WebSocket in state
    setWs(socket);

    // Clean up on unmount
    return () => {
      socket.close();
    };
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
  }, [chat]);

  useEffect(() => {
    if (listening) {
      setMessage((prevMessage) => ({ ...prevMessage, message: transcript }));
    }
  }, [transcript, listening]);

  const handleRecordVoice = () => {
    if (listening) {
      SpeechRecognition.stopListening();
    } else {
      SpeechRecognition.startListening({ language: 'ru' });
    }
  };

  async function sendMessageToBot() {
    try {
      if (message.message.trim() === '') {
        toast.warn('Please, write your message', {
          position: 'top-right',
          autoClose: 5000,
          hideProgressBar: false,
          closeOnClick: true,
          pauseOnHover: true,
          draggable: true,
          progress: undefined,
          theme: 'light',
        });
        throw new Error('Please, write your message');
      }
      setIsWriting(true);
      setChat([...chat, { message: message.message, userType: 'user' }]);
      setMessage({ message: '', userType: 'user' });

      if (ws) {
        ws.send(
          JSON.stringify({
            type: 'message',
            message: message.message,
            userType: 'user',
          })
        );
      }
    } catch (error) {
      console.error('Ошибка при выполнении запроса:', error);
    }
  }

  return (
    <div className={style.container}>
      <h2 className={style.ourBotTitle}>Chat demo</h2>

      <div className={style.ourBotGround}>
        <div className={style.botContainer}>
          <div className={style.botWrapper}>
            <div className={style.chat} ref={chatContainerRef}>
              {chat.map((msg, index) => (
                <ChatMessage
                  key={index}
                  message={msg.message}
                  userType={msg.userType}
                />
              ))}
              {isWriting && (
                <div className={style.botWriting}>
                  <p>CyberMan is typing</p>
                  <span className={style.loader}></span>
                </div>
              )}
            </div>

            <div className={style.inpForm}>
              <input
                className={style.inpText}
                type="text"
                placeholder="Type a message"
                value={message.message}
                onKeyDown={(e) =>
                  e.key === 'Enter' &&
                  sendMessageToBot()
                }
                onChange={(e) =>
                  setMessage({ ...message, message: e.target.value })
                }
              />
              <button
                disabled={isWriting}
                className={style.btnVoice}
                onClick={handleRecordVoice}
              >
                <img src={microphone} alt="micro-icon" />
              </button>
              <button
                disabled={isWriting}
                className={style.inpSubmit}
                type="submit"
                value=""
                onClick={sendMessageToBot}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Bot;
