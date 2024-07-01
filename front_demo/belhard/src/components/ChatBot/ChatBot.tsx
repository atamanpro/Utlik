import { useState, useEffect, useRef, KeyboardEvent } from "react";
import { toast } from 'react-toastify';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import styles from './ChatBot.module.scss';
import microphone from './assets/icons/microphone.svg';
import ChatMessage from "../ChatMessage/ChatMessage";
import { IChatBotProps, IChatMessage } from "../../types/Chat";

function ChatBot({ closeChat, isLeft }: IChatBotProps) {
    const {
        transcript,
        listening
    } = useSpeechRecognition();
    const [message, setMessage] = useState<IChatMessage>({ message: '', userType: '' });
    const [isWriting, setIsWriting] = useState(false);
    const [chat, setChat] = useState<IChatMessage[]>([{ message: "Привет, чем я могу тебе помочь? ", userType: 'system' }]);


    const chatContainerRef = useRef<HTMLDivElement>(null);
    const inpTextRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chat]);

    useEffect(() => {
        if (listening) {
            setMessage(prevMessage => ({ ...prevMessage, message: transcript }));
        }
    }, [transcript, listening]);


    const handleRecordVoice = () => {
        if (listening) {
            SpeechRecognition.stopListening();
        } else {
            SpeechRecognition.startListening();
        }
    }

    async function sendMessageToBot() {
        try {
            if (message.message.trim() === "") {
                toast.warn('Please, write your message', {
                    position: "top-right",
                    autoClose: 5000,
                    hideProgressBar: false,
                    closeOnClick: true,
                    pauseOnHover: true,
                    draggable: true,
                    progress: undefined,
                    theme: "light"
                });
                throw new Error("Please, write your message");
            }
            setIsWriting(true);
            setChat([...chat, { message: message.message, userType: 'user' }]);
            setMessage({ message: '' });
            if (inpTextRef.current) {
                inpTextRef.current.innerText = "";
            }
            setTimeout(() => {
                setIsWriting(false);
            }, 7000)
        } catch (error) {
            console.error('Ошибка при выполнении запроса:', error);
        }
    }

    const handleInput = () => {
        if (inpTextRef.current) {
            const message = inpTextRef.current.innerText;
            setMessage({ message, userType: 'user' });
        }
    }

    const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
        if (!isWriting && (e.key === 'Enter') && !e.shiftKey) {
            e.preventDefault();
            sendMessageToBot()
        }
    }

    return (
        <div className={styles.botContainer}>
            <div className={styles.botWrapper}>
                <div className={styles.chat} ref={chatContainerRef}>
                    <div className={styles.chatMessages}>
                        {
                            chat.map((msg, index) =>
                                <ChatMessage key={index} message={msg.message} userType={msg.userType} />
                            )}
                    </div>
                    {isWriting &&
                        <div className={styles.botWriting}>
                            <p>CyberMan is typing</p>
                            <span className={styles.loader}></span>
                        </div>
                    }
                </div>

                <div className={styles.inpForm}>
                    <div className={styles.inpText}
                        contentEditable
                        placeholder="Введите свое сообщение"
                        role="textbox"
                        onKeyDown={handleKeyDown}
                        onInput={handleInput}
                        ref={inpTextRef} />
                    <div className={styles.btnsWrapper}>
                        <button disabled={isWriting} className={styles.btnVoice} onClick={handleRecordVoice}><img src={microphone} alt="micro-icon" /></button>
                        <button disabled={isWriting} className={styles.inpSubmit} type="submit" value="" onClick={sendMessageToBot} />
                    </div>
                </div>
            </div>
            <button className={isLeft ? styles.closeBtnRight : styles.closeBtnLeft} onClick={closeChat}>×</button>
        </div >
    );
}

export default ChatBot;
