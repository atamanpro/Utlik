import { useState } from 'react';
import classNames from 'classnames';
import ChatBot from '../ChatBot/ChatBot';
import styles from './ChatWidget.module.scss';
import { IChatWidgetProps } from '../../types/Chat';

function ChatWidget({ isLeft = false }: IChatWidgetProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [isExiting, setIsExiting] = useState(false);

    const handleVisibility = () => {
        if (isOpen) {
            setIsExiting(true);
            setTimeout(() => {
                setIsOpen(false);
                setIsExiting(false);
            }, 200);
        } else {
            setIsOpen(true);
        }
    }


    return (
        <div className={styles.widgetOverlay}>
            <div className={classNames(styles.widgetContainer, {
                [styles.widgetContainerIn]: !isExiting,
                [styles.widgetContainerOut]: isExiting,
                [styles.widgetContainerRight]: !isLeft
            })}>
                {isOpen &&
                    <div className={classNames({
                        [styles.widgetChat]: !isExiting,
                        [styles.widgetChatExiting]: isExiting
                    })}>
                        <ChatBot closeChat={handleVisibility} isLeft={isLeft} />
                    </div>
                }
                {!isOpen && <button className={styles.widgetButton} onClick={handleVisibility}>
                    <p>Напишите нам, мы онлайн</p>
                    <span className={styles.loader}></span>
                </button>}
            </div>
        </div>
    )
}

export default ChatWidget;
