import styles from "./ChatMessage.module.scss";
import { IChatMessage } from "../../types/Chat";

function ChatMessage({ message, userType }: IChatMessage) {
    return (
        <div className={userType && userType === 'user' ? styles.reqAi : styles.resAi}>
            <p>{message}</p>
        </div>
    )
}

export default ChatMessage;