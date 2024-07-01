import style from "../../bot.module.scss";

type ChatMessageProps = {
    message: string;
    userType: string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, userType }) => {
    return (
        <div className={userType && userType === 'user' ? style.reqAi : style.resAi}>
            <p>{message}</p>
        </div>
    )
}

export default ChatMessage;