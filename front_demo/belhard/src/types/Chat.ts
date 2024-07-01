export interface IChatMessage {
    message: string;
    userType?: string;
}

export interface IChatBotProps {
    closeChat: () => void;
    isLeft?: boolean;
}

export interface IChatWidgetProps {
    isLeft?: boolean;
}