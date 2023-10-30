from abc import *


class AbstractWebhook(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, webhook_url: str) -> None:
        """
        웹훅으로 메세지를 보내주는 클래스

        Args:
            webhook_url: 웹훅 url
        """
        raise NotImplementedError

    @abstractmethod
    def send(self, msg: str) -> bool:
        """
        웹훅을 보냅니다

        Args:
            msg: 보낼 메세지

        Returns:
            success: 전송에 성공하였는지 여부
        """
        raise NotImplementedError
