import pymsteams
from ..abstract_webhook import AbstractWebhook


class TeamsWebhook(AbstractWebhook):
    def __init__(self, webhook_url: str) -> None:
        self.webhook_connector = (
            pymsteams.connectorcard(webhook_url)
            if webhook_url.startswith("http")
            else None
        )

    def convert_html(self, text: str) -> str:
        """
        text를 html 형식으로 바꿔줍니다

        Args:
            test: html형식으로 변환할 텍스트
        """
        return f"<pre>{text}</pre>".replace("\n", "<br>")

    def send(self, msg: str) -> bool:
        if self.webhook_connector is None:
            return False

        msg = self.convert_html(msg)
        self.webhook_connector.summary(msg)
        self.webhook_connector.text(msg)
        return self.webhook_connector.send()
